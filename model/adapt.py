"""Stage 2: adapt the synthetic ConvNeXt model to real drawings with prototype imprinting and fine-tuning."""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ogham_convnext import (
    IMAGE_EXTENSIONS, LETTERS, REPO_ROOT, OghamDataset, build_optimizer, build_scheduler, get_device,
    load_model, predict_batches, save_checkpoint, support_samples, synthetic_samples, train_epoch,
)


def imprint(model, paths, letters, device):
    dataset = OghamDataset(paths, letters, model.mean, model.std, cache=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    covered = model.imprint(loader, device)
    print(f"Imprinted {covered[0]} direction and {covered[1]} count prototypes from {len(paths)} images", flush=True)


def pseudo_label(model, dirs, device, threshold, workers):
    paths = sorted(p for d in dirs for p in Path(d).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        return [], []
    # Labels are placeholders; only the predictions are used.
    dataset = OghamDataset(paths, ["B"] * len(paths), model.mean, model.std)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)
    probs, _, _ = predict_batches(model, loader, device)
    confidence, index = probs.max(1)
    keep = (confidence >= threshold).nonzero().flatten().tolist()
    return [paths[i] for i in keep], [LETTERS[index[i]] for i in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "convnext_synthetic.pt"))
    parser.add_argument("--support", nargs="+",
                        default=[str(REPO_ROOT / "data" / "support" / "pen-marker"),
                                 str(REPO_ROOT / "data" / "support" / "whiteboard")])
    parser.add_argument("--unlabelled", nargs="+", default=[])
    parser.add_argument("--pseudo-threshold", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-4 / 3)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--data", default=str(REPO_ROOT / "output"))
    parser.add_argument("--out", default=str(REPO_ROOT / "models" / "convnext_adapted.pt"))
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"Device: {device}", flush=True)
    model, source = load_model(args.model, device)
    model.train()

    support_dirs = [str(Path(d).resolve()) for d in args.support]
    sup_paths, sup_letters = support_samples(support_dirs)
    print(f"Support images: {len(sup_paths)} from {len(support_dirs)} directories", flush=True)
    imprint(model, sup_paths, sup_letters, device)

    pseudo_paths, pseudo_letters = [], []
    if args.unlabelled:
        pseudo_paths, pseudo_letters = pseudo_label(model, args.unlabelled, device, args.pseudo_threshold, args.workers)
    print(f"Pseudo-labelled images: {len(pseudo_paths)}", flush=True)

    syn_paths, syn_letters = synthetic_samples(args.data)
    groups = [(OghamDataset(sup_paths, sup_letters, model.mean, model.std, augment=True, cache=True), 0.4),
              (OghamDataset(pseudo_paths, pseudo_letters, model.mean, model.std, augment=True, cache=True), 0.2),
              (OghamDataset(syn_paths, syn_letters, model.mean, model.std, augment=True), 0.4)]
    groups = [(ds, w) for ds, w in groups if len(ds) > 0]
    total_weight = sum(w for _, w in groups)
    sample_weights = torch.cat([torch.full((len(ds),), w / total_weight / len(ds), dtype=torch.double)
                                for ds, w in groups])
    print("Sampling mix: " + ", ".join(f"{len(ds)} images at {w / total_weight:.0%}" for ds, w in groups), flush=True)

    sampler = WeightedRandomSampler(sample_weights, args.samples_per_epoch, replacement=True)
    loader = DataLoader(ConcatDataset([ds for ds, _ in groups]), batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, persistent_workers=args.workers > 0)

    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr)
    steps_per_epoch = len(loader)
    scheduler = build_scheduler(optimizer, round(args.warmup_epochs * steps_per_epoch), args.epochs * steps_per_epoch)

    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_epoch(model, loader, optimizer, scheduler, device, desc=f"adapt epoch {epoch}/{args.epochs}")
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"Epoch {epoch}/{args.epochs} loss {loss:.4f} ({time.time() - start:.0f}s)", flush=True)

    imprint(model, sup_paths, sup_letters, device)
    save_checkpoint(args.out, model, args=vars(args), history=history, support_dirs=support_dirs,
                    pseudo_labelled=len(pseudo_paths), source_checkpoint=args.model,
                    source_git_commit=source.get("git_commit"))
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
