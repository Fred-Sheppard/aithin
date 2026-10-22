"""Stage 1: train the ConvNeXt Ogham model on the synthetic set."""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ogham_convnext import (
    REPO_ROOT, OghamConvNeXt, OghamDataset, accuracies, build_optimizer, build_scheduler,
    get_device, predict_batches, save_checkpoint, synthetic_samples, train_epoch,
)


def stratified_split(letters, val_fraction, seed):
    rng = np.random.default_rng(seed)
    letters = np.asarray(letters)
    train_idx, val_idx = [], []
    for letter in sorted(set(letters)):
        idx = rng.permutation(np.nonzero(letters == letter)[0])
        n_val = int(round(len(idx) * val_fraction))
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    return sorted(train_idx), sorted(val_idx)


def evaluate_split(model, paths, letters, device, augment, args):
    dataset = OghamDataset(paths, letters, model.mean, model.std, augment=augment, seed=1 if augment else None)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=args.workers,
                        persistent_workers=False)
    return accuracies(*predict_batches(model, loader, device, desc=f"val {'augmented' if augment else 'clean'}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Train on N random samples and validate on 500")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--data", default=str(REPO_ROOT / "output"))
    parser.add_argument("--out", default=str(REPO_ROOT / "models" / "convnext_synthetic.pt"))
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print(f"Device: {device}", flush=True)

    paths, letters = synthetic_samples(args.data)
    train_idx, val_idx = stratified_split(letters, 0.1, seed=0)
    if args.limit:
        rng = np.random.default_rng(0)
        train_idx = sorted(rng.choice(train_idx, size=min(args.limit, len(train_idx)), replace=False).tolist())
        val_idx = sorted(rng.choice(val_idx, size=min(500, len(val_idx)), replace=False).tolist())
    train_paths, train_letters = [paths[i] for i in train_idx], [letters[i] for i in train_idx]
    val_paths, val_letters = [paths[i] for i in val_idx], [letters[i] for i in val_idx]
    print(f"Train {len(train_paths)}, validation {len(val_paths)}", flush=True)

    model = OghamConvNeXt(pretrained=True).to(device)
    train_set = OghamDataset(train_paths, train_letters, model.mean, model.std, augment=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True, persistent_workers=args.workers > 0)

    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, round(args.warmup_epochs * steps_per_epoch), args.epochs * steps_per_epoch)

    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, desc=f"epoch {epoch}/{args.epochs}")
        clean = evaluate_split(model, val_paths, val_letters, device, False, args)
        augmented = evaluate_split(model, val_paths, val_letters, device, True, args)
        history.append({"epoch": epoch, "train_loss": loss, "clean": clean, "augmented": augmented})
        print(f"Epoch {epoch}/{args.epochs} loss {loss:.4f} ({time.time() - start:.0f}s)", flush=True)
        for name, m in (("clean", clean), ("augmented", augmented)):
            print(f"  val {name:9s} direction {m['direction_acc']:.4f} count {m['count_acc']:.4f} "
                  f"letter {m['letter_acc']:.4f}", flush=True)
        save_checkpoint(args.out, model, args=vars(args), history=history)

    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
