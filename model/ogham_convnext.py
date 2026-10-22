"""Pretrained ConvNeXt Ogham letter recogniser: letters, preprocessing, augmentation, model and prediction."""
import math
import subprocess
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from ogham_data import (  # noqa: F401  re-exported for train.py and adapt.py
    DIRECTIONS, IMAGE_EXTENSIONS, LETTER_TABLE, LETTER_TO_INDEX, LETTERS, REPO_ROOT, csv_samples, find_image,
    letter_count_index, letter_direction, load_grayscale, progress, support_samples, synthetic_samples,
)

BACKBONE = "convnext_tiny.fb_in22k"
IMAGE_SIZE = 224


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True).strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------- image helpers

def border_median(img):
    a = np.asarray(img)
    border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    return int(np.median(border))


def resize(img, size):
    """Box filter when shrinking, bilinear when enlarging."""
    if img.size == size:
        return img
    shrinking = size[0] * size[1] < img.size[0] * img.size[1]
    return img.resize(size, Image.BOX if shrinking else Image.BILINEAR)


def pad_to_square(img, fill=None):
    w, h = img.size
    if w == h:
        return img
    if fill is None:
        fill = border_median(img)
    side = max(w, h)
    canvas = Image.new("L", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def to_unit_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)


def normalise(x, mean, std):
    """[0, 1] tensor of shape (1, H, W) -> 3-channel normalised tensor."""
    x = x.expand(3, -1, -1)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def preprocess(img, mean, std):
    img = pad_to_square(img)
    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return normalise(to_unit_tensor(img), mean, std)


def shrink_to_fit(img, limit=600):
    w, h = img.size
    if max(w, h) <= limit:
        return img
    scale = limit / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BOX)


# ---------------------------------------------------------------- augmentation

def crop_to_ink(img, rng):
    bg = border_median(img)
    a = np.asarray(img)
    ys, xs = np.nonzero(a < min(128, bg - 40))
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    left, top, right, bottom = (int(round(m * side)) for m in rng.uniform(0.08, 0.35, size=4))
    box = (x0 - left, y0 - top, x1 + right, y1 + bottom)
    canvas = Image.new("L", (box[2] - box[0], box[3] - box[1]), bg)
    canvas.paste(img, (-box[0], -box[1]))
    return canvas


def thicken(img, rng):
    choice = rng.integers(3)
    if choice == 0:
        return img
    k = max(3, round(3 * max(img.size) / 300))
    if k % 2 == 0:
        k += 1
    if choice == 2:
        k += 2
    return img.filter(ImageFilter.MinFilter(k))


def random_aspect(img, rng):
    r = math.exp(rng.uniform(math.log(0.75), math.log(1.33)))
    w, h = img.size
    return resize(img, (max(1, round(w * r)), h))


def geometric_warp(x, rng):
    """x: (1, H, W) in [0, 1]. Areas sampled outside the image take the background colour."""
    size = x.shape[-1]
    rot = math.radians(rng.uniform(-12, 12))
    shear = math.radians(rng.uniform(-8, 8))
    scale = 1.0 / rng.uniform(0.85, 1.0)
    tx, ty = rng.uniform(-0.06, 0.06, size=2)
    rotation = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    shearing = np.array([[1.0, math.tan(shear)], [0.0, 1.0]])
    theta = np.zeros((2, 3), dtype=np.float32)
    theta[:, :2] = scale * rotation @ shearing
    theta[:, 2] = (tx, ty)
    grid = F.affine_grid(torch.from_numpy(theta).unsqueeze(0), (1, 1, size, size), align_corners=False)
    if rng.random() < 0.3:
        std_pixels = rng.uniform(1, 3)
        noise = torch.from_numpy(rng.normal(0, std_pixels, size=(1, 2, 6, 6)).astype(np.float32))
        offset = F.interpolate(noise, size=(size, size), mode="bicubic", align_corners=False)
        grid = grid + offset.permute(0, 2, 3, 1) * (2.0 / size)
    bg = float(np.median(np.concatenate([x[0, 0].numpy(), x[0, -1].numpy(), x[0, :, 0].numpy(), x[0, :, -1].numpy()])))
    warped = F.grid_sample((x - bg).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped[0] + bg


def gaussian_blur(x, sigma):
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    x = x.unsqueeze(0)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return x[0]


def appearance(x, rng):
    size = x.shape[-1]
    ink = 1.0 - x
    if rng.random() < 0.3:
        field = torch.from_numpy(rng.uniform(0, 1, size=(1, 1, 16, 16)).astype(np.float32))
        field = F.interpolate(field, size=(size, size), mode="bilinear", align_corners=False)[0]
        ink = ink * (0.6 + 0.4 * field)
    paper = rng.uniform(0.75, 1.0)
    x = paper - ink * rng.uniform(0.5, 1.0) * paper
    if rng.random() < 0.5:
        gx, gy = rng.uniform(-0.15, 0.15, size=2)
        coords = torch.linspace(-1, 1, size)
        x = x + gx * coords.view(1, 1, -1) + gy * coords.view(1, -1, 1)
    if rng.random() < 0.5:
        x = gaussian_blur(x, rng.uniform(0.3, 1.5))
    x = x + torch.from_numpy(rng.normal(0, rng.uniform(0, 0.05), size=tuple(x.shape)).astype(np.float32))
    return x.clamp(0, 1)


def augment(img, mean, std, rng):
    img = shrink_to_fit(img)
    if rng.random() < 0.6:
        img = crop_to_ink(img, rng)
    img = thicken(img, rng)
    img = random_aspect(img, rng)
    img = resize(pad_to_square(img), (IMAGE_SIZE, IMAGE_SIZE))
    x = geometric_warp(to_unit_tensor(img), rng)
    x = appearance(x, rng)
    return normalise(x, mean, std)


# ---------------------------------------------------------------- datasets

class OghamDataset(Dataset):
    """Labelled images. With augment=True each sample is freshly augmented; with
    a fixed seed the augmentation of sample i is the same every time."""

    def __init__(self, paths, letters, mean, std, augment=False, seed=None, cache=False):
        self.paths = [str(p) for p in paths]
        self.letters = list(letters)
        self.mean, self.std = tuple(mean), tuple(std)
        self.augment = augment
        self.seed = seed
        # Large photos are shrunk once up front; augmentation shrinks to 600 px anyway.
        self.cache = [shrink_to_fit(load_grayscale(p)) for p in self.paths] if cache else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.cache[i] if self.cache is not None else load_grayscale(self.paths[i])
        if self.augment:
            rng = np.random.default_rng(None if self.seed is None else (self.seed, i))
            x = augment(img, self.mean, self.std, rng)
        else:
            x = preprocess(img, self.mean, self.std)
        letter = self.letters[i]
        return x, letter_direction(letter), letter_count_index(letter)


# ---------------------------------------------------------------- model

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=16.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, in_features) * 0.01)

    def forward(self, x):
        return self.scale * F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t()


class Head(nn.Module):
    def __init__(self, in_features, num_classes, dim=256):
        super().__init__()
        self.projection = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, dim))
        self.classifier = CosineClassifier(dim, num_classes)

    def forward(self, features):
        return self.classifier(self.projection(features))


class OghamConvNeXt(nn.Module):
    def __init__(self, backbone=BACKBONE, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        features = self.backbone.num_features
        self.direction_head = Head(features, len(DIRECTIONS))
        self.count_head = Head(features, 5)

    @property
    def mean(self):
        return tuple(self.backbone.pretrained_cfg["mean"])

    @property
    def std(self):
        return tuple(self.backbone.pretrained_cfg["std"])

    def head_parameters(self):
        return list(self.direction_head.parameters()) + list(self.count_head.parameters())

    def forward(self, x):
        features = self.backbone(x)
        return self.direction_head(features), self.count_head(features)

    @staticmethod
    def letter_log_probs(direction_logits, count_logits):
        log_dir = F.log_softmax(direction_logits, dim=-1)
        log_count = F.log_softmax(count_logits, dim=-1)
        return (log_dir.unsqueeze(2) + log_count.unsqueeze(1)).flatten(1)

    @torch.no_grad()
    def imprint(self, loader, device):
        """Blend each class weight with the mean normalised embedding of its support examples."""
        was_training = self.training
        self.eval()
        heads = [self.direction_head, self.count_head]
        sums = [torch.zeros_like(h.classifier.weight) for h in heads]
        counts = [torch.zeros(h.classifier.weight.shape[0], device=sums[0].device) for h in heads]
        for x, direction, count in loader:
            features = self.backbone(x.to(device))
            for i, (head, labels) in enumerate(zip(heads, (direction, count))):
                emb = F.normalize(head.projection(features), dim=-1)
                labels = labels.to(device)
                sums[i].index_add_(0, labels, emb)
                counts[i].index_add_(0, labels, torch.ones_like(labels, dtype=counts[i].dtype))
        for head, s, c in zip(heads, sums, counts):
            w = head.classifier.weight
            has = c > 0
            prototype = s[has] / c[has].unsqueeze(1)
            w.data[has] = F.normalize(0.5 * F.normalize(w.data[has], dim=-1) + 0.5 * prototype, dim=-1)
        self.train(was_training)
        return [int(c.gt(0).sum()) for c in counts]


def build_optimizer(model, backbone_lr, head_lr):
    return torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": backbone_lr},
         {"params": model.head_parameters(), "lr": head_lr}],
        weight_decay=0.05,
    )


def build_scheduler(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def loss_fn(direction_logits, count_logits, direction, count):
    return (F.cross_entropy(direction_logits, direction, label_smoothing=0.1)
            + F.cross_entropy(count_logits, count, label_smoothing=0.1))


def train_epoch(model, loader, optimizer, scheduler, device, desc="train"):
    model.train()
    total, n = 0.0, 0
    bar = progress(total=len(loader) * loader.batch_size, desc=desc, unit="img")
    for x, direction, count in loader:
        x, direction, count = x.to(device), direction.to(device), count.to(device)
        loss = loss_fn(*model(x), direction, count)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(x)
        n += len(x)
        bar.update(len(x))
        bar.set_postfix(loss=f"{total / n:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", refresh=False)
    bar.close()
    return total / max(1, n)


@torch.no_grad()
def predict_batches(model, loader, device, desc="predict"):
    """Returns (letter_probs [N, 20], true direction [N], true count index [N])."""
    model.eval()
    probs, directions, counts = [], [], []
    for x, direction, count in progress(loader, desc=desc, unit="batch", leave=False):
        log_probs = model.letter_log_probs(*model(x.to(device)))
        probs.append(log_probs.exp().cpu())
        directions.append(direction)
        counts.append(count)
    return torch.cat(probs), torch.cat(directions), torch.cat(counts)


def accuracies(letter_probs, direction, count):
    pred = letter_probs.argmax(1)
    pred_direction, pred_count = pred // 5, pred % 5
    true_letter = direction * 5 + count
    return {
        "direction_acc": (pred_direction == direction).float().mean().item(),
        "count_acc": (pred_count == count).float().mean().item(),
        "letter_acc": (pred == true_letter).float().mean().item(),
    }


# ---------------------------------------------------------------- checkpoints and prediction

def save_checkpoint(path, model, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "backbone": model.backbone_name,
                "git_commit": git_commit(), **extra}, path)


def load_model(path, device=None, pretrained=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = OghamConvNeXt(checkpoint["backbone"], pretrained=pretrained)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device or get_device()).eval()
    return model, checkpoint


_model_cache = {}


def predict_image(model_path, image_path):
    key = str(Path(model_path).resolve())
    if key not in _model_cache:
        _model_cache.clear()
        _model_cache[key] = load_model(model_path)[0]
    model = _model_cache[key]
    device = next(model.parameters()).device
    x = preprocess(load_grayscale(image_path), model.mean, model.std).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model.letter_log_probs(*model(x)).exp()[0].cpu()
    top = probs.topk(3)
    return {
        "predicted": LETTERS[top.indices[0]],
        "confidence": float(top.values[0]),
        "top_predictions": [(LETTERS[i], float(p)) for p, i in zip(top.values, top.indices)],
    }
