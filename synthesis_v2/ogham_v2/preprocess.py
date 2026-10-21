"""Image preprocessing: use the SAME function for the synthetic training data and for real images at inference.

canonical() (recommended): grayscale -> blank check -> autocontrast -> robust ink-box crop -> 10% margin ->
    square pad -> INTER_AREA resize to 128. Scale-invariant; a blank page stays blank.
autocontrast_nearest(): autocontrast -> nearest-neighbour squash to 128, i.e. `preprocess_image` on branch
    `image-preprocessing-investigation` (model/image_preprocessing.py). Only for models that keep that function.
Both return uint8 HxW arrays (dark ink on white); divide by 255 for model input.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

SIZE = 128


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return Image.open(img)


def is_blank(gray: np.ndarray, min_contrast: float = 40) -> bool:
    """No ink: nothing (after a light blur) is at least `min_contrast` darker than the paper (the median).

    Checked on the raw image, before autocontrast (which would stretch paper noise into "ink"). A contrast
    test rather than a percentile test, so thin strokes covering < 1% of the image still count as ink.
    """
    g = cv2.GaussianBlur(np.asarray(gray, np.uint8), (3, 3), 0)
    return float(np.median(g)) - float(g.min()) < min_contrast


def ink_bbox(gray: np.ndarray):
    """Bounding box (x0, y0, x1, y1) of the character's ink, or None for a blank page.

    Otsu on a blurred image; drops specks; drops components touching the border (edge shadows,
    show-through at the crop edge) unless the character itself touches the border.
    """
    h, w = gray.shape
    if is_blank(gray):
        return None
    k = max(3, (min(h, w) // 200) * 2 + 1)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    comps = stats[1:]
    areas = comps[:, cv2.CC_STAT_AREA]
    keep = areas >= max(0.02 * areas.max(), 1e-4 * h * w)
    comps = comps[keep]
    x, y, cw, ch = (comps[:, i] for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    touches = (x <= 0) | (y <= 0) | (x + cw >= w) | (y + ch >= h)
    inner = ~touches
    if inner.any() and comps[inner, cv2.CC_STAT_AREA].max() >= 0.3 * comps[:, cv2.CC_STAT_AREA].max():
        sel = inner
    else:
        sel = np.ones(len(comps), bool)
    return int(x[sel].min()), int(y[sel].min()), int((x + cw)[sel].max()), int((y + ch)[sel].max())


def canonical(img, size: int = SIZE, margin: float = 0.10, cutoff: float = 1.0) -> np.ndarray:
    pil = _to_pil(img).convert("L")
    if is_blank(np.asarray(pil)):
        return np.full((size, size), 255, np.uint8)
    pil = ImageOps.autocontrast(pil, cutoff=cutoff)
    gray = np.asarray(pil, dtype=np.uint8)
    box = ink_bbox(gray)
    if box is not None:
        x0, y0, x1, y1 = box
        m = int(round(margin * max(x1 - x0, y1 - y0)))
        h, w = gray.shape
        gray = gray[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)]
    h, w = gray.shape
    side = max(h, w)
    canvas = np.full((side, side), 255, np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = gray
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def autocontrast_nearest(img, size: int = SIZE, cutoff: float = 1.0) -> np.ndarray:
    pil = ImageOps.autocontrast(_to_pil(img).convert("L"), cutoff=cutoff)
    return np.asarray(pil.resize((size, size), Image.NEAREST), dtype=np.uint8)


MODES = {"canonical": canonical, "autocontrast": autocontrast_nearest}


def apply(mode: str, img) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"unknown preprocessing mode {mode!r}; choose from {sorted(MODES)}")
    return MODES[mode](img)
