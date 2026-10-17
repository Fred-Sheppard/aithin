"""Image preprocessing shared by synthetic and real data (DR-005).

canonical(): grayscale -> autocontrast -> robust ink bbox crop -> margin -> square pad -> INTER_AREA.
legacy():    exactly what keras load_img(grayscale, target_size=(128,128)) does (PIL 'L' + nearest squash).
Both return uint8 HxW arrays (black ink on white); training divides by 255.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

VERSION = "canonical-v1"
SIZE = 128


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return Image.open(img)


def legacy(img, size: int = SIZE) -> np.ndarray:
    pil = _to_pil(img)
    if pil.mode not in ("L", "I;16", "I"):
        pil = pil.convert("L")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.NEAREST)
    return np.asarray(pil, dtype=np.uint8)


def ink_bbox(gray: np.ndarray):
    """Bounding box (x0, y0, x1, y1) of the character's ink, or None for a blank page.

    Otsu on a blurred image; drops specks; drops components touching the border (edge shadows,
    show-through at the crop edge) unless the character itself touches the border.
    """
    h, w = gray.shape
    lo, hi = np.percentile(gray, [1, 99])
    if hi - lo < 40:
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


def is_blank(gray: np.ndarray, min_range: float = 40) -> bool:
    """No ink: checked on the RAW image, before autocontrast (which would stretch paper noise into 'ink')."""
    lo, hi = np.percentile(gray, [1, 99])
    return hi - lo < min_range


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
    """Teammate branch `image-preprocessing-investigation`: autocontrast then nearest squash."""
    pil = ImageOps.autocontrast(_to_pil(img).convert("L"), cutoff=cutoff)
    return np.asarray(pil.resize((size, size), Image.NEAREST), dtype=np.uint8)


def apply(mode: str, img) -> np.ndarray:
    if mode == "canonical":
        return canonical(img)
    if mode == "legacy":
        return legacy(img)
    if mode == "autocontrast":
        return autocontrast_nearest(img)
    if mode == "render":                      # E3: real image -> proxy line detector -> canonical render
        from .canonical_render import render
        from .detectors.proxy import detect
        return render(detect(np.asarray(_to_pil(img).convert("L"))))
    raise ValueError(f"unknown preprocessing mode {mode!r}")
