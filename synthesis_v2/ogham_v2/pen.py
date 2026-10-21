"""Pen/ink model and supersampled variable-width rasteriser.

pen modes:
  constant: a single pen (width as a fraction of stem length), full-black ink, no taper
  mixture:  ballpoint vs marker (p_marker), each with its own width, pressure taper and ink intensity ranges
render() returns a grayscale uint8 image (ink dark on white) plus render info (canvas size, stem->image transform).
"""
import cv2
import numpy as np


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def sample_pen(cfg: dict, rng) -> dict:
    p = cfg["pen"]
    if p["mode"] == "constant":
        return {"kind": "constant", "width": _u(rng, p["width"]), "taper": 0.0, "intensity": (1.0, 1.0), "wobble": 0.0}
    marker = rng.random() < p["p_marker"]
    k = p["marker"] if marker else p["ballpoint"]
    return {"kind": "marker" if marker else "ballpoint", "width": _u(rng, k["width"]), "taper": _u(rng, k["taper"]),
            "intensity": (_u(rng, k["intensity"]), _u(rng, k["intensity"])), "wobble": k.get("width_wobble", 0.0)}


def _to_img_transform(tilt_deg: float, scale: float):
    t = np.radians(tilt_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def upright(pts):                       # stem-local (v,u) -> rotated upright (X right, Y up), in px
        return (np.asarray(pts, float) * scale) @ R.T
    return upright


def render(glyph, pen: dict, cfg: dict, rng):
    rc = cfg["render"]
    ss = int(rc.get("supersample", 4))
    S = _u(rng, rc["stem_px"])
    upright = _to_img_transform(glyph.tilt, S)

    main = [p for p in glyph.stem] + glyph.strokes + [h for _, h in glyph.hooks]
    allpts = upright(np.concatenate(main))
    w_px = pen["width"] * S
    lo, hi = allpts.min(0) - w_px, allpts.max(0) + w_px
    size = hi - lo
    m = [_u(rng, rc["margin"]) * size.max() for _ in range(4)]          # left, right, top, bottom
    ext = glyph.params.get("extension")
    if ext:                       # stem runs on past the crop edge: cut the crop tight on that side
        m[2 if ext == "up" else 3] = _u(rng, (0.0, 0.06)) * size.max()
    W, H = size[0] + m[0] + m[1], size[1] + m[2] + m[3]
    aspect = _u(rng, rc["aspect"])                                          # target h / w
    if H / W < aspect:
        extra = aspect * W - H
        f = rng.random()
        m[2] += f * extra
        m[3] += (1 - f) * extra
        H = aspect * W
    else:
        extra = H / aspect - W
        f = rng.random()
        m[0] += f * extra
        m[1] += (1 - f) * extra
        W = H / aspect
    W, H = int(np.ceil(W)), int(np.ceil(H))
    x0, ytop = lo[0] - m[0], hi[1] + m[2]

    def to_img(pts):                         # stem-local -> image px (x right, y down)
        up = upright(pts)
        return np.stack([up[:, 0] - x0, ytop - up[:, 1]], 1)

    cov = np.zeros((H * ss, W * ss), np.float32)
    items = []
    for poly in glyph.stem:
        items.append((poly, 1.0))
    for k, poly in enumerate(glyph.strokes):
        items.append((poly, 1.0))
    for _, h in glyph.hooks:
        items.append((h, 0.8))
    for e in glyph.extras:
        items.append((e, 1.0))
    lo_i, hi_i = pen["intensity"]
    for poly, wmul in items:
        intensity = _u(rng, (min(lo_i, hi_i), max(lo_i, hi_i)))
        _draw(cov, to_img(poly) * ss, w_px * wmul * ss, intensity, pen["taper"], pen["wobble"], rng)
    cov = cv2.resize(cov, (W, H), interpolation=cv2.INTER_AREA)
    img = np.clip(255.0 * (1.0 - cov), 0, 255).astype(np.uint8)
    return img, {"to_img": to_img, "stem_px": S, "width_px": w_px, "size": (W, H)}


def _draw(cov, pts, width, intensity, taper, wobble, rng):
    n = len(pts)
    t = np.linspace(0, 1, n)
    wmul = 1 - taper * (1 - np.clip(np.minimum(t, 1 - t) / 0.12, 0, 1))
    if wobble:
        wmul *= 1 + wobble * np.sin(2 * np.pi * _u(rng, (1, 3)) * t + _u(rng, (0, 6.28)))
    pad = width * max(1.0, wmul.max()) + 4
    H, W = cov.shape
    x0, y0 = max(0, int(pts[:, 0].min() - pad)), max(0, int(pts[:, 1].min() - pad))
    x1, y1 = min(W, int(pts[:, 0].max() + pad) + 1), min(H, int(pts[:, 1].max() + pad) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    layer = np.zeros((y1 - y0, x1 - x0), np.float32)
    p = np.round((pts - [x0, y0]) * 4).astype(np.int32)     # 2 bits of sub-pixel precision (shift=2)
    for i in range(n - 1):
        th = max(1, int(round(width * 0.5 * (wmul[i] + wmul[i + 1]))))
        cv2.line(layer, tuple(map(int, p[i])), tuple(map(int, p[i + 1])), 1.0, th, cv2.LINE_AA, shift=2)
    for i in (0, n - 1):
        r = max(1, int(round(width * wmul[i] / 2)))
        cv2.circle(layer, tuple(map(int, p[i])), r * 4, 1.0, -1, cv2.LINE_AA, shift=2)
    region = cov[y0:y1, x0:x1]
    np.maximum(region, intensity * np.clip(layer, 0, 1), out=region)
