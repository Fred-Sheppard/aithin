"""E2: appearance and capture randomisation (paper, ink, lighting, capture, artefacts). Output is RGB uint8.

Operates on the E1 render (grayscale, ink dark on white) treated as ink coverage. Every section is optional and
has its own probability, so E2a..E2d are cumulative config overlays. Fast numpy/OpenCV implementations are used
instead of Augraphy (which costs 0.1-1 s per image); the effects mirror Augraphy's Ink/Paper/Post phases.
Real blank-paper photo textures (deferred, DR-013) can be dropped into ogham_v2/textures/*.png.
"""
from pathlib import Path

import cv2
import numpy as np

TEXTURES = Path(__file__).resolve().parent / "textures"
INK = {  # RGB
    "black": (25, 25, 30), "blue": (40, 50, 120), "purple": (75, 60, 105), "grey": (95, 95, 100),
}


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _lowfreq(rng, h, w, cells=4):
    g = rng.normal(0, 1, (cells, cells)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _paper(pc, rng, h, w):
    base = _u(rng, pc["base"])
    tint = rng.normal(0, pc.get("tint", 0), 3)
    paper = np.empty((h, w, 3), np.float32)
    paper[:] = base + tint
    if pc.get("lowfreq"):
        paper += _u(rng, (0, pc["lowfreq"])) * _lowfreq(rng, h, w)[..., None]
    if pc.get("grain"):
        g = rng.normal(0, _u(rng, (0, pc["grain"])), (h, w)).astype(np.float32)
        paper += cv2.GaussianBlur(g, (0, 0), 0.7)[..., None]
    if rng.random() < pc.get("fibre_p", 0):
        for _ in range(int(rng.integers(5, 40))):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            a, ln = _u(rng, (0, np.pi)), _u(rng, (3, 15))
            cv2.line(paper, (x, y), (int(x + ln * np.cos(a)), int(y + ln * np.sin(a))), (base - _u(rng, (8, 25)),) * 3, 1, cv2.LINE_AA)
    if rng.random() < pc.get("ruled_p", 0):
        step = int(_u(rng, (0.08, 0.18)) * h)
        col = np.array([170, 190, 230]) if rng.random() < 0.7 else np.array([150, 150, 150])
        off = int(rng.integers(0, step))
        for y in range(off, h, max(4, step)):
            cv2.line(paper, (0, y), (w, y), col.tolist(), 1, cv2.LINE_AA)
        if rng.random() < pc.get("grid_p_given_ruled", 0.3):
            for x in range(off, w, max(4, step)):
                cv2.line(paper, (x, 0), (x, h), col.tolist(), 1, cv2.LINE_AA)
    tex = sorted(TEXTURES.glob("*.png")) if TEXTURES.exists() else []
    if tex and rng.random() < pc.get("texture_p", 0):
        t = cv2.imread(str(tex[int(rng.integers(len(tex)))]))[..., ::-1].astype(np.float32)
        t = cv2.resize(t, (w, h))
        paper = 0.5 * paper + 0.5 * t
    return paper


def _ink(cov, ic, pen, rng):
    kind = pen.get("kind", "constant")
    choices = ic["marker" if kind == "marker" else "ballpoint"]
    names, probs = zip(*choices.items())
    name = names[int(rng.choice(len(names), p=np.array(probs) / sum(probs)))]
    colour = np.array(INK[name], np.float32) + rng.normal(0, 8, 3)
    if rng.random() < ic.get("bleed_p", 0):                      # ink bleed: soft spread into the paper
        k = _u(rng, (0.6, 1.5))
        cov = np.maximum(cov, 0.6 * cv2.GaussianBlur(cov, (0, 0), k))
    if rng.random() < ic.get("lowink_p", 0):                     # low ink: streaky dropouts along the stroke
        h, w = cov.shape
        streak = np.clip(1 - _u(rng, (0.2, 0.6)) * np.clip(_lowfreq(rng, h, w, cells=int(rng.integers(6, 16))), 0, None), 0, 1)
        cov = cov * streak
    return cov, colour, name


def _lighting(img, lc, rng):
    h, w = img.shape[:2]
    if lc.get("gradient"):
        a = _u(rng, (0, 2 * np.pi))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = ((xx / w - 0.5) * np.cos(a) + (yy / h - 0.5) * np.sin(a))
        img = img * (1 + _u(rng, (0, lc["gradient"])) * ramp)[..., None]
    if rng.random() < lc.get("shadow_p", 0):                      # soft diagonal shadow band (hand / phone)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = _u(rng, (0, np.pi))
        d = (xx - _u(rng, (0, w))) * np.cos(a) + (yy - _u(rng, (0, h))) * np.sin(a)
        width = _u(rng, (0.1, 0.5)) * max(h, w)
        band = 1 / (1 + np.exp(-d / (0.15 * width)))
        img = img * (1 - _u(rng, lc["shadow_strength"]) * band)[..., None]
    if lc.get("vignette"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx / w - 0.5) * 2) ** 2 + ((yy / h - 0.5) * 2) ** 2)
        img = img * (1 - _u(rng, (0, lc["vignette"])) * np.clip(r - 0.5, 0, None))[..., None]
    if lc.get("wb"):
        img = img * (1 + rng.normal(0, lc["wb"], 3)).astype(np.float32)
    if lc.get("gamma"):
        g = _u(rng, lc["gamma"])
        img = 255 * (np.clip(img, 0, 255) / 255) ** g
    return img


def _capture(img, cc, rng):
    h, w = img.shape[:2]
    if rng.random() < cc.get("persp_p", 0) or cc.get("rotate"):
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        j = cc.get("persp", 0) * min(h, w) if rng.random() < cc.get("persp_p", 0) else 0
        dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        if cc.get("rotate"):
            R = cv2.getRotationMatrix2D((w / 2, h / 2), _u(rng, (-cc["rotate"], cc["rotate"])), 1.0)
            M = np.vstack([R, [0, 0, 1]]) @ M
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < cc.get("blur_p", 0):
        img = cv2.GaussianBlur(img, (0, 0), _u(rng, cc["blur"]))
    if rng.random() < cc.get("motion_p", 0):
        k = int(_u(rng, (3, 9)))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1 / k
        R = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), _u(rng, (0, 180)), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(ker, R, (k, k)))
    if rng.random() < cc.get("rescale_p", 0):
        f = _u(rng, cc["rescale"])
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if cc.get("noise"):
        img = img + rng.normal(0, _u(rng, (0, cc["noise"])), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < cc.get("jpeg_p", 0):
        q = int(_u(rng, cc["jpeg"]))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


def _artefacts(img, cov, ac, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if rng.random() < ac.get("showthrough_p", 0):                 # faint mirrored ink from the back of the page
        ghost = cv2.GaussianBlur(cov[:, ::-1], (0, 0), _u(rng, (1, 3)))
        ghost = np.roll(ghost, (int(rng.integers(-h // 4, h // 4)), int(rng.integers(-w // 4, w // 4))), (0, 1))
        out *= (1 - _u(rng, (0.05, 0.18)) * ghost)[..., None]
    if rng.random() < ac.get("edge_band_p", 0):                   # dark band at one crop edge (scanner lid / page edge)
        side = int(rng.integers(4))
        bw = max(2, int(_u(rng, (0.02, 0.08)) * (w if side < 2 else h)))
        dark = _u(rng, (0.3, 0.75))
        if side == 0:
            out[:, :bw] *= dark
        elif side == 1:
            out[:, -bw:] *= dark
        elif side == 2:
            out[:bw] *= dark
        else:
            out[-bw:] *= dark
    if rng.random() < ac.get("speckle_p", 0):
        n = int(rng.integers(5, 60))
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        for y, x in zip(ys, xs):
            cv2.circle(out, (int(x), int(y)), int(rng.integers(1, 3)), (float(_u(rng, (60, 160))),) * 3, -1, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


def _loose(gray, ac, rng):
    """Pad the render so the character fills only part of the frame (loose phone/scan crops)."""
    if rng.random() >= ac.get("loose_p", 0):
        return gray
    h, w = gray.shape
    fill = _u(rng, ac.get("loose_fill", (0.15, 0.6)))
    H, W = int(h / fill), int(w / fill)
    out = np.full((H, W), 255, np.uint8)
    y, x = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
    out[y:y + h, x:x + w] = gray
    if max(H, W) > 900:                                            # keep native size bounded
        s = 900 / max(H, W)
        out = cv2.resize(out, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return out


def apply(gray: np.ndarray, pen: dict, cfg: dict, rng) -> np.ndarray:
    ac = cfg["appearance"]
    art = ac.get("artefacts") or {}
    gray = _loose(gray, art, rng)
    cov = 1.0 - gray.astype(np.float32) / 255.0
    h, w = cov.shape
    paper = _paper(ac.get("paper") or {"base": (250, 255)}, rng, h, w)
    colour = np.array(INK["black"], np.float32)
    if ac.get("ink"):
        cov, colour, _ = _ink(cov, ac["ink"], pen, rng)
    img = paper * (1 - cov[..., None]) + colour * cov[..., None]
    if ac.get("lighting"):
        img = _lighting(img, ac["lighting"], rng)
    img = np.clip(img, 0, 255)
    if art:
        img = _artefacts(img, cov, art, rng)
    if ac.get("capture"):
        img = _capture(img.astype(np.float32), ac["capture"], rng)
    return np.clip(img, 0, 255).astype(np.uint8)
