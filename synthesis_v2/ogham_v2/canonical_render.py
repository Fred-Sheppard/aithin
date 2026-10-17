"""Detector JSON -> 128x128 canonical raster (the shared representation for E3, DR-008).

Everything is expressed in stem coordinates (stem from (0,0) to (0,1), d = perpendicular offset), then drawn
at a fixed scale with fixed-width anti-aliased lines: stroke width, ink, paper and camera are all discarded.
A missing stem (detector failure / blank page) renders as a blank canvas.
"""
import cv2
import numpy as np

SIZE = 128
SCALE = 68.0          # px per stem length; canvas covers t in [-0.44, 1.44], d in [-0.94, 0.94]
LINE = 3              # px


def _stem_frame(det):
    st = det["stem"]
    sp0, sp1 = np.asarray(st["p0"], float), np.asarray(st["p1"], float)
    L = float(np.linalg.norm(sp1 - sp0)) or 1.0
    u_hat = (sp1 - sp0) / L
    right = np.array([-u_hat[1], u_hat[0]])
    return sp0, u_hat, right, L


def to_px(d, t):
    return SIZE / 2 + d * SCALE, SIZE / 2 + (0.5 - t) * SCALE


def render(det: dict, unattached: str = "faint") -> np.ndarray:
    img = np.zeros((SIZE * 4, SIZE * 4), np.uint8)       # ink coverage, 4x supersampled
    if not det or not det.get("stem"):
        return np.full((SIZE, SIZE), 255, np.uint8)
    sp0, u_hat, right, L = _stem_frame(det)

    def seg(p0, p1, value):
        a = [(np.asarray(p, float) - sp0) for p in (p0, p1)]
        (x0, y0), (x1, y1) = (to_px(v @ right / L, v @ u_hat / L) for v in a)
        cv2.line(img, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)), value, LINE * 4, cv2.LINE_AA, shift=2)

    seg(sp0, sp0 + L * u_hat, 255)
    for s in det.get("strokes", []):
        seg(s["p0"], s["p1"], 255)
    if unattached != "omit":
        for u in det.get("unattached", []):
            seg(u["p0"], u["p1"], 110 if unattached == "faint" else 255)
    small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return (255 - small).astype(np.uint8)
