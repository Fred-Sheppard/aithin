"""The teammate's line-detector JSON schema, and exact 'oracle' JSON from generator geometry.

Field semantics (inferred from the teammate's sample; confirm with them - DR to be added):
  stem.p0 = bottom end, stem.p1 = top end (image px); stem.angle = degrees in a y-up frame (90 = vertical)
  stroke.p0/p1 = left / right end (image px)
  t = projection along the stem from stem.p0 (0) to stem.p1 (1); d = signed perpendicular offset / stem length
      (negative = left of the stem); position = (t0 + t1) / 2; length = stroke length / stem length
  angle = stem.angle - stroke direction angle (90 = perpendicular to the stem)
  side in {left, right, across}; balance = |d0| / (|d0| + |d1|); gap = distance from the stem if not touching
"""
import numpy as np


def _angle(p0, p1) -> float:
    return float(np.degrees(np.arctan2(-(p1[1] - p0[1]), p1[0] - p0[0])))


ACROSS_MIN_ABS = 0.055   # between the validator's max side-stroke overshoot (0.05) and min crossing half (0.06)
ACROSS_MIN_FRAC = 0.12


def stroke_record(p0, p1, sp0, u_hat, right, L, stem_angle, support=1.0) -> dict:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t0, t1 = float((p0 - sp0) @ u_hat / L), float((p1 - sp0) @ u_hat / L)
    d0, d1 = float((p0 - sp0) @ right / L), float((p1 - sp0) @ right / L)
    if d0 > d1:
        p0, p1, t0, t1, d0, d1 = p1, p0, t1, t0, d1, d0
    short = min(-d0, d1)                    # extent of the shorter half if the stroke crosses the stem
    if short >= max(ACROSS_MIN_ABS, ACROSS_MIN_FRAC * (d1 - d0)):
        side, gap = "across", 0.0
    elif abs(d1) < abs(d0):                 # most of the stroke is left of the stem (slight overshoot allowed)
        side, gap = "left", max(0.0, -d1)
    else:
        side, gap = "right", max(0.0, d0)
    return {
        "p0": p0.tolist(), "p1": p1.tolist(), "t0": t0, "t1": t1, "d0": d0, "d1": d1,
        "position": (t0 + t1) / 2, "length": float(np.linalg.norm(p1 - p0) / L),
        "angle": stem_angle - _angle(p0, p1), "side": side,
        "balance": abs(d0) / (abs(d0) + abs(d1) + 1e-9), "gap": gap, "support": support,
    }


def oracle(glyph, info: dict) -> dict:
    to_img = info["to_img"]
    sp0, sp1 = to_img(np.array([[0.0, 0.0], [0.0, 1.0]]))
    L = float(np.linalg.norm(sp1 - sp0))
    u_hat = (sp1 - sp0) / L
    right = np.array([-u_hat[1], u_hat[0]])
    stem_angle = _angle(sp0, sp1)
    strokes = []
    for poly in glyph.strokes:
        a, b = to_img(poly[[0, -1]])
        strokes.append(stroke_record(a, b, sp0, u_hat, right, L, stem_angle))
    strokes.sort(key=lambda s: s["position"])
    unattached = [{"p0": e[0].tolist(), "p1": e[1].tolist()} for e in (to_img(x[[0, -1]]) for x in glyph.extras)]
    return {
        "image_size": list(info["size"]), "stroke_width": float(info["width_px"]),
        "stem": {"p0": sp0.tolist(), "p1": sp1.tolist(), "length": L, "angle": stem_angle, "support": 1.0},
        "ambiguous": False, "strokes": strokes, "unattached": unattached,
    }
