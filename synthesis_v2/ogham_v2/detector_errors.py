"""E3b: simulate line-detector failures on (exact) oracle JSON.

Error types (rates per stroke unless noted), applied in stem-normalised space:
  drop        a stroke is missed
  split       a crossing stroke is reported as separate left + right pieces (marker-00031-style)
  merge       two adjacent strokes are reported as one (their average)
  spurious    an extra short stroke / unattached segment (per glyph)
  jitter      endpoint noise (stem lengths, sd)
  truncate    the detected stem is shorter (per glyph); strokes outside it are lost
"""
import copy

import numpy as np

from .detector_schema import stroke_record


def _frame(det):
    st = det["stem"]
    sp0, sp1 = np.asarray(st["p0"], float), np.asarray(st["p1"], float)
    L = float(np.linalg.norm(sp1 - sp0))
    u_hat = (sp1 - sp0) / L
    return sp0, u_hat, np.array([-u_hat[1], u_hat[0]]), L, st["angle"]


def apply(det: dict, ec: dict, rng) -> dict:
    det = copy.deepcopy(det)
    sp0, u_hat, right, L, ang = _frame(det)
    P = lambda d, t: sp0 + t * L * u_hat + d * L * right          # noqa: E731
    strokes = []
    for s in det["strokes"]:
        if rng.random() < ec.get("drop", 0):
            continue
        jit = ec.get("jitter", 0)
        d0, t0, d1, t1 = (v + rng.normal(0, jit) for v in (s["d0"], s["t0"], s["d1"], s["t1"]))
        if s["side"] == "across" and rng.random() < ec.get("split", 0):
            tc = s["position"]
            strokes.append(stroke_record(P(d0, t0), P(-0.01, tc), sp0, u_hat, right, L, ang))
            strokes.append(stroke_record(P(0.01, tc), P(d1, t1), sp0, u_hat, right, L, ang))
            continue
        strokes.append(stroke_record(P(d0, t0), P(d1, t1), sp0, u_hat, right, L, ang))
    strokes.sort(key=lambda r: r["position"])
    out, i = [], 0
    while i < len(strokes):
        if i + 1 < len(strokes) and rng.random() < ec.get("merge", 0):
            a, b = strokes[i], strokes[i + 1]
            out.append(stroke_record(P((a["d0"] + b["d0"]) / 2, (a["t0"] + b["t0"]) / 2),
                                     P((a["d1"] + b["d1"]) / 2, (a["t1"] + b["t1"]) / 2), sp0, u_hat, right, L, ang))
            i += 2
        else:
            out.append(strokes[i])
            i += 1
    if rng.random() < ec.get("spurious", 0):
        t = rng.uniform(0.05, 0.95)
        ln = rng.uniform(0.08, 0.3)
        if rng.random() < 0.5:
            side = rng.choice([-1, 1])
            out.append(stroke_record(P(0, t), P(side * ln, t + rng.normal(0, 0.03)), sp0, u_hat, right, L, ang))
        else:
            d = rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
            q0, q1 = P(d, t), P(d + ln * 0.5, t + rng.normal(0, 0.05))
            det["unattached"].append({"p0": q0.tolist(), "p1": q1.tolist()})
    if rng.random() < ec.get("truncate", 0):
        lo, hi = rng.uniform(0, 0.15), 1 - rng.uniform(0, 0.15)
        out = [s for s in out if lo <= s["position"] <= hi]
    det["strokes"] = sorted(out, key=lambda r: r["position"])
    return det
