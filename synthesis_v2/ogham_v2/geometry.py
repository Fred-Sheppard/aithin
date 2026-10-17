"""E1 stroke model: Ogham characters as parametric trajectories in stem coordinates (DR-007).

Stem-local frame: u runs along the stem (0 = bottom end, 1 = top end; stem length = 1), v is the signed
perpendicular offset (positive = right of the stem when it points up). All randomness lives in trajectory
parameters; every sample must pass `validate` (label-preserving constraints) or it is resampled.

Config sections used (see configs/*.yaml): geometry, imperfection (E1b), hard (E1d), distractors (E1d), validator.
"""
from dataclasses import dataclass, field

import numpy as np

from .alphabet import ACROSS, DIAGONAL, LEFT, RIGHT, SPEC

N_PTS = 32


@dataclass
class Glyph:
    letter: str
    stem: list            # polylines (K,2) in (v,u); >1 piece when there are pen-lift gaps
    strokes: list         # main stroke polylines, ordered bottom->top, each running left end -> right end
    hooks: list           # (stroke_index, polyline) small start/end hooks drawn with their stroke
    extras: list          # distractor polylines (not part of the label): stem extension, neighbour strokes, stray marks
    tilt: float           # whole-character rotation in degrees (applied at render time)
    params: dict = field(default_factory=dict)


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _line(p0, p1, n=N_PTS):
    return np.linspace(np.asarray(p0, float), np.asarray(p1, float), n)


def _perturb(poly: np.ndarray, imp: dict, rng) -> np.ndarray:
    """Hand-drawn imperfection along the polyline normal: bow (curvature), low-freq waviness, tremor."""
    p0, p1 = poly[0], poly[-1]
    length = float(np.linalg.norm(p1 - p0)) or 1e-6
    tangent = (p1 - p0) / length
    normal = np.array([-tangent[1], tangent[0]])
    t = np.linspace(0, 1, len(poly))
    off = 4 * _u(rng, (-imp["curvature"], imp["curvature"])) * t * (1 - t)
    off += _u(rng, (0, imp["wave"])) * np.sin(2 * np.pi * _u(rng, (0.5, 1.5)) * t + _u(rng, (0, 2 * np.pi)))
    if imp.get("tremor", 0) > 0:
        noise = rng.normal(0, imp["tremor"], len(poly))
        noise = np.convolve(noise, np.ones(3) / 3, mode="same")
        off += noise
    return poly + np.outer(off * length, normal)


def _crossing_u(poly: np.ndarray):
    """u where the polyline crosses v=0 (or the u of its point nearest the stem)."""
    v, u = poly[:, 0], poly[:, 1]
    s = np.sign(v)
    idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    if len(idx):
        i = idx[0]
        f = v[i] / (v[i] - v[i + 1])
        return float(u[i] + f * (u[i + 1] - u[i]))
    return float(u[np.argmin(np.abs(v))])


def _chord_angle(poly: np.ndarray) -> float:
    """Chord angle in degrees relative to the stem-perpendicular; positive = '\\' slant (right end lower)."""
    (v0, u0), (v1, u1) = poly[0], poly[-1]
    return float(np.degrees(np.arctan2(-(u1 - u0), v1 - v0)))


def _segments_intersect(a0, a1, b0, b1) -> bool:
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a0, a1, b0) * orient(a0, a1, b1) < 0 and orient(b0, b1, a0) * orient(b0, b1, a1) < 0


def sample_glyph(letter: str, cfg: dict, rng, width: float) -> Glyph:
    spec = SPEC[letter]
    n, d = spec.count, spec.direction
    g = cfg["geometry"]
    hard = cfg.get("hard") or {}
    kind = str(rng.choice(hard["kinds"])) if hard and rng.random() < hard.get("p", 0) else None
    params = {"hard": kind}

    # --- positions along the stem
    s_lo, s_hi = g["spacing"]
    max_span = g.get("max_group_span", 0.75)
    if n > 1:
        s_hi_eff = max(s_lo, min(s_hi, max_span / (n - 1)))
        s = s_lo * _u(rng, (1.0, 1.15)) if kind == "near_touching" else _u(rng, (s_lo, s_hi_eff))
        jit = g["spacing_jitter"] * (1.75 if kind == "uneven_spacing" else 1.0)
        gaps = s * (1 + rng.uniform(-jit, jit, n - 1))
        span = gaps.sum()
        if span > max_span:
            gaps *= max_span / span
            span = max_span
    else:
        s, gaps, span = 0.0, np.zeros(0), 0.0
    start = (1 - span) * _u(rng, g["group_pos"])
    pos = start + np.concatenate([[0.0], np.cumsum(gaps)])

    # --- lengths, angles, balance
    key = "side" if d in (RIGHT, LEFT) else d
    lrange = g["length"][key]
    if kind == "short_strokes":
        lrange = (lrange[0], lrange[0] + 0.25 * (lrange[1] - lrange[0]))
    L_c = _u(rng, lrange)
    lj = g["length_jitter"]
    if d == DIAGONAL:
        th_c = _u(rng, g["diagonal_angle"])
    else:
        th_c = float(np.clip(rng.normal(0, g["angle_sd"]), -g["angle_max"], g["angle_max"]))
    b_c = _u(rng, g["balance"])
    tilt_sd = g["tilt_sd"] * (1.8 if kind == "steep_tilt" else 1.0)
    tilt = float(np.clip(rng.normal(0, tilt_sd), -g["tilt_max"], g["tilt_max"]))

    imp = cfg.get("imperfection") or {}
    strokes, hooks = [], []
    for k in range(n):
        L = L_c * (1 + _u(rng, (-lj, lj)))
        th = np.radians(th_c + _u(rng, (-g["angle_per_stroke"], g["angle_per_stroke"])))
        sign = -1.0 if d == DIAGONAL else 1.0
        dvec = np.array([np.cos(th), sign * np.sin(th)]) if d == DIAGONAL else np.array([np.cos(th), np.sin(th)])
        anchor = np.array([0.0, pos[k]])
        if d in (RIGHT, LEFT):
            ov = _u(rng, imp["overshoot"]) if imp else 0.0      # >0 = crosses the stem slightly
            if d == RIGHT:
                p0 = anchor - ov * dvec
                p1 = p0 + L * dvec
            else:
                p1 = anchor + ov * dvec
                p0 = p1 - L * dvec
        else:
            b = float(np.clip(b_c + _u(rng, (-g["balance_jitter"], g["balance_jitter"])), 0.05, 0.95))
            p0 = anchor - b * L * dvec
            p1 = anchor + (1 - b) * L * dvec
        poly = _line(p0, p1)
        if imp:
            poly = _perturb(poly, imp, rng)
            if rng.random() < imp.get("p_hook", 0):
                end = poly[0] if rng.random() < 0.5 else poly[-1]
                ang = np.radians(_u(rng, (60, 120)) * rng.choice([-1, 1]))
                rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ dvec
                hooks.append((k, _line(end, end + _u(rng, (0.015, 0.04)) * rot, 6)))
        strokes.append(poly)

    # --- stem
    stem = _line((0.0, 0.0), (0.0, 1.0), 48)
    if imp:
        stem = _perturb(stem, {**imp, "curvature": imp.get("stem_curvature", imp["curvature"] / 2)}, rng)
    pieces = [stem]
    if imp and rng.random() < imp.get("p_stem_gap", 0):
        for _ in range(10):
            ug = _u(rng, (0.05, 0.95))
            if np.min(np.abs(pos - ug)) > 0.05:
                gap = _u(rng, (0.01, 0.03))
                pieces = [stem[stem[:, 1] < ug - gap / 2], stem[stem[:, 1] > ug + gap / 2]]
                pieces = [p for p in pieces if len(p) >= 2]
                break

    extras, ext_side = _distractors(cfg.get("distractors") or {}, rng, width)
    params.update({"extension": ext_side, "spacing": s, "positions": pos.round(4).tolist(), "length": L_c, "angle": th_c, "balance": b_c,
                   "tilt": tilt})
    return Glyph(letter, pieces, strokes, hooks, extras, tilt, params)


def _distractors(dc: dict, rng, width: float):
    """Returns (extra polylines, side of the stem extension: 'up' | 'down' | None)."""
    extras, side_ext = [], None
    if not dc:
        return extras, side_ext
    if rng.random() < dc.get("p_stem_extension", 0):
        up = rng.random() < 0.5
        side_ext = "up" if up else "down"
        ext = _u(rng, (0.3, 0.8))
        u0, u1 = (1.0, 1.0 + ext) if up else (-ext, 0.0)
        extras.append(_line((0.0, u0), (0.0, u1), 16))
        if rng.random() < dc.get("p_neighbour", 0):
            side = rng.choice(["right", "left", "across"])
            L = _u(rng, (0.25, 0.6))
            for k in range(int(rng.integers(1, 4))):
                u = (1.0 + 0.15 + 0.12 * k) if up else (-0.15 - 0.12 * k)
                if side == "right":
                    extras.append(_line((0, u), (L, u), 8))
                elif side == "left":
                    extras.append(_line((-L, u), (0, u), 8))
                else:
                    extras.append(_line((-L / 2, u), (L / 2, u), 8))
    for _ in range(int(rng.integers(0, dc.get("max_stray", 0) + 1))):
        if rng.random() < dc.get("p_stray", 0):
            c = np.array([_u(rng, (-0.7, 0.7)), _u(rng, (-0.1, 1.1))])
            a = _u(rng, (0, np.pi))
            ln = _u(rng, (0.005, 0.06))
            extras.append(_line(c, c + ln * np.array([np.cos(a), np.sin(a)]), 4))
    return extras, side_ext


def validate(glyph: Glyph, cfg: dict, width: float):
    """Label-preserving constraints. Returns (ok, reason)."""
    vc = cfg["validator"]
    spec = SPEC[glyph.letter]
    d = spec.direction
    if len(glyph.strokes) != spec.count:
        return False, "count"
    anchors, angles = [], []
    for poly in glyph.strokes:
        v = poly[:, 0]
        vmin, vmax = float(v.min()), float(v.max())
        th = _chord_angle(poly)
        if d == RIGHT:
            if vmin < -vc["max_overshoot"] or vmax < vc["min_side_len"] or abs(th) > vc["side_max_angle"]:
                return False, "right"
        elif d == LEFT:
            if vmax > vc["max_overshoot"] or -vmin < vc["min_side_len"] or abs(th) > vc["side_max_angle"]:
                return False, "left"
        else:
            half = max(vc["min_half"], vc["min_half_frac"] * (vmax - vmin))
            if -vmin < half or vmax < half:
                return False, "cross_halves"
            if d == ACROSS and abs(th) > vc["across_max_angle"]:
                return False, "across_angle"
            if d == DIAGONAL and not (vc["diag_min_angle"] <= th <= vc["diag_max_angle"]):
                return False, "diag_angle"
        anchors.append(_crossing_u(poly) if d in (ACROSS, DIAGONAL) else float(poly[np.argmin(np.abs(v)), 1]))
        angles.append(th)
    if not all(0.0 < a < 1.0 for a in anchors):
        return False, "anchor_off_stem"
    if len(anchors) > 1:
        order = np.argsort(anchors)
        perp = np.cos(np.radians(np.mean(np.abs(angles))))
        gaps = np.diff(np.asarray(anchors)[order]) * perp
        if gaps.min() < max(vc["min_spacing"], vc["min_spacing_w"] * width):
            return False, "spacing"
        min_gap = max(vc.get("min_curve_gap", 0.035), vc.get("min_curve_gap_w", 1.6) * width)
        for i in range(len(glyph.strokes)):
            for j in range(i + 1, len(glyph.strokes)):
                a, b = glyph.strokes[i], glyph.strokes[j]
                if _segments_intersect(a[0], a[-1], b[0], b[-1]):
                    return False, "strokes_intersect"
                # curved / non-parallel strokes must not converge anywhere along their length
                if np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min() < min_gap:
                    return False, "strokes_converge"
    return True, ""


def sample_valid(letter: str, cfg: dict, rng, width: float, max_tries: int = 200):
    rejects = {}
    for _ in range(max_tries):
        glyph = sample_glyph(letter, cfg, rng, width)
        ok, why = validate(glyph, cfg, width)
        if ok:
            glyph.params["rejects"] = rejects
            return glyph
        rejects[why] = rejects.get(why, 0) + 1
    raise RuntimeError(f"could not sample a valid {letter}: {rejects}")
