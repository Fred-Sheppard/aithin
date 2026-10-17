"""Proxy line detector (stand-in for the teammate's detector; same JSON output).

1. normalise: autocontrast, crop to ink (preprocess.ink_bbox), resize so the ink box is ~WORK px
2. binarise (Otsu) and skeletonise
3. stem = strongest near-vertical Hough line on the skeleton; its extent = longest run of supporting pixels
4. remove the stem band; remaining skeleton components are stroke pieces (PCA line fit each)
5. pair collinear pieces on opposite sides that meet at the stem -> 'across' strokes; lone pieces touching
   the stem -> 'left'/'right'; pieces away from the stem -> 'unattached'
Coordinates in the output are in the ORIGINAL image's pixels.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize

from ..detector_schema import stroke_record
from ..preprocess import ink_bbox, is_blank

WORK = 400


def _normalise(gray):
    if is_blank(gray):
        return None
    g = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1.0))
    box = ink_bbox(g)
    h, w = g.shape
    if box is None:
        return None
    x0, y0, x1, y1 = box
    m = int(0.08 * max(x1 - x0, y1 - y0)) + 2
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    crop = g[y0:y1, x0:x1]
    s = WORK / max(crop.shape)
    small = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return small, (x0, y0, s)


def _pca_line(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    proj = (pts - c) @ d
    return c + proj.min() * d, c + proj.max() * d


def detect(gray: np.ndarray) -> dict:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    norm = _normalise(gray)
    empty = {"image_size": [int(gray.shape[1]), int(gray.shape[0])], "stroke_width": 0.0, "stem": None,
             "ambiguous": True, "strokes": [], "unattached": []}
    if norm is None:
        return empty
    small, (ox, oy, s) = norm
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    skel = skeletonize(ink > 0)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    width = float(2 * np.median(dist[skel])) if skel.any() else 1.0
    ys, xs = np.nonzero(skel)
    if len(xs) < 10:
        return empty

    # --- stem: strongest near-vertical Hough line (+-30 deg) whose supporting run is long enough
    pts = np.stack([xs, ys], 1).astype(float)
    band = max(4.0, 1.5 * width, 0.02 * WORK)                        # tolerant of wavy hand-drawn stems
    lines = cv2.HoughLines(skel.astype(np.uint8) * 255, 1, np.pi / 360, threshold=max(10, int(0.1 * WORK)),
                           min_theta=0, max_theta=np.pi)
    stem = None
    for rho, theta in ([] if lines is None else lines[:, 0]):
        if min(theta, np.pi - theta) > np.radians(30):
            continue
        nrm = np.array([np.cos(theta), np.sin(theta)])             # line: x*cos + y*sin = rho
        direc = np.array([-np.sin(theta), np.cos(theta)])
        on = np.abs(pts @ nrm - rho) <= band
        if on.sum() < 10:
            continue
        order = np.sort(pts[on] @ direc)
        segs = np.split(order, np.flatnonzero(np.diff(order) > max(8.0, 0.06 * WORK)) + 1)
        run = max(segs, key=lambda a: a[-1] - a[0])
        if run[-1] - run[0] >= 0.2 * WORK:
            stem = (rho, nrm, direc, on, run[0], run[-1])
            break
    if stem is None:
        return empty
    rho, nrm, direc, on, a_lo, a_hi = stem
    foot = rho * nrm
    e0, e1 = foot + a_lo * direc, foot + a_hi * direc
    sp0, sp1 = (e0, e1) if e0[1] > e1[1] else (e1, e0)               # p0 = bottom end
    L = float(np.linalg.norm(sp1 - sp0))
    u_hat = (sp1 - sp0) / L
    right = np.array([-u_hat[1], u_hat[0]])
    stem_angle = float(np.degrees(np.arctan2(-(sp1[1] - sp0[1]), sp1[0] - sp0[0])))
    support = float(on.sum() / max(L, 1))

    # --- stroke pieces: skeleton minus the stem band
    rest = skel.copy()
    rest[ys[on], xs[on]] = False
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    pieces = []
    for i in range(1, n):
        yy, xx = np.nonzero(lab == i)
        if len(xx) < max(4, 0.04 * L):
            continue
        q0, q1 = _pca_line(np.stack([xx, yy], 1).astype(float))
        t = np.array([(q0 - sp0) @ u_hat, (q1 - sp0) @ u_hat]) / L
        d = np.array([(q0 - sp0) @ right, (q1 - sp0) @ right]) / L
        if abs(d[0]) > abs(d[1]):                                     # inner end first
            q0, q1, t, d = q1, q0, t[::-1], d[::-1]
        if np.linalg.norm(q1 - q0) < 0.05 * L:
            continue
        vq = q1 - q0
        rel = np.degrees(np.arccos(min(1.0, abs(vq @ u_hat) / (np.linalg.norm(vq) + 1e-9))))
        if rel < 20:                                                  # parallel to the stem: stem fragment
            continue
        pieces.append({"q0": q0, "q1": q1, "t": t, "d": d, "n": len(xx)})

    near = (band + 2 * width) / L + 0.03
    used, strokes, unattached = set(), [], []
    left = [i for i, p in enumerate(pieces) if p["d"][1] < 0 and abs(p["d"][0]) <= near]
    rightp = [i for i, p in enumerate(pieces) if p["d"][1] > 0 and abs(p["d"][0]) <= near]

    def ang(p):
        v = p["q1"] - p["q0"]
        a = np.degrees(np.arctan2(v[1], v[0]))
        return a

    def cross_t(p):                     # where the piece's line meets the stem (d = 0)
        (t0, t1), (d0, d1) = p["t"], p["d"]
        return t0 if abs(d1 - d0) < 1e-9 else t0 - d0 * (t1 - t0) / (d1 - d0)

    pairs = []
    for i in left:
        for j in rightp:
            pi_, pj = pieces[i], pieces[j]
            da = abs(((ang(pi_) - ang(pj) + 180) % 360) - 180)       # opposite directions -> ~180
            coll = abs(180 - da)
            dt = abs(cross_t(pi_) - cross_t(pj))
            if coll < 15 and dt < 0.05:
                pairs.append((coll + 100 * dt, i, j))
    for _, i, j in sorted(pairs):
        if i in used or j in used:
            continue
        used |= {i, j}
        a, b = pieces[i]["q1"], pieces[j]["q1"]
        strokes.append(stroke_record(a, b, sp0, u_hat, right, L, stem_angle, support=1.0))
    for i, p in enumerate(pieces):
        if i in used:
            continue
        if abs(p["d"][0]) <= near:
            inner = sp0 + cross_t(p) * L * u_hat if abs(p["d"][0]) <= near else p["q0"]
            strokes.append(stroke_record(inner, p["q1"], sp0, u_hat, right, L, stem_angle, support=1.0))
        else:
            unattached.append({"p0": p["q0"], "p1": p["q1"]})
    strokes = [s for s in strokes if 0.0 <= s["position"] <= 1.0] + []
    strokes.sort(key=lambda r: r["position"])

    def back(p):                       # working px -> original image px
        p = np.asarray(p, float)
        return [float(p[0] / s + ox), float(p[1] / s + oy)]
    for r in strokes:
        r["p0"], r["p1"] = back(r["p0"]), back(r["p1"])
    return {
        "image_size": [int(gray.shape[1]), int(gray.shape[0])], "stroke_width": width / s,
        "stem": {"p0": back(sp0), "p1": back(sp1), "length": L / s, "angle": stem_angle, "support": min(1.0, support)},
        "ambiguous": False, "strokes": strokes,
        "unattached": [{"p0": back(u["p0"]), "p1": back(u["p1"])} for u in unattached],
    }
