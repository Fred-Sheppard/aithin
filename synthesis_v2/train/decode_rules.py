"""Rule-based decoder: detector JSON -> letter (E3-ref; also used to score detectors).

Group from stroke side/angle (right -> B, left -> H, across: |angle-90| > diag_thresh -> M else A); count = #strokes.
"""
import numpy as np

GROUP_LETTERS = {"B": ["B", "L", "F", "S", "N"], "H": ["H", "D", "T", "C", "Q"],
                 "M": ["M", "G", "NG", "Z", "R"], "A": ["A", "O", "U", "E", "I"]}


def decode(det: dict, diag_thresh: float = 17.0):
    strokes = det.get("strokes") or []
    if not det.get("stem") or not strokes:
        return None, {"reason": "no_stem_or_strokes"}
    sides = [s["side"] for s in strokes]
    groups = []
    for s in strokes:
        if s["side"] == "right":
            groups.append("B")
        elif s["side"] == "left":
            groups.append("H")
        else:
            groups.append("M" if abs(s["angle"] - 90) > diag_thresh else "A")
    vals, counts = np.unique(groups, return_counts=True)
    group = str(vals[np.argmax(counts)])
    n = min(5, len(strokes))
    return GROUP_LETTERS[group][n - 1], {"groups": groups, "sides": sides, "count": len(strokes)}
