"""Adapter for the teammate's line detector (to be wired up when their code is available).

Must expose detect(gray: np.ndarray) -> dict in the schema documented in ogham_v2/detector_schema.py.
"""


def detect(gray):
    raise NotImplementedError("teammate detector not available yet - use the proxy detector (DR-017)")
