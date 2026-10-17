"""Line detectors behind one interface: detect(gray uint8 image) -> detector JSON dict (see detector_schema)."""


def get(name: str = "proxy"):
    if name == "proxy":
        from .proxy import detect
        return detect
    if name == "teammate":
        from .teammate import detect
        return detect
    raise ValueError(name)
