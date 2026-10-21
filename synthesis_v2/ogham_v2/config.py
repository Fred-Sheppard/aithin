"""Generator configs: YAML files in configs/. A config may set `base: <name>` to inherit from another one
(deep merge; a value of null removes a section), e.g. `base: default` plus a few overrides."""
import copy
from pathlib import Path

import yaml

from .paths import CONFIGS


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(name_or_path) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIGS / f"{name_or_path}.yaml"
    raw = yaml.safe_load(p.read_text()) or {}
    base = raw.pop("base", None)
    cfg = _merge(load(base), raw) if base else raw
    cfg["name"] = raw.get("name", p.stem)
    return cfg
