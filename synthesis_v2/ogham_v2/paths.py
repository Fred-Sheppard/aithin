from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # aithin/synthesis_v2
REPO = ROOT.parent                                   # aithin

CONFIGS = ROOT / "configs"
LABEL_MAP = REPO / "model" / "label_to_letter.json"
