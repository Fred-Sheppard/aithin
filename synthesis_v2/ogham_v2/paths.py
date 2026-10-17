from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # aithin/synthesis_v2
REPO = ROOT.parent                                   # aithin
PROJECT = REPO.parent                                # AI-Project

DATA = ROOT / "data"
RUNS = ROOT / "runs"
DOCS = ROOT / "docs"
RESULTS = DOCS / "results"
EVAL = ROOT / "eval"
CONFIGS = ROOT / "configs"

LABEL_MAP = REPO / "model" / "label_to_letter.json"
LEGACY_SYNTHESIS = REPO / "synthesis"
REAL_SETS = {
    "marker": PROJECT / "ogham-marker-labelled",
    "pen": PROJECT / "ogham-pen-labelled",
}
LEGACY_CHECKPOINTS = {
    "hybrid": Path.home() / "Desktop/ISE/Block-7/AI/Project/model/best_ogham_hybrid_model_save.h5",
    "augmented": Path.home() / "Desktop/ISE/Block-7/AI/Project/model/best_ogham_augmented_model.h5",
}
