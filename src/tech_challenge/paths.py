from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"

BREAST_CANCER_DATASET = DATA_DIR / "breast-cancer-wisconsin-diagnostic-data-set.csv"
BEST_MODEL = MODEL_DIR / "best_model.pkl"
AG_EXPERIMENT_RESULTS = RESULTS_DIR / "ag_experiment_results.json"
