"""Shared paths for the IN014 pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEANED = ROOT / "data" / "cleaned"
FIGURES = ROOT / "outputs" / "charts"
METRICS = ROOT / "outputs" / "metrics"
MODELS = ROOT / "outputs" / "models"
SAMPLES = ROOT / "outputs" / "samples"
REPORTS = ROOT / "outputs" / "reports"


def ensure_output_dirs() -> None:
    for d in (CLEANED, FIGURES, METRICS, MODELS, SAMPLES, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
