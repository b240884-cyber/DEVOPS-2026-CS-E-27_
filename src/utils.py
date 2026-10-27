"""Shared utilities for data loading, paths, persistence, and logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "model" if (PROJECT_ROOT / "model").exists() else PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
UCI_STUDENT_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"


def configure_logging() -> None:
    """Configure concise console logging for scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories(directory: Path | None = None) -> None:
    """Create project artifact directories if they do not exist."""
    dirs = [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]
    if directory:
        dirs.append(directory)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    """Persist a JSON artifact with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Load a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(obj: object, path: Path) -> None:
    """Save a Python artifact with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> object:
    """Load a joblib artifact with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)
