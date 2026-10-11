"""Paths and config loaders shared by every component.

The product catalogue is loaded by src.recommend.load_catalogue (P5), not here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


@lru_cache
def settings() -> dict:
    return _load_yaml("settings.yaml")


@lru_cache
def reason_codes() -> dict:
    """Full reason-code file: {"reason_codes": {...}, "data_flags": {...}}."""
    return _load_yaml("reason_codes.yaml")


@lru_cache
def categories() -> dict:
    return _load_yaml("categories.yaml")
