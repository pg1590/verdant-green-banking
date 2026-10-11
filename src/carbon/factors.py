"""Load config/emission_factors.csv. Owner: P2.

A blank value means "not yet copied from the source file". Using a blank factor
raises instead of silently defaulting, so no estimate is ever built on a guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from src.common import config


class MissingFactorError(KeyError):
    pass


@dataclass(frozen=True)
class Factor:
    factor_id: str
    kind: str
    value: float
    unit: str
    source: str
    edition: str
    reference: str


@lru_cache
def _table() -> pd.DataFrame:
    return pd.read_csv(config.CONFIG_DIR / "emission_factors.csv", dtype={"edition": str}).set_index("factor_id")


def get(factor_id: str) -> Factor:
    t = _table()
    if factor_id not in t.index:
        raise MissingFactorError(f"unknown factor {factor_id!r}")
    row = t.loc[factor_id]
    if pd.isna(row["value"]):
        raise MissingFactorError(f"factor {factor_id!r} has no value yet; fill it from {row['source']}")
    return Factor(factor_id, row["kind"], float(row["value"]), row["unit"], row["source"],
                  "" if pd.isna(row["edition"]) else row["edition"],
                  "" if pd.isna(row["reference"]) else row["reference"])


def missing_values() -> list[str]:
    """Factor ids still blank (handy for a checklist before code freeze)."""
    t = _table()
    return t.index[t["value"].isna()].tolist()
