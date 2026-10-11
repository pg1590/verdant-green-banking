"""Evidence gating for reason codes (config/reason_codes.yaml). Owner: P3.

A reason code may be shown only if the customer's raw evidence crosses the
code's threshold, so HIGH_FUEL_SPEND never appears for someone spending £20.
P3 combines this with SHAP: emit codes that are among the top positive SHAP
contributors AND pass passes_threshold(). The rule-based stand-in scorer
(scripts/build_mock_bundle.py) uses the threshold check alone.

Threshold types: percentile (fitted on the TRAINING population with
fit_thresholds), absolute, flag, and external (skipped while its value is null).
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.common import config
from src.common.records import to_python


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or v is pd.NA


def fit_thresholds(train_features: pd.DataFrame) -> dict[str, object]:
    """Resolve every code's threshold to a concrete value (None = cannot be emitted)."""
    resolved = {}
    for code, spec in config.reason_codes()["reason_codes"].items():
        th = spec["threshold"]
        if th["type"] == "percentile":
            values = pd.to_numeric(train_features[spec["feature"]], errors="coerce").dropna()
            resolved[code] = float(values.quantile(th["value"] / 100)) if len(values) else None
        else:
            resolved[code] = th["value"]
    return resolved


def passes_threshold(code: str, row: Mapping, thresholds: Mapping[str, object]) -> bool:
    spec = config.reason_codes()["reason_codes"][code]
    value, th = row.get(spec["feature"]), thresholds.get(code)
    if _missing(value) or th is None:
        return False
    if spec["threshold"]["type"] == "flag":
        return value in th if isinstance(th, list) else value == th
    return float(value) >= float(th)


def evidence_codes(row: Mapping, product_code: str, thresholds: Mapping[str, object]) -> list[str]:
    """All of a product's codes whose evidence crosses the threshold, in dictionary order."""
    codes = config.reason_codes()["reason_codes"]
    return [c for c, spec in codes.items()
            if product_code in spec["products"] and passes_threshold(c, row, thresholds)]


def evidence_for(row: Mapping, codes: list[str]) -> dict:
    """The raw feature values behind each code (P5 requires non-empty evidence)."""
    specs = config.reason_codes()["reason_codes"]
    out = {}
    for c in codes:
        value = to_python(row.get(specs[c]["feature"]))
        out[specs[c]["feature"]] = round(value, 2) if isinstance(value, float) else value
    return out
