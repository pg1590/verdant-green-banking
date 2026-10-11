"""Output contract of the propensity subsystem (P3 -> P5 recommendation engine, P4 LLM layer).

One row per customer x product in `data/processed/scores.parquet`. Every customer is scored
for every product: eligibility/suppression is decided downstream by P5, never here.
"""
from __future__ import annotations

import json

import pandas as pd

PRODUCTS = {
    "GP01": "EV_FINANCE",
    "GP02": "GREEN_HOME",
    "GP03": "SOLAR_FINANCE",
    "GP04": "PAPERLESS_BANKING",
}

SCORE_COLUMNS = {
    "customer_id": "str",
    "product_code": "str: GP01..GP04",
    "opportunity": "str: EV_FINANCE | GREEN_HOME | SOLAR_FINANCE | PAPERLESS_BANKING",
    "propensity": "float in [0,1]: calibrated probability of take-up",
    "confidence": "float in [0,1]: reliability of the propensity "
                  "(completeness x ensemble agreement x data quality, src/models/confidence.py)",
    "reason_codes": "JSON list[str]: codes from config/reason_codes.yaml, max 3, may be empty",
    "evidence": "JSON dict: {evidence_key: raw value} for each emitted reason code",
    "in_model_population": "bool: P5 catalogue eligibility with purpose='training' (the population the "
                           "model was trained and evaluated on; NOT display eligibility)",
    "model_version": "str, e.g. gp01_lgbm_v3",
    "feature_cutoff_date": "str ISO date: data timestamp for FR7",
}


def validate_scores(df: pd.DataFrame, reason_codes: dict | None = None) -> None:
    """Raise AssertionError if `df` breaks the contract. P5 can call this in their tests."""
    missing = set(SCORE_COLUMNS) - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    assert df["product_code"].isin(PRODUCTS.keys()).all(), "unknown product_code"
    assert df["propensity"].between(0, 1).all(), "propensity outside [0,1]"
    assert df["confidence"].between(0, 1).all(), "confidence outside [0,1]"
    assert not df.duplicated(["customer_id", "product_code"]).any(), "duplicate customer x product"
    if reason_codes is not None:
        for codes in df["reason_codes"]:
            unknown = set(json.loads(codes)) - set(reason_codes)
            assert not unknown, f"reason codes not in dictionary: {unknown}"
