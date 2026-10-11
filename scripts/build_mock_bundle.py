"""Build a labelled-MOCK integration bundle from the feature store, so P5 and P4
can test with hundreds or thousands of customers instead of the 10 demo fixtures.

Predictions are a transparent RULE stand-in, not a trained model: propensity is
the percentile rank of one signal per product, scaled to the product's target
take-up rate; reason codes and evidence use the real threshold gating
(src/models/reason_codes.py). Impacts are INSUFFICIENT_DATA until P2's engine
exists. The bundle is DEMO_MOCK with model_version "mock_rule_v0", so P5's
service will never treat it as team outputs.

    python scripts/build_mock_bundle.py [--features data/processed/features.parquet]
    python run_all.py --bundle data/processed/mock_team_bundle.json --serve app
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.common import config  # noqa: E402
from src.common.records import to_records  # noqa: E402
from src.models.reason_codes import evidence_codes, evidence_for, fit_thresholds  # noqa: E402
from src.recommend.service import RecommendationService  # noqa: E402

SIGNAL = {"GP01": "spend_fuel_m", "GP02": "est_gas_kwh_year",
          "GP03": "est_electricity_kwh_year", "GP04": "avg_app_logins_m"}
SNAPSHOT = ["customer_id", "age", "region", "income_band", "home_status", "property_type", "epc_rating",
            "main_heating_fuel", "has_solar", "vehicle_type", "statement_pref", "marketing_consent",
            "feature_cutoff_date", "data_completeness", "conflict_flags", "ev_replacement_use_case",
            "ev_purchase_use_case", "green_home_use_case", "solar_use_case", "requires_paper_statements"]


def build_bundle(features: pd.DataFrame, splits: pd.DataFrame | None = None) -> dict:
    train = features if splits is None else features[features["customer_id"].isin(
        splits.loc[splits["split"] == "train", "customer_id"])]
    thresholds = fit_thresholds(train)
    with open(config.CONFIG_DIR / "label_rules.yaml") as f:
        rates = {code: spec["target_rate"] for code, spec in yaml.safe_load(f)["products"].items()}
    ranks = {code: features[col].rank(pct=True).fillna(0) for code, col in SIGNAL.items()}
    rows = to_records(features)
    predictions = {}
    for i, row in enumerate(rows):
        preds = {}
        for code in SIGNAL:
            codes = evidence_codes(row, code, thresholds)
            preds[code] = {
                "propensity": round(min(1.0, 2 * rates[code] * float(ranks[code].iloc[i])), 4),
                "confidence": round(0.9 * row["data_completeness"], 4),
                "reason_codes": codes,
                "evidence": evidence_for(row, codes),
                "model_version": "mock_rule_v0",
                "data_mode": "DEMO_MOCK",
            }
        predictions[row["customer_id"]] = preds
    customers = to_records(features[SNAPSHOT])
    return {
        "schema_version": "1.0", "data_mode": "DEMO_MOCK", "customers": customers,
        "predictions": predictions,
        "impacts": {c["customer_id"]: {code: {"status": "INSUFFICIENT_DATA"} for code in SIGNAL} for c in customers},
        "portfolio_history": [],
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", type=Path, default=config.PROCESSED_DIR / "features.parquet")
    p.add_argument("--splits", type=Path, default=None, help="splits.parquet (thresholds fitted on train)")
    p.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "mock_team_bundle.json")
    args = p.parse_args(argv)
    features = pd.read_parquet(args.features)
    splits_path = args.splits or next((d / "splits.parquet" for d in (config.RAW_DIR, config.SAMPLE_DIR)
                                       if (d / "splits.parquet").exists()), None)
    splits = pd.read_parquet(splits_path) if splits_path else None
    bundle = build_bundle(features, splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, allow_nan=False) + "\n", encoding="utf-8")
    summary = RecommendationService.from_file(args.out).summary()  # same validation P5 applies
    print(f"wrote {args.out} ({len(bundle['customers'])} customers); P5 validation OK")
    print(f"  shown per product: {summary['opportunities_by_product']}; "
          f"suppressed rows: {summary['suppressed_opportunities']}")


if __name__ == "__main__":
    main()
