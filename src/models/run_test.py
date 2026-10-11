"""ONE-SHOT held-out test evaluation (submission item I).

    python -m src.models.run_test            # refuses if already run
    python -m src.models.run_test --force    # re-run is logged in outputs/test_runs.log
    python -m src.models.run_test --config path/to/model.yaml

Writes outputs/test_predictions.csv (all test customers x products) and
reports/metrics_test.json (metrics on the model population only).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from . import evaluate as ev
from .config import load_config, path
from .data import encoded_columns, load_tables
from .predict import score, to_p5_predictions
from src.common.records import to_records
from src.recommend.service import RecommendationService
from src.schemas.tables import TEST_PREDICTION_COLUMNS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    c = cfg["columns"]
    out_csv = path(cfg, "test_predictions")
    log = out_csv.parent / "test_runs.log"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if log.exists() and not args.force:
        raise SystemExit(f"Test set already evaluated (see {log}). Use --force only if models are frozen.")

    features, labels, splits = load_tables(cfg)
    test_ids = splits.loc[splits[c["split"]] == "test", c["id"]]
    scores = score(features[features[c["id"]].isin(test_ids)].reset_index(drop=True), cfg)
    lab = labels[[c["id"], c["product"], c["label"]]].astype({c["id"]: str})
    preds = scores.merge(lab, left_on=["customer_id", "product_code"], right_on=[c["id"], c["product"]], how="left")
    preds = preds[[*scores.columns, c["label"]]]
    test_features = features[features[c["id"]].isin(test_ids)].reset_index(drop=True)
    service = RecommendationService({
        "schema_version": "1.0", "data_mode": "TEAM_OUTPUTS",
        "customers": to_records(test_features.drop(columns=encoded_columns(cfg), errors="ignore")),
        "predictions": to_p5_predictions(scores), "impacts": {}})
    display = {(cid, r["product_code"]): r["eligible_to_show"] for cid in service.customers
               for r in service.recommendations(cid, include_suppressed=True)["recommendations"]}
    preds["eligible_to_show"] = [display[(r.customer_id, r.product_code)] for r in preds.itertuples()]
    preds[list(TEST_PREDICTION_COLUMNS)].to_csv(out_csv, index=False)

    metrics = {}
    for code, d in preds[preds["in_model_population"]].groupby("product_code"):
        metrics[code] = ev.metrics(d[c["label"]].to_numpy(), d["propensity"].to_numpy())
    (path(cfg, "reports_dir") / "metrics_test.json").write_text(json.dumps(metrics, indent=2))

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log, "a") as f:
        f.write(f"{stamp} force={args.force} versions={sorted(scores['model_version'].unique())}\n")
    for code, m in metrics.items():
        print(f"[{code}] TEST roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f} brier={m['brier']:.4f}")
    print(f"wrote {len(preds)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
