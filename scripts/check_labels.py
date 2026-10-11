"""Label sanity report for P1/P3: are the synthetic labels learnable, but not trivially?

For each product, on its eligible population: take-up rate, validation AUC of a
quick LightGBM vs the single-signal rule baseline. Healthy v0 labels: LightGBM
roughly 0.65-0.85 and above the rule. Near 1.0 means labels leak from features
(hidden traits not doing their job); near 0.5 means nothing is learnable.
This is a generator check only; P3's real evaluation lives in src/models/.

    python scripts/check_labels.py [--source data/raw]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.common import config  # noqa: E402
from src.common.records import to_records  # noqa: E402
from src.recommend import eligible_customer_product_pairs  # noqa: E402
from src.schemas.tables import NON_MODEL_COLUMNS  # noqa: E402

RULE_SIGNAL = {"GP01": "spend_fuel_m", "GP02": "est_gas_kwh_year",
               "GP03": "est_electricity_kwh_year", "GP04": "avg_app_logins_m"}


def model_matrix(features: pd.DataFrame) -> pd.DataFrame:
    drop = [*NON_MODEL_COLUMNS, *config.settings()["model"]["excluded_features"]]
    X = features.drop(columns=drop).set_index(features["customer_id"])
    for col in X.columns:
        dtype = str(X[col].dtype)
        if dtype in ("object", "str", "string"):
            X[col] = X[col].astype("category")
        elif dtype in ("boolean", "Int64", "bool"):
            X[col] = X[col].astype(float)
    return X


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, default=config.RAW_DIR)
    p.add_argument("--features", type=Path, default=config.PROCESSED_DIR / "features.parquet")
    args = p.parse_args(argv)
    features = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.source / "labels.parquet")
    split = pd.read_parquet(args.source / "splits.parquet").set_index("customer_id")["split"]
    pairs = pd.DataFrame(eligible_customer_product_pairs(to_records(features)))
    X = model_matrix(features)
    print(f"{'product':8}{'eligible':>9}{'take-up':>9}{'val pos':>8}{'LGBM AUC':>10}{'rule AUC':>10}")
    for code, signal in RULE_SIGNAL.items():
        ids = pairs.loc[(pairs.product_code == code) & pairs.eligible_to_show, "customer_id"]
        y = labels[labels.product_code == code].set_index("customer_id")["took_up"].reindex(ids)
        tr, va = ids[split.reindex(ids).values == "train"], ids[split.reindex(ids).values == "val"]
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1)
        model.fit(X.loc[tr], y.loc[tr])
        auc = roc_auc_score(y.loc[va], model.predict_proba(X.loc[va])[:, 1])
        rule = roc_auc_score(y.loc[va], X.loc[va, signal].fillna(0))
        print(f"{code:8}{len(ids):>9}{y.mean():>9.1%}{int(y.loc[va].sum()):>8}{auc:>10.3f}{rule:>10.3f}")


if __name__ == "__main__":
    main()
