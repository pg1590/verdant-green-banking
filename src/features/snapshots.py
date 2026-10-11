"""Comparable dated portfolio snapshots for the business trend. Owners: P1 (inputs), P5 (aggregates).

Each snapshot rebuilds the feature store over its own 12-month window ending on the
snapshot date, from the same raw tables and P2 categories. Nothing dated after the
snapshot is read. Data starts 2025-07-01, so only month-ends from the canonical
cutoff (2026-06-30) to data_end (2026-09-30) have a full 12-month window; earlier
dates would need a shorter, non-comparable window and are not used.

Interpretation (docs/data.md "Trend snapshots"):
- Same customers in every snapshot (fixed cohort), same catalogue policy, same impact method.
- The frozen current models rescore every snapshot. This is retrospective rescoring,
  not a record of models that were deployed at the time.
- Profile attributes are the 2026-06-30 snapshot; later snapshots see change only
  through transactions, digital activity and dated holdings.
- Post-cutoff snapshots overlap the label window. They are for display only and are
  never used for model evaluation.

    python -m src.features.snapshots      # -> data/processed/portfolio_history.json
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from src.carbon.engine import compute_impacts
from src.common import config
from src.common.records import to_records
from src.models import config as model_config
from src.models.config import feature_list, load_config, path
from src.models.predict import score, to_p5_predictions
from src.recommend.service import RecommendationService
from .build import TXN_COLUMNS, build_features, load_tables, read_categories

SNAPSHOT_DATES = ('2026-06-30', '2026-07-31', '2026-08-31', '2026-09-30')
SERIES = 'RETROSPECTIVE_RESCORING_FIXED_COHORT'


def window(date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(date)
    return end - pd.DateOffset(months=12) + pd.Timedelta(1, unit='D'), end


def frozen_bundles(cfg: dict) -> dict:
    """The current model files. Unlike predict.load_bundles there is no training-data hash
    check: snapshot features differ from the training store by design."""
    bundles = {}
    for code in cfg['products']:
        # P3 v3 names files with bundle_stem(); v2 used model_version(). Support both during the merge.
        stem = (model_config.bundle_stem(cfg, code) if hasattr(model_config, 'bundle_stem')
                else model_config.model_version(cfg, code))
        bundle = joblib.load(path(cfg, 'models_dir') / f'{stem}.pkl')
        if bundle['features'] != feature_list(cfg, code):
            raise ValueError(f'{code}: model configuration changed; retrain before building snapshots')
        bundles[code] = bundle
    return bundles


def snapshot_summary(features: pd.DataFrame, splits: pd.DataFrame, cfg: dict, bundles: dict) -> dict:
    """Score one snapshot and aggregate it with the same P5 service as the dashboard."""
    predictions = to_p5_predictions(score(features, cfg, bundles))
    records = to_records(features.merge(splits, on='customer_id', how='left').fillna({'split': 'scenario'}))
    impacts = {r['customer_id']: compute_impacts(r) for r in records}
    bundle = {'schema_version': '1.0', 'data_mode': 'TEAM_OUTPUTS', 'customers': records,
              'predictions': predictions, 'impacts': impacts, 'portfolio_history': []}
    return RecommendationService(bundle).summary()


def build_history(tables: dict, categories: pd.DataFrame, splits: pd.DataFrame,
                  dates=SNAPSHOT_DATES, customer_ids=None) -> list[dict]:
    """`customer_ids` restricts the fixed cohort (e.g. the committed preview) after features are built."""
    cfg = load_config()
    bundles = frozen_bundles(cfg)
    history = []
    for date in dates:
        start, end = window(date)
        features = build_features(tables, categories, cutoff=end, start=start)
        if customer_ids is not None:
            features = features[features['customer_id'].isin(set(customer_ids))].reset_index(drop=True)
        summary = snapshot_summary(features, splits, cfg, bundles)
        history.append({
            'date': end.strftime('%Y-%m-%d'),
            'customers_with_opportunities': summary['customers_with_opportunities'],
            'eligible_opportunities': summary['eligible_opportunities'],
            'opportunities_by_product': summary['opportunities_by_product'],
            'kg_co2e_per_year': summary['potential_impact']['kg_co2e_per_year']['value'],
            'customers_analysed': summary['customers_analysed'],
            'window_start': start.strftime('%Y-%m-%d'), 'window_end': end.strftime('%Y-%m-%d'),
            'model_versions': sorted({b['model_version'] for b in bundles.values()}),
            'series': SERIES,
        })
        print(f"  {history[-1]['date']}: {history[-1]['customers_with_opportunities']} customers with opportunities")
    return history


def load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    categories = read_categories(config.RAW_DIR, config.PROCESSED_DIR / 'transactions_categorised.parquet')
    tables = load_tables(config.RAW_DIR, TXN_COLUMNS)
    return tables, categories, pd.read_parquet(config.PROCESSED_DIR / 'splits.parquet')


def main() -> None:
    history = build_history(*load_inputs())
    out = config.PROCESSED_DIR / 'portfolio_history.json'
    out.write_text(json.dumps(history, indent=2, allow_nan=False))
    print(f'{len(history)} dated snapshots -> {out}')


if __name__ == '__main__':
    main()
