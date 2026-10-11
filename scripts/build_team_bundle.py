"""Export trained P3 predictions and P2 impacts using P1's canonical snapshots."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.common import config
from src.common.records import to_records
from src.features.build import CUSTOMER_360_COLUMNS
from src.recommend.service import RecommendationService


def build_bundle(features, predictions, impacts, history=None):
    return {'schema_version': '1.0', 'data_mode': 'TEAM_OUTPUTS',
            'customers': to_records(features), 'predictions': predictions,
            'impacts': impacts, 'portfolio_history': history or []}


def main():
    features = pd.read_parquet(config.PROCESSED_DIR / 'features.parquet')
    splits = pd.read_parquet(config.PROCESSED_DIR / 'splits.parquet')
    features = features.merge(splits, on='customer_id', how='left')
    features['split'] = features['split'].fillna('scenario')
    view_path = config.PROCESSED_DIR / 'customer_360.parquet'
    if view_path.exists():
        # FR1 display fields (holdings, activity); never model inputs.
        view = pd.read_parquet(view_path, columns=['customer_id', *CUSTOMER_360_COLUMNS])
        features = features.merge(view, on='customer_id', how='left', validate='one_to_one')
    predictions = json.loads((config.PROCESSED_DIR / 'p3_predictions.json').read_text())
    impacts = json.loads((config.PROCESSED_DIR / 'impacts.json').read_text())
    history_path = config.PROCESSED_DIR / 'portfolio_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    bundle = build_bundle(features, predictions, impacts, history)
    backend = RecommendationService(bundle)
    path = config.PROCESSED_DIR / 'team_bundle.json'
    path.write_text(json.dumps(bundle, allow_nan=False))
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUTS_DIR / 'portfolio_summary.json').write_text(json.dumps(backend.summary(), indent=2, allow_nan=False))
    print(f'Validated TEAM_OUTPUTS: {len(features)} customers, {len(history)} trend snapshots -> {path}')
    return path


if __name__ == '__main__':
    main()
