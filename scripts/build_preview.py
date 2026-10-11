"""Rebuild the committed preview (data/preview/team_bundle.json) from a full local pipeline run.

    python scripts/build_preview.py

Scope: the first 200 population customers (C10001-C10200) plus the 13 judge scenarios,
with their trained predictions, P2 impacts and Customer 360 fields copied unchanged from
data/processed/team_bundle.json. The trend is recomputed for this 213-customer cohort, so
preview counts reconcile with the preview's own summary.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config
from src.features.snapshots import build_history, load_inputs
from src.recommend.service import RecommendationService

N_POPULATION = 200
OUT = config.DATA_DIR / 'preview' / 'team_bundle.json'


def main():
    full = json.loads((config.PROCESSED_DIR / 'team_bundle.json').read_text(encoding='utf-8'))
    population = [c['customer_id'] for c in full['customers'] if not c['customer_id'].startswith('C9')]
    ids = sorted(population)[:N_POPULATION] + sorted(c['customer_id'] for c in full['customers']
                                                     if c['customer_id'].startswith('C9'))
    keep = set(ids)
    preview = {
        'schema_version': full['schema_version'], 'data_mode': full['data_mode'],
        'customers': [c for c in full['customers'] if c['customer_id'] in keep],
        'predictions': {k: v for k, v in full['predictions'].items() if k in keep},
        'impacts': {k: v for k, v in full['impacts'].items() if k in keep},
        'portfolio_history': build_history(*load_inputs(), customer_ids=ids),
        'bundle_scope': 'PREVIEW',
        'training_population_customers': sum(not c['customer_id'].startswith('C9') for c in full['customers']),
    }
    summary = RecommendationService(preview).summary()
    if summary['customers_with_opportunities'] != preview['portfolio_history'][0]['customers_with_opportunities']:
        raise ValueError('Preview trend does not reconcile with the preview summary at the cutoff')
    OUT.write_text(json.dumps(preview, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    print(f"Preview: {len(ids)} customers, {len(preview['portfolio_history'])} snapshots -> {OUT}")


if __name__ == '__main__':
    main()
