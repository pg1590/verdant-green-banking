"""Seeded P1 → P2 → P3 → P4 → P5 integration; test evaluation is opt-in."""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=None, help='Normal customers; P1 also adds fixed judge scenarios')
    parser.add_argument('--fast', action='store_true', help='P3 fixed parameters instead of grid search')
    parser.add_argument('--skip-generate', action='store_true', help='Reuse existing data/raw tables')
    parser.add_argument('--evaluate-test', action='store_true', help='Run P3 held-out test once after models are frozen')
    parser.add_argument('--live-llm', action='store_true', help='Allow configured live API calls while warming explanations')
    args = parser.parse_args(argv)
    from src.data_gen.generate import main as generate
    from src.categorisation.model import main as categorise
    from src.categorisation.evaluate import main as categoriser_eval
    from src.features.build import main as features
    from src.models.train import main as train
    from src.models.predict import main as predict
    from src.carbon.engine import main as impacts
    from src.features.snapshots import main as snapshots
    from scripts.build_team_bundle import main as export
    from src.llm.explain import main as explanations

    if not args.live_llm:
        os.environ['LLM_PROVIDER'] = 'offline'
    stages = []
    if not args.skip_generate:
        stages.append(('P1 synthetic data', lambda: generate(['--n', str(args.n)] if args.n else [])))
    categorised = config.PROCESSED_DIR / 'transactions_categorised.parquet'
    def copy_supervision():
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name in ('labels', 'splits'):
            shutil.copy2(config.RAW_DIR / f'{name}.parquet', config.PROCESSED_DIR / f'{name}.parquet')
    stages.extend([
        ('P1 labels and splits', copy_supervision),
        ('P2 trained categorisation', lambda: categorise(['--data-dir', str(config.RAW_DIR)])),
        ('P2 merchant-holdout validation', lambda: categoriser_eval(['--data-dir', str(config.RAW_DIR), '--split', 'val'])),
        ('P1/P2 feature store', lambda: features(['--source', str(config.RAW_DIR), '--categorised', str(categorised)])),
        ('P3 training and validation', lambda: train(['--fast'] if args.fast else [])),
        ('P3 predictions', predict),
        ('P2 sourced impacts or explicit missing data', impacts),
        ('P1/P5 dated trend snapshots (retrospective rescoring)', snapshots),
        ('P5 validated team bundle', export),
        ('P4 validated explanations', explanations),
    ])
    for name, run in stages:
        start = time.monotonic()
        print(f'==> {name}', flush=True)
        run()
        print(f'    finished in {time.monotonic() - start:.1f}s', flush=True)
    if args.evaluate_test:
        from src.models.run_test import main as evaluate_test
        evaluate_test([])
    print('Ready: make app or make api (uses data/processed/team_bundle.json).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
