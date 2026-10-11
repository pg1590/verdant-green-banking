#!/usr/bin/env python3
"""P5 reproducible integration entry point; never pretends to train teammate models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from src.recommend.demo import write_demo
from src.recommend.service import RecommendationService

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Explicit TEAM_OUTPUTS JSON integration bundle')
    parser.add_argument('--serve', choices=['app', 'api'], help='Launch UI or API after validation')
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--demo', action='store_true', help='Use explicitly labelled mock fixtures')
    parser.add_argument('--n', type=int, help='Customers for a new seeded team pipeline')
    parser.add_argument('--fast', action='store_true', help='Use P3 fixed training parameters')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.bundle:
        bundle = args.bundle.resolve()
        service = RecommendationService.from_file(bundle)
    elif args.demo:
        bundle = write_demo(ROOT / 'data' / 'processed' / 'demo_bundle.json')
        service = RecommendationService.from_file(bundle)
    elif args.serve and args.n is None and not (ROOT / 'data/processed/team_bundle.json').exists() and (ROOT / 'data/preview/team_bundle.json').exists():
        bundle = ROOT / 'data/preview/team_bundle.json'
        service = RecommendationService.from_file(bundle)
    else:
        bundle = ROOT / 'data' / 'processed' / 'team_bundle.json'
        if not bundle.exists() or args.n is not None:
            from scripts.run_pipeline import main as pipeline
            pipeline_args = (['--n', str(args.n)] if args.n is not None else [])
            if args.fast:
                pipeline_args.append('--fast')
            pipeline(pipeline_args)
        service = RecommendationService.from_file(bundle)
    print(f'Validated {len(service.customers)} customers · {service.mode}')
    print(json.dumps(service.summary(), indent=2, allow_nan=False))
    if not args.skip_tests:
        env = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
        # Tests use their own explicit fixtures, independent of a supplied bundle.
        env.pop('GREEN_BUNDLE_PATH', None)
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/scenarios'], check=True, env=env)
    if args.serve:
        env = {**os.environ, 'GREEN_BUNDLE_PATH': str(bundle)}
        command = [sys.executable, '-m', 'streamlit', 'run', 'app/main.py'] if args.serve == 'app' else [sys.executable, '-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', '8000']
        subprocess.run(command, check=True, env=env)


if __name__ == '__main__':
    main()
