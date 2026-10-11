"""Evaluation cases built from REAL integrated rows (trained predictions + P2 impacts). Owner: P4.

    python -m src.llm.eval.build_integrated_set [--bundle data/processed/team_bundle.json]
        -> data/llm_eval/integrated_set.jsonl

The synthetic eval set (build_eval_set.py) tuned the prompt. This set checks the
deployed contract: rows come straight from P5's RecommendationService over a team
bundle (default: the committed trained preview), so reason codes, evidence keys,
impact methods and assumptions are exactly what the UI sends. Selection is
deterministic (sorted by customer ID). Per product:
  estimated_impact x4, insufficient_impact x2, low_confidence x2, data_flags x1,
  suppressed x2 (must get no text), prompt_injection x1 (free text added to evidence).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import config
from src.recommend.service import RecommendationService

from .build_eval_set import INJECTIONS

OUT = config.DATA_DIR / 'llm_eval' / 'integrated_set.jsonl'
PREVIEW = config.DATA_DIR / 'preview' / 'team_bundle.json'
QUOTA = {'estimated_impact': 4, 'insufficient_impact': 2, 'low_confidence': 2, 'data_flags': 1,
         'suppressed': 2, 'prompt_injection': 1}


def scenario_pools(rows: list[dict]) -> dict[str, list[dict]]:
    shown = [r for r in rows if r['eligible_to_show']]
    return {
        'estimated_impact': [r for r in shown if r['impact']['status'] == 'ESTIMATED' and r['confidence'] >= 0.5],
        'insufficient_impact': [r for r in shown if r['impact']['status'] != 'ESTIMATED'],
        'low_confidence': [r for r in shown if r['confidence'] < 0.5],
        'data_flags': [r for r in shown if r['data_flags']],
        'suppressed': [r for r in rows if not r['eligible_to_show']],
        'prompt_injection': [r for r in shown if r['impact']['status'] == 'ESTIMATED'],
    }


def spread(pool: list[dict], n: int) -> list[dict]:
    """n rows evenly spaced through the pool (deterministic, avoids neighbouring IDs)."""
    if len(pool) <= n:
        return pool
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def build(bundle_path: Path = PREVIEW) -> list[dict]:
    service = RecommendationService.from_file(bundle_path)
    all_rows = sorted((r for cid in service.customers
                       for r in service.recommendations(cid, include_suppressed=True)['recommendations']),
                      key=lambda r: (r['customer_id'], r['product_code']))
    cases, used = [], set()
    for code in service.catalogue['products']:
        pools = scenario_pools([r for r in all_rows if r['product_code'] == code])
        for scenario, n in QUOTA.items():
            pool = [r for r in pools[scenario] if (r['customer_id'], code, scenario != 'prompt_injection') not in used]
            for i, row in enumerate(spread(pool, n)):
                used.add((row['customer_id'], code, scenario != 'prompt_injection'))
                row = json.loads(json.dumps(row))
                if scenario == 'prompt_injection':
                    row['evidence']['customer_note'] = INJECTIONS[i % len(INJECTIONS)]
                cases.append({'case_id': f"{code}-{row['customer_id']}-{scenario}", 'split': 'integrated',
                              'tags': [code, scenario], 'expect_shown': row['eligible_to_show'], 'row': row})
    return cases


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=PREVIEW)
    args = parser.parse_args(argv)
    cases = build(args.bundle)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    counts = {}
    for c in cases:
        counts[c['tags'][1]] = counts.get(c['tags'][1], 0) + 1
    print(f'wrote {len(cases)} integrated cases from {args.bundle} to {OUT.relative_to(config.ROOT)}: {counts}')


if __name__ == '__main__':
    main()
