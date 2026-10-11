"""Repeatable memory demo (brief, slide 15): feedback changes what is shown, never the scores.

    python -m src.memory.demo                      # committed trained preview, customer C10144
    python -m src.memory.demo --customer C90001 --bundle data/processed/team_bundle.json

Runs through P5's RecommendationService (the same path as the API and UI) with an
in-memory store: show -> dismiss / not relevant -> cool-off and re-ordering -> take-up
-> erase. It then checks that propensity, confidence, eligibility and impact are
identical before and after, and exits non-zero if not.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from src.common import config
from src.recommend.service import RecommendationService

from .store import MemoryStore

T0 = date(2026, 7, 1)
PREVIEW = config.DATA_DIR / 'preview' / 'team_bundle.json'
PROTECTED = ('propensity', 'confidence', 'eligible_to_show', 'impact', 'ranking_score', 'rank')
# (day, actions the customer takes after seeing the cards); codes are filled in from the shown rows
SESSIONS = [(0, [(1, 'dismissed'), (0, 'clicked')]), (7, [(1, 'not_relevant')]), (45, [(0, 'took_up')]), (60, [])]


def protected_view(service: RecommendationService, cid: str) -> dict:
    return {r['product_code']: {k: r[k] for k in PROTECTED}
            for r in service.recommendations(cid, include_suppressed=True)['recommendations']}


def show(service: RecommendationService, cid: str, today: date) -> list[str]:
    rows = service.recommendations(cid, include_suppressed=True, on=today)['recommendations']
    shown = []
    for r in sorted(rows, key=lambda r: (r.get('memory', {}).get('display_rank') or 99, r['product_code'])):
        m = r.get('memory', {})
        if not r['eligible_to_show']:
            continue
        rank = f"#{m['display_rank']}" if m.get('display_rank') else 'hidden'
        print(f"  {rank:>7}  {r['product_code']} {r['product_name']:<18} propensity={r['propensity']:.3f}  "
              f"{m.get('action', 'SHOW')}: {m.get('reason', 'NO_MEMORY_EFFECT')}")
        if m.get('display_rank'):
            service.record_event(cid, r['product_code'], 'shown', today)
            shown.append(r['product_code'])
    return shown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=PREVIEW)
    parser.add_argument('--customer', default='C10144')
    args = parser.parse_args(argv)
    service = RecommendationService.from_file(args.bundle)
    service.memory_store = MemoryStore(':memory:')
    cid = args.customer
    if cid not in service.customers:
        raise SystemExit(f'{cid} is not in {args.bundle}')
    before = protected_view(service, cid)
    order = [r['product_code'] for r in service.recommendations(cid)['recommendations']]
    if len(order) < 2:
        raise SystemExit(f'{cid} needs at least two eligible products for this demo')
    for day, actions in SESSIONS:
        today = T0 + timedelta(days=day)
        print(f'\nSession on day {day} ({today}):')
        show(service, cid, today)
        for index, event in actions:
            code = order[index]
            service.record_event(cid, code, event, today)
            print(f"   -> customer {event.replace('_', ' ')}: {code}")
    print(f"\nErase memory: {service.forget_memory(cid)['deleted_events']} events deleted")
    show(service, cid, T0 + timedelta(days=61))
    after = protected_view(service, cid)
    unchanged = before == after
    print(f"\nPropensity, confidence, eligibility, impact and P5 rank unchanged: {'yes' if unchanged else 'NO'}")
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
