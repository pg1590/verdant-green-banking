"""P5 business views: opportunity table, dated trend and Customer 360 fields."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.recommend.service import HIGH_PRIORITY_LIFT, RecommendationService

PREVIEW = Path(__file__).resolve().parents[2] / 'data' / 'preview' / 'team_bundle.json'


@pytest.fixture(scope='module')
def preview():
    return json.loads(PREVIEW.read_text(encoding='utf-8'))


def test_opportunity_counts_reconcile_with_summary(preview):
    service = RecommendationService(preview)
    summary, table = service.summary(), service.opportunities()
    by_code = {p['product_code']: p for p in table['products']}
    assert {c: p['displayed_opportunities'] for c, p in by_code.items()} == summary['opportunities_by_product']
    assert sum(p['displayed_opportunities'] for p in by_code.values()) == summary['eligible_opportunities']
    for p in by_code.values():
        assert p['displayed_opportunities'] <= p['structurally_eligible']
        assert 0 <= p['high_priority'] <= p['displayed_opportunities']
        if p['high_priority_threshold'] is not None:
            assert p['high_priority_threshold'] == pytest.approx(HIGH_PRIORITY_LIFT * p['cohort_mean_propensity'], abs=2e-4)
        assert p['expected_takeups'] <= p['displayed_opportunities']
    assert 'Not incremental uplift' in table['definitions']['expected_takeups']


def test_every_product_has_a_supported_impact_example(preview):
    service = RecommendationService(preview)
    estimated = {r['product_code'] for rows in service._rows.values() for r in rows
                 if r['eligible_to_show'] and r['impact']['status'] == 'ESTIMATED'}
    assert estimated == {'GP01', 'GP02', 'GP03', 'GP04'}
    metrics = service.summary()['potential_impact']
    assert all(metrics[k]['value'] and metrics[k]['coverage_customers'] > 0 for k in metrics)


def test_dated_history_with_metadata_is_accepted_and_bad_history_rejected(preview):
    history = [{'date': d, 'customers_with_opportunities': n, 'window_start': '2025-07-01',
                'window_end': d, 'series': 'RETROSPECTIVE_RESCORING_FIXED_COHORT',
                'opportunities_by_product': {'GP01': 1}}
               for d, n in [('2026-06-30', 120), ('2026-07-31', 118)]]
    service = RecommendationService({**preview, 'portfolio_history': history})
    assert service.summary()['portfolio_history'] == history
    with pytest.raises(ValueError, match='unique and increasing'):
        RecommendationService({**preview, 'portfolio_history': history[::-1]})
    with pytest.raises(ValueError, match='nonnegative integers'):
        RecommendationService({**preview, 'portfolio_history': [{**history[0], 'customers_with_opportunities': 1.5}]})


def test_customer_360_fields_are_validated(preview):
    bundle = deepcopy(preview)
    bundle['customers'][0].update(products_held=['current_account', 'mortgage'], n_transactions_window=310,
                                  active_months_window=12, last_transaction_date='2026-06-28')
    RecommendationService(bundle)
    bundle['customers'][0]['products_held'] = ['green_mortgage']
    with pytest.raises(ValueError, match='products_held'):
        RecommendationService(bundle)
    bundle['customers'][0].update(products_held=[], n_transactions_window=-1)
    with pytest.raises(ValueError, match='n_transactions_window'):
        RecommendationService(bundle)


def test_portfolio_cache_refreshes_after_feedback_from_any_process(preview, tmp_path):
    from datetime import date
    from src.memory.store import MemoryStore
    db = tmp_path / 'feedback.db'
    service = RecommendationService(preview, memory_store=MemoryStore(db))
    cid = next(c for c in service.customers if service.recommendations(c)['recommendations'])
    code = service.recommendations(cid)['recommendations'][0]['product_code']
    before = service.summary()
    assert service.summary() == before                      # served from the cache
    service.record_event(cid, code, 'dismissed', date.today())
    after = service.summary()
    assert after['eligible_opportunities'] == before['eligible_opportunities'] - 1
    service.forget_memory(cid)
    assert service.summary() == before
    # Another process (the Streamlit app vs the API) writes to the same database.
    MemoryStore(db).record(cid, code, 'not_relevant', date.today())
    assert service.opportunities()['products'] != RecommendationService(preview).opportunities()['products']
    assert service.summary()['eligible_opportunities'] == before['eligible_opportunities'] - 1


def test_cached_views_are_copies(preview):
    service = RecommendationService(preview)
    service.summary()['opportunities_by_product']['GP01'] = -1
    assert service.summary()['opportunities_by_product']['GP01'] >= 0


def test_engagement_funnel_counts_distinct_customers_and_refreshes(preview, tmp_path):
    from datetime import date
    from src.memory.store import MemoryStore
    service = RecommendationService(preview, memory_store=MemoryStore(tmp_path / 'f.db'))
    assert all(p['shown'] == 0 and p['click_rate'] is None for p in service.engagement()['products'])
    ids = list(service.customers)[:3]
    for cid in ids:
        service.record_event(cid, 'GP04', 'shown', date.today())
        service.record_event(cid, 'GP04', 'shown', date.today())   # repeat views count once
    service.record_event(ids[0], 'GP04', 'clicked', date.today())
    service.record_event(ids[1], 'GP04', 'not_relevant', date.today())
    gp04 = next(p for p in service.engagement()['products'] if p['product_code'] == 'GP04')
    assert (gp04['shown'], gp04['clicked'], gp04['not_relevant']) == (3, 1, 1)
    assert gp04['click_rate'] == round(1 / 3, 4) and gp04['decline_rate'] == round(1 / 3, 4)
