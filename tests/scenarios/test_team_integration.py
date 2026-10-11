from copy import deepcopy
from datetime import date
import asyncio

import pytest

from src.carbon.engine import compute_impacts
from src.carbon.factors import get
from src.llm.client import CacheOnlyProvider, ResponseCache
from src.memory.store import MemoryStore
from src.models.config import load_config, load_reason_codes
from src.recommend.demo import demo_bundle
from src.recommend.service import RecommendationService
from src.recommend.validation import validate_team_customer


def test_memory_cools_off_offer_without_changing_model_or_eligibility():
    store = MemoryStore(':memory:')
    backend = RecommendationService(demo_bundle(), memory_store=store)
    before = deepcopy(backend._rows['C10001'])
    backend.record_event('C10001', 'GP01', 'dismissed')
    assert 'GP01' not in {r['product_code'] for r in backend.recommendations('C10001')['recommendations']}
    audit = next(r for r in backend.recommendations('C10001', include_suppressed=True)['recommendations'] if r['product_code'] == 'GP01')
    assert audit['eligible_to_show'] is True
    assert audit['memory']['reason'] == 'DISMISSED_COOL_OFF'
    assert backend._rows['C10001'] == before
    assert backend.forget_memory('C10001')['deleted_events'] == 1
    assert 'GP01' in {r['product_code'] for r in backend.recommendations('C10001')['recommendations']}


def test_api_feedback_and_erasure(monkeypatch):
    import api.main as api
    import httpx
    backend = RecommendationService(demo_bundle(), memory_store=MemoryStore(':memory:'))
    monkeypatch.setattr(api, 'get_service', lambda: backend)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
            response = await client.post('/customers/C10001/events', json={'product_code':'GP01', 'event':'dismissed'})
            assert response.status_code == 200
            insights = (await client.get('/customers/C10001/insight')).json()['insights']
            assert 'GP01' not in {r['product_code'] for r in insights}
            assert (await client.delete('/customers/C10001/memory')).json()['deleted_events'] == 1
            assert (await client.post('/customers/unknown/events', json={'product_code':'GP01', 'event':'clicked'})).status_code == 404
            assert (await client.post('/customers/C10001/events', json={'product_code':'GP01', 'event':'bad'})).status_code == 422
    asyncio.run(check())


def test_ev_carbon_formula_and_missing_assumptions():
    row = {'vehicle_type': 'petrol', 'num_vehicles': 1, 'est_annual_km': 12000, 'paper_statements_12m': 12}
    impacts = compute_impacts(row)
    ev = impacts['GP01']
    expected = 12000 * (get('petrol_car_avg_per_km').value - get('bev_car_avg_per_km').value - get('bev_car_avg_td_per_km').value)
    assert ev['status'] == 'ESTIMATED'
    assert ev['kg_co2e_per_year'] == pytest.approx(expected, abs=0.001)
    assert ev['factor_edition'] == 'DESNZ GHG Conversion Factors 2026'
    # No gas heating, roof or electricity use is known here, so Home and Solar stay unavailable.
    assert all(impacts[code]['status'] == 'INSUFFICIENT_DATA' for code in ['GP02','GP03'])
    assert impacts['GP04']['status'] == 'ESTIMATED' and impacts['GP04']['paper_sheets_per_year'] == 12
    assert compute_impacts({**row, 'vehicle_type':'ev'})['GP01']['status'] == 'INSUFFICIENT_DATA'
    assert compute_impacts({**row, 'num_vehicles':2})['GP01']['status'] == 'INSUFFICIENT_DATA'


def test_hidden_persona_is_rejected():
    row = deepcopy(demo_bundle()['customers'][0])
    row['persona'] = 'Eco-minded professional'
    with pytest.raises(ValueError, match='persona'):
        validate_team_customer(row)


def test_models_exclude_age_and_freeze_percentiles_on_train():
    import pandas as pd
    cfg = load_config()
    assert all('age' not in p['features'] for p in cfg['products'].values())
    train = pd.DataFrame({'spend_fuel_m':[10,20,30,40], 'est_annual_km':[1000,2000,3000,4000],
                         'spend_automotive_m':[1,2,3,4], 'spend_home_improvement_m':[10,20,30,40],
                         'avg_app_logins_m':[2,4,6,8]})
    dictionary = load_reason_codes(cfg, train)
    assert dictionary['HIGH_FUEL_SPEND']['threshold'] == 32.5
    assert load_reason_codes(cfg)['HIGH_FUEL_SPEND']['threshold'] is None


def test_cache_replays_without_sdk_or_credentials(tmp_path):
    cache = ResponseCache(tmp_path)
    provider = CacheOnlyProvider('gemini', 'test-model')
    provider.cache = cache
    key = cache.key('gemini', 'test-model', 'system', 'user', {'type':'object'})
    cache.put(key, {'output': {'headline':'Cached', 'text':'Validated cached body'}, '_meta': {}})
    assert provider.complete_json('system', 'user', {'type':'object'})['headline'] == 'Cached'
    assert provider.last_meta['cache_hit'] is True


def test_partial_window_uses_only_its_tariff_and_annualises_standing_charge():
    from src.carbon.prices import window_energy_prices
    prices = window_energy_prices('2026-04-01', '2026-06-30')
    assert prices['elec_unit'] == pytest.approx(0.2467)
    assert prices['elec_standing_year'] == pytest.approx(0.5721 * 365)
    # Tariffs are sourced to 2026-09-30 (the last trend snapshot); later windows must fail.
    with pytest.raises(ValueError, match='entire feature window'):
        window_energy_prices('2026-04-01', '2026-10-31')
