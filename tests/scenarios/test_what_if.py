"""What-if simulator: changed circumstances flow through rules, models and impact; nothing is saved."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from src.recommend.service import RecommendationService

PREVIEW = Path(__file__).resolve().parents[2] / 'data' / 'preview' / 'team_bundle.json'


@pytest.fixture(scope='module')
def service():
    return RecommendationService(json.loads(PREVIEW.read_text(encoding='utf-8')))


def by_code(result):
    return {p['product_code']: p for p in result['products']}


def test_selling_the_car_withdraws_ev_finance(service):
    result = service.what_if('C90001', {'vehicle_type': 'none'})
    ev = by_code(result)['GP01']
    assert ev['before']['eligible_to_show'] and not ev['after']['eligible_to_show']
    assert 'NO_EV_PURCHASE_USE_CASE' in ev['after']['suppression_reasons']
    assert result['changes'] == {'vehicle_type': 'none'} and result['scope'] == 'SIMULATION_NOT_SAVED'
    assert any('car count was set to 0' in n for n in result['notes'])


def test_becoming_a_homeowner_opens_home_and_solar(service):
    result = by_code(service.what_if('C90007', {'home_status': 'owner_mortgage', 'property_type': 'semi_detached'}))
    for code in ('GP02', 'GP03'):
        assert not result[code]['before']['eligible_to_show'] and result[code]['after']['eligible_to_show']


def test_installing_solar_removes_the_solar_estimate(service):
    solar = by_code(service.what_if('C10144', {'has_solar': True}))['GP03']
    assert solar['after']['impact_reason'] == 'EXISTING_OR_UNKNOWN_SOLAR_INSTALLATION'
    assert 'ALREADY_HAS_SOLAR' in solar['after']['suppression_reasons']


def test_nothing_is_saved_and_bad_input_is_rejected(service):
    before = json.dumps(service.customers['C10144'], sort_keys=True)
    service.what_if('C10144', {'statement_pref': 'digital'})
    assert json.dumps(service.customers['C10144'], sort_keys=True) == before
    with pytest.raises(ValueError, match='not editable'):
        service.what_if('C10144', {'spend_fuel_m': 999})
    with pytest.raises(ValueError, match='not one of'):
        service.what_if('C10144', {'vehicle_type': 'spaceship'})


def test_api_what_if(service, monkeypatch):
    from api.main import app
    monkeypatch.setattr('api.main.get_service', lambda: service)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://t') as c:
            ok = await c.post('/customers/C90001/what-if', json={'changes': {'vehicle_type': 'none'}})
            assert ok.status_code == 200 and ok.json()['customer_id'] == 'C90001'
            bad = await c.post('/customers/C90001/what-if', json={'changes': {'age': 12}})
            assert bad.status_code == 400
            assert (await c.get('/customers/C10144/optional')).status_code == 200
            assert (await c.get('/portfolio/optional')).status_code == 200
    asyncio.run(check())
