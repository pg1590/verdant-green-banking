import asyncio
import pytest
pytest.importorskip('fastapi', reason='Install requirements.txt to run API integration checks')
httpx = pytest.importorskip('httpx')
from api.main import app
from src.recommend.demo import demo_bundle
from src.recommend.service import RecommendationService


def test_api_contract(monkeypatch):
    monkeypatch.setattr('api.main.get_service', RecommendationService.demo)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.get('/health')).json()['data_mode'] == 'DEMO_MOCK'
            for endpoint in ['/customers', '/customers/C10001/recommendations', '/customers/C10001/insight', '/portfolio/summary', '/segments']:
                assert (await client.get(endpoint)).status_code == 200
            assert (await client.get('/customers/unknown/recommendations')).status_code == 404
            result = (await client.get('/customers/C10002/recommendations')).json()
            assert all(r['eligible_to_show'] for r in result['recommendations'])
            audit = (await client.get('/customers/C10002/recommendations?include_suppressed=true')).json()
            assert len(audit['recommendations']) == 4
            schema = (await client.get('/openapi.json')).json()
            assert '/portfolio/summary' in schema['paths']
    asyncio.run(check())


def test_api_preserves_structured_p4_explanation(monkeypatch):
    backend = RecommendationService(demo_bundle(), insight_provider=lambda row:
        {'headline': 'P4 API headline', 'text': 'P4 API explanation'})
    monkeypatch.setattr('api.main.get_service', lambda: backend)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get('/customers/C10002/insight')
            assert response.status_code == 200
            for item in response.json()['insights']:
                assert item['headline'] == 'P4 API headline'
                assert item['text'] == 'P4 API explanation'
                assert item['source'] == 'P4_VALIDATED_PROVIDER'
                assert item['fallback_reason'] is None
    asyncio.run(check())


def test_api_opportunity_table_reconciles_with_summary(monkeypatch):
    monkeypatch.setattr('api.main.get_service', RecommendationService.demo)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            table = (await client.get('/portfolio/opportunities')).json()
            summary = (await client.get('/portfolio/summary')).json()
            shown = {p['product_code']: p['displayed_opportunities'] for p in table['products']}
            assert shown == summary['opportunities_by_product']
            assert set(table['definitions']) >= {'high_priority', 'expected_takeups'}
    asyncio.run(check())
