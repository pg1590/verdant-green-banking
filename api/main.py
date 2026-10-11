"""FastAPI boundary over the shared, validated recommendation service."""
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from src.recommend.service import RecommendationService
from src.recommend.runtime import load_runtime_service
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title='Verdant · Explainable Green Banking', version='0.1.0',
              description='Explainable product opportunities and portfolio analytics using synthetic data. Display relevance is not credit approval.')


@lru_cache(maxsize=1)
def get_service() -> RecommendationService:
    service = load_runtime_service()
    for view in (service.summary, service.opportunities, service.segments):
        view()  # warm the portfolio cache once at startup
    return service


def customer_result(customer_id: str, method: str, **kwargs):
    try:
        return getattr(get_service(), method)(customer_id, **kwargs)
    except KeyError:
        raise HTTPException(status_code=404, detail='Customer not found') from None
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.get('/health')
def health():
    service = get_service()
    return {'status': 'ok', 'data_mode': service.mode, 'policy_version': service.catalogue['policy_version'],
            'bundle_scope': service.bundle.get('bundle_scope', 'FULL_OR_EXPLICIT_BUNDLE')}


@app.get('/customers')
def customers():
    service = get_service()
    return {'data_mode': service.mode, 'customers': list(service.customers.values())}


@app.get('/customers/{customer_id}/recommendations')
def recommendations(customer_id: str, include_suppressed: bool = Query(False)):
    return customer_result(customer_id, 'recommendations', include_suppressed=include_suppressed)


@app.get('/customers/{customer_id}/insight')
def insight(customer_id: str):
    return customer_result(customer_id, 'insight')


class CustomerEvent(BaseModel):
    product_code: Literal['GP01', 'GP02', 'GP03', 'GP04']
    event: Literal['shown', 'clicked', 'dismissed', 'not_relevant', 'took_up']


@app.post('/customers/{customer_id}/events')
def customer_event(customer_id: str, event: CustomerEvent):
    return customer_result(customer_id, 'record_event', product_code=event.product_code, event=event.event)


@app.delete('/customers/{customer_id}/memory')
def forget_memory(customer_id: str):
    return customer_result(customer_id, 'forget_memory')


@app.get('/portfolio/summary')
def portfolio_summary():
    return get_service().summary()


@app.get('/portfolio/opportunities')
def portfolio_opportunities():
    return get_service().opportunities()


class WhatIf(BaseModel):
    changes: dict


@app.post('/customers/{customer_id}/what-if')
def what_if(customer_id: str, body: WhatIf):
    return customer_result(customer_id, 'what_if', changes=body.changes)


@app.get('/customers/{customer_id}/optional')
def optional_opportunities(customer_id: str):
    return customer_result(customer_id, 'optional_opportunities')


class Question(BaseModel):
    question: str


@app.post('/portfolio/ask')
def portfolio_ask(body: Question):
    try:
        return get_service().ask(body.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.get('/portfolio/engagement')
def portfolio_engagement():
    return get_service().engagement()


@app.get('/portfolio/optional')
def portfolio_optional():
    return get_service().optional_summary()


@app.get('/segments')
def segments():
    return get_service().segments()


@app.get('/segments/insight')
def segment_insight(region: str, product_code: str):
    try:
        return get_service().business_insight(region, product_code)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
