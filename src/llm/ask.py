"""Ask the portfolio: plain-English questions from the business team, answered from computed facts.

The LLM never calculates. Step 1 picks ONE query from a fixed menu (router). Step 2 runs that query
in P5's service, which computes every number. Step 3 phrases an answer from those facts only, and
the checker rejects any number that is not in the facts, plus approval or guarantee language.
With no provider, a keyword router and a template answer give the same facts. Owner: P4.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone

from src.schemas import tables as T

from .client import LLMUnavailable, llm_settings
from .validator import BANNED_PATTERNS, allowed_numbers, extract_numbers

ASK_VERSION = 'ask-v1'
INTENTS = ('portfolio_overview', 'product', 'region', 'top_segments', 'impact', 'optional_products', 'out_of_scope')

ROUTER_SYSTEM = """\
You route a bank relationship team's question about a green-products portfolio to ONE predefined query.
You never answer the question and never compute anything. Pick the closest intent:
- portfolio_overview: how many customers, opportunities, overall picture
- product: one product's opportunities (needs product_code)
- region: one region's opportunities across products (needs region)
- top_segments: best region + product combinations, where to focus
- impact: potential CO2e, energy or paper savings
- optional_products: Green Investments or Green Loans
- out_of_scope: anything else (individual customers' personal data, credit decisions, predictions about the future)
Products: GP01 EV Finance, GP02 Green Home Loan, GP03 Solar Finance, GP04 Paperless Banking.
Regions: """ + ', '.join(T.REGIONS) + '. Use "none" when no product or region applies.'

ROUTER_SCHEMA = {'type': 'object', 'properties': {
    'intent': {'type': 'string', 'enum': list(INTENTS)},
    'product_code': {'type': 'string', 'enum': [*T.PRODUCT_CODES, 'none']},
    'region': {'type': 'string', 'enum': [*T.REGIONS, 'none']}},
    'required': ['intent', 'product_code', 'region'], 'additionalProperties': False}

ANSWER_SYSTEM = """\
You answer a bank relationship team's question using ONLY the FACTS given. Write 2-4 short, plain sentences.
Use numbers exactly as written in the FACTS; do not add, round, combine or calculate numbers.
Opportunities and expected take-ups are model estimates, not sales; impact figures are potential, not achieved.
Never say anyone is approved, eligible or guaranteed to save. If the FACTS do not answer the question, say so."""

ANSWER_SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}},
                 'required': ['answer'], 'additionalProperties': False}

KEYWORDS = {'GP01': ('ev', 'electric car', 'electric vehicle', 'car'), 'GP02': ('green home', 'home loan', 'heat pump', 'retrofit', 'insulation'),
            'GP03': ('solar', 'panel'), 'GP04': ('paperless', 'paper', 'statement', 'digital')}


def keyword_route(question: str) -> dict:
    """Offline router: deterministic keyword matching over the same menu."""
    q = question.lower()
    region = next((r for r in T.REGIONS if r.lower() in q), None)
    product = next((code for code, words in KEYWORDS.items() if any(re.search(r'\b' + w, q) for w in words)), None)
    if any(w in q for w in ('investment', 'invest', 'green loan', 'optional')):
        intent = 'optional_products'
    elif any(w in q for w in ('co2', 'carbon', 'emission', 'impact', 'energy', 'kwh', 'saving')):
        intent = 'impact'
    elif region:
        intent = 'region'
    elif product:
        intent = 'product'
    elif any(w in q for w in ('segment', 'focus', 'best', 'top', 'where', 'priorit')):
        intent = 'top_segments'
    elif any(w in q for w in ('how many', 'overview', 'total', 'portfolio', 'summary', 'customers')):
        intent = 'portfolio_overview'
    else:
        intent = 'out_of_scope'
    return {'intent': intent, 'product_code': product, 'region': region}


def _pct(x):
    return f'{x:.1%}' if isinstance(x, (int, float)) else 'not available'


def _num(x, digits=0):
    return f'{x:,.{digits}f}' if isinstance(x, (int, float)) else 'not available'


def facts_for(service, route: Mapping) -> dict:
    """Run the chosen query in the service. Every number the answer may use is formatted here."""
    intent, code, region = route['intent'], route.get('product_code'), route.get('region')
    names = {c: p['name'] for c, p in service.catalogue['products'].items()}
    if intent == 'product' and code in names:
        p = next(p for p in service.opportunities()['products'] if p['product_code'] == code)
        return {'query': f'{names[code]} opportunities', 'facts': [
            f'{_num(p["displayed_opportunities"])} customers are currently shown {names[code]}, out of {_num(p["structurally_eligible"])} the product rules allow.',
            f'{_num(p["high_priority"])} of them are high priority (at least 1.5 times the average likelihood for this product).',
            f'The models expect about {_num(p["expected_takeups"], 1)} take-ups if every shown customer were contacted (an estimate, not sales).',
            f'Median likelihood of take-up is {_pct(p["propensity_median"])}; {_num(p["impact_estimated"])} shown customers have an impact estimate.']}
    if intent in ('region', 'top_segments'):
        rows = [r for r in service.segments()['segments']
                if (region is None or r['region'] == region) and (code is None or r['product_code'] == code)]
        scope = ' / '.join(x for x in (region, names.get(code)) if x) or 'all regions and products'
        if intent == 'region':
            rows = sorted(rows, key=lambda r: -r['customers'])
        return {'query': f'Segments ({scope}), ranked by likelihood x confidence', 'facts': [
            f'{r["region"]} / {names[r["product_code"]]}: {_num(r["customers"])} customers shown, '
            f'mean likelihood {_pct(r["mean_propensity"])}.' for r in rows[:5]]
            or [f'No opportunities are currently shown for {scope}.']}
    if intent == 'impact':
        m = service.summary()['potential_impact']
        return {'query': 'Potential annual impact (one main finance product per customer; paperless separate)', 'facts': [
            f'Potential emissions avoided: {_num((m["kg_co2e_per_year"]["value"] or 0) / 1000)} tonnes CO2e a year, '
            f'from {_num(m["kg_co2e_per_year"]["coverage_customers"])} customers with an estimate.',
            f'Less purchased energy: {_num((m["energy_kwh_per_year"]["value"] or 0) / 1e6, 1)} GWh a year '
            f'({_num(m["energy_kwh_per_year"]["coverage_customers"])} customers).',
            f'Paper saved: {_num(m["paper_sheets_per_year"]["value"])} sheets a year '
            f'({_num(m["paper_sheets_per_year"]["coverage_customers"])} customers).',
            'These are potential savings if customers acted, not achieved reductions.']}
    if intent == 'optional_products':
        o = service.optional_summary()
        return {'query': 'Optional products (rule-based signals, no propensity model)', 'facts': [
            f'{p["product_name"]}: {_num(p["eligible_customers"])} customers have matching signals; most common: '
            + ', '.join(k.replace('_', ' ').lower() for k in list(p['by_signal'])[:2]) + '.'
            for p in o['products']] + ['These are rule-based signals from customer data, not model predictions.']}
    if intent == 'portfolio_overview':
        s = service.summary()
        return {'query': 'Portfolio overview', 'facts': [
            f'{_num(s["customers_analysed"])} customers analysed; {_num(s["customers_with_opportunities"])} have at least one suggestion.',
            f'{_num(s["eligible_opportunities"])} opportunities are shown in total: ' + ', '.join(
                f'{names[c]} {_num(n)}' for c, n in s['opportunities_by_product'].items()) + '.']}
    return {'query': 'Out of scope', 'facts': [
        'This assistant answers questions about portfolio opportunities, segments, regions and potential impact. '
        'It cannot discuss individual customers, credit decisions or predictions beyond the computed figures.']}


def check_answer(answer: str, facts: Mapping) -> list[str]:
    problems = []
    unknown = extract_numbers(answer) - allowed_numbers(facts)
    if unknown:
        problems.append('numbers not in the facts: ' + ', '.join(f'{n:g}' for n in sorted(unknown)))
    for name, pattern in BANNED_PATTERNS[:2]:  # approval/eligibility and guarantee language
        if re.search(pattern, answer, re.I):
            problems.append(f'banned language: {name}')
    if not answer.strip():
        problems.append('empty answer')
    return problems


def ask(service, question: str, provider=None) -> dict:
    question = (question or '').strip()[:500]
    if not question:
        raise ValueError('Ask a question about the portfolio')
    if provider is None:
        from .explain import _default_provider
        provider = _default_provider()
    route, router_source, error = None, 'KEYWORDS', None
    try:
        candidate = provider.complete_json(ROUTER_SYSTEM, f'QUESTION: {question}', ROUTER_SCHEMA)
        if candidate.get('intent') in INTENTS:
            route = {'intent': candidate['intent'],
                     'product_code': candidate.get('product_code') if candidate.get('product_code') in T.PRODUCT_CODES else None,
                     'region': candidate.get('region') if candidate.get('region') in T.REGIONS else None}
            router_source = 'LLM'
    except LLMUnavailable as e:
        error = str(e)
    route = route or keyword_route(question)
    facts = facts_for(service, route)
    answer, source, problems = None, 'TEMPLATE', []
    if router_source == 'LLM' and route['intent'] != 'out_of_scope':
        message = f'QUESTION: {question}\nFACTS ({facts["query"]}):\n' + '\n'.join(f'- {f}' for f in facts['facts'])
        for _ in range(1 + int(llm_settings().get('max_retries', 1))):
            try:
                text = provider.complete_json(ANSWER_SYSTEM, message, ANSWER_SCHEMA).get('answer', '')
            except LLMUnavailable as e:
                error = str(e)
                break
            problems = check_answer(text, facts)
            if not problems:
                answer, source = text, 'LLM'
                break
            message += '\nYour previous answer was rejected: ' + '; '.join(problems) + '. Use only the FACTS.'
    answer = answer or ' '.join(facts['facts'])
    return {'question': question, 'answer': answer, 'route': route, 'facts': facts,
            'provenance': {'router': router_source, 'text_source': source, 'rejected': problems if source == 'TEMPLATE' else [],
                           'model': getattr(provider, 'model', None), 'version': ASK_VERSION, 'llm_error': error,
                           'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}}
