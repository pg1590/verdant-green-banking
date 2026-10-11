"""Optional opportunities (brief slide 4): Green Investments (GP05) and Green Loans (GP06).

Rules decide eligibility from the customer's own data; every signal carries its evidence.
No propensity is produced (there are no take-up labels to train on), and impact stays
INSUFFICIENT_DATA with a reason because no sourced method exists. Rules live in
config/optional_products.yaml; this module never reads model scores or changes GP01-GP04.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import yaml

HOUSES = {'detached', 'semi_detached', 'terraced', 'bungalow'}
OWNERS = {'owner_mortgage', 'owner_outright'}

# Customer-facing wording for each signal (used in explanations and the UI).
SIGNAL_TEXT = {
    'EXISTING_INVESTMENTS': 'you already hold investments with us',
    'EXISTING_SAVINGS': 'you already hold savings with us',
    'DRIVES_AN_EV': 'you drive an electric car',
    'HAS_SOLAR_PANELS': 'you have solar panels',
    'LOW_CARBON_TRAVEL': 'you spend well above average on public transport',
    'EFFICIENT_HOME': 'your home already has a high energy rating',
    'EXISTING_CREDIT_CARD': 'you have a credit card with us',
    'EXISTING_VEHICLE_LOAN': 'you have a vehicle loan with us',
    'EXISTING_MORTGAGE': 'you have a mortgage with us',
    'HOME_EV_CHARGER_USE_CASE': 'you charge your electric car away from home, and you own a house',
    'EFFICIENT_APPLIANCES_USE_CASE': 'your electricity use is above the typical high level',
}


@lru_cache(maxsize=1)
def load_optional() -> dict:
    with (Path(__file__).resolve().parents[2] / 'config' / 'optional_products.yaml').open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def _signals(c: Mapping, t: Mapping) -> dict[str, dict]:
    """Every signal that holds for this customer, with the evidence behind it."""
    out = {}
    def add(code, ok, **evidence):
        if ok:
            out[code] = evidence
    add('EXISTING_INVESTMENTS', c.get('has_investments') is True, has_investments=True)
    add('EXISTING_SAVINGS', c.get('has_savings') is True, has_savings=True)
    add('DRIVES_AN_EV', c.get('vehicle_type') == 'ev', vehicle_type='ev')
    add('HAS_SOLAR_PANELS', c.get('has_solar') is True, has_solar=True)
    travel = c.get('spend_public_transport_m')
    add('LOW_CARBON_TRAVEL', isinstance(travel, (int, float)) and travel >= t['low_carbon_travel_gbp_m'],
        spend_public_transport_m=travel)
    add('EFFICIENT_HOME', c.get('epc_rating') in {'A', 'B'}, epc_rating=c.get('epc_rating'))
    add('EXISTING_CREDIT_CARD', c.get('has_credit_card') is True, has_credit_card=True)
    add('EXISTING_VEHICLE_LOAN', c.get('has_vehicle_loan') is True, has_vehicle_loan=True)
    add('EXISTING_MORTGAGE', c.get('has_mortgage') is True, has_mortgage=True)
    charging = c.get('spend_fuel_m')
    add('HOME_EV_CHARGER_USE_CASE', c.get('vehicle_type') == 'ev' and c.get('num_vehicles') == 1
        and isinstance(charging, (int, float)) and charging >= t['public_charging_gbp_m']
        and c.get('home_status') in OWNERS and c.get('property_type') in HOUSES,
        avg_monthly_charging_spend_gbp=charging, home_status=c.get('home_status'))
    kwh = c.get('est_electricity_kwh_year')
    add('EFFICIENT_APPLIANCES_USE_CASE', isinstance(kwh, (int, float)) and kwh > t['high_electricity_kwh_year'],
        est_electricity_kwh_year=kwh)
    return out


def evaluate(customer: Mapping) -> list[dict]:
    """One row per optional product for this customer (eligible or with suppression reasons)."""
    cfg = load_optional()
    signals = _signals(customer, cfg['thresholds'])
    rows = []
    for code, p in cfg['products'].items():
        reasons = []
        age = customer.get('age')
        if not isinstance(age, int) or age < 18:
            reasons.append('UNDER_18')
        if customer.get('marketing_consent') is not True:
            reasons.append('NO_MARKETING_CONSENT')
        held = [s for s in p['requires_one_of'] if s in signals]
        if not held:
            reasons.append(p['missing_requirement_reason'])
        support = [s for s in p['requires_signal'] if s in signals]
        if not support:
            reasons.append(p['missing_signal_reason'])
        use_cases = [u for u, s in zip(p.get('intended_use_cases', []), p['requires_signal']) if s in signals]
        if 'impact_reason_by_use_case' in p:
            impact_reason = p['impact_reason_by_use_case'][use_cases[0]] if use_cases else p['missing_signal_reason']
        else:
            impact_reason = p['impact_reason']
        codes = held + support
        shown = not reasons
        rows.append({
            'customer_id': customer.get('customer_id'), 'product_code': code, 'product_name': p['name'],
            'opportunity': p['opportunity'], 'kind': p['kind'], 'optional': True,
            'scope': cfg['scope'], 'propensity': None, 'signal_count': len(codes),
            'reason_codes': codes if shown else [], 'use_cases': use_cases,
            'evidence': {k: v for s in codes for k, v in signals[s].items()} if shown else {},
            'explanation': ('We noticed that ' + '; '.join(SIGNAL_TEXT[s] for s in codes) + '.') if shown else None,
            'impact': {'status': 'INSUFFICIENT_DATA', 'reason': impact_reason},
            'eligible_to_show': shown, 'suppression_reasons': reasons,
            'action': p['action'] if shown else None,
            'data_timestamp': customer.get('feature_cutoff_date'),
            'policy_version': cfg['policy_version'],
            'eligibility_scope': 'DISPLAY_ONLY_NOT_ADVICE_OR_CREDIT_APPROVAL',
        })
    return rows
