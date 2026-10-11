"""Optional products (slide 4): rule-based signals, no propensity, no invented impact."""
import json
from pathlib import Path

from src.carbon.engine import UNAVAILABLE_REASONS
from src.recommend.optional import evaluate
from src.recommend.service import RecommendationService

BASE = {'customer_id': 'X1', 'age': 40, 'marketing_consent': True, 'feature_cutoff_date': '2026-06-30',
        'has_investments': False, 'has_savings': False, 'has_credit_card': False, 'has_vehicle_loan': False,
        'has_mortgage': False, 'vehicle_type': 'petrol', 'num_vehicles': 1, 'has_solar': False,
        'spend_public_transport_m': 10.0, 'epc_rating': 'D', 'spend_fuel_m': 80.0,
        'home_status': 'renter', 'property_type': 'flat', 'est_electricity_kwh_year': 2500.0}


def rows(**change):
    return {r['product_code']: r for r in evaluate({**BASE, **change})}


def test_green_investments_needs_holdings_and_a_preference_signal():
    assert rows()['GP05']['suppression_reasons'] == ['NO_EXISTING_INVESTMENTS_OR_SAVINGS', 'NO_SUSTAINABILITY_PREFERENCE_SIGNAL']
    assert rows(has_savings=True)['GP05']['suppression_reasons'] == ['NO_SUSTAINABILITY_PREFERENCE_SIGNAL']
    shown = rows(has_investments=True, spend_public_transport_m=60.0)['GP05']
    assert shown['eligible_to_show'] and shown['reason_codes'] == ['EXISTING_INVESTMENTS', 'LOW_CARBON_TRAVEL']
    assert shown['evidence']['spend_public_transport_m'] == 60.0
    assert shown['propensity'] is None and shown['scope'] == 'RULE_BASED_SIGNAL_NO_PROPENSITY_MODEL'
    assert shown['impact'] == {'status': 'INSUFFICIENT_DATA', 'reason': 'NO_DOCUMENTED_INVESTMENT_IMPACT_METHOD'}


def test_green_loans_need_borrowing_and_an_eligible_use_case():
    assert rows(has_credit_card=True)['GP06']['suppression_reasons'] == ['NO_ELIGIBLE_GREEN_LOAN_USE_CASE']
    appliances = rows(has_credit_card=True, est_electricity_kwh_year=5000.0)['GP06']
    assert appliances['eligible_to_show'] and appliances['use_cases'] == ['efficient_appliances']
    charger = rows(has_mortgage=True, vehicle_type='ev', spend_fuel_m=45.0, home_status='owner_mortgage',
                   property_type='detached')['GP06']
    assert charger['use_cases'] == ['home_ev_charger']
    assert charger['impact']['reason'] == 'NO_EMISSIONS_CHANGE_FROM_HOME_CHARGING'
    # A flat or a renter cannot install a home charger.
    assert not rows(has_mortgage=True, vehicle_type='ev', spend_fuel_m=45.0)['GP06']['eligible_to_show']


def test_consent_and_age_still_apply():
    r = rows(has_investments=True, has_solar=True, marketing_consent=False, age=17)['GP05']
    assert not r['eligible_to_show'] and {'UNDER_18', 'NO_MARKETING_CONSENT'} <= set(r['suppression_reasons'])
    assert r['reason_codes'] == [] and r['action'] is None


def test_every_optional_impact_reason_has_customer_text():
    for r in evaluate({**BASE, 'has_credit_card': True, 'est_electricity_kwh_year': 5000.0, 'has_investments': True,
                       'has_solar': True}):
        assert r['impact']['reason'] in UNAVAILABLE_REASONS


def test_service_and_preview_keep_optional_products_out_of_the_ranked_list():
    preview = json.loads((Path(__file__).resolve().parents[2] / 'data/preview/team_bundle.json').read_text(encoding='utf-8'))
    service = RecommendationService(preview)
    summary = service.optional_summary()
    assert {p['product_code'] for p in summary['products']} == {'GP05', 'GP06'}
    assert sum(p['eligible_customers'] for p in summary['products']) > 0
    assert set(service.summary()['opportunities_by_product']) == {'GP01', 'GP02', 'GP03', 'GP04'}
