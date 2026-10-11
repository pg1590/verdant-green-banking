import pytest
from src.recommend.demo import demo_bundle
from src.recommend.eligibility import check_eligibility, eligible_customer_product_pairs


@pytest.fixture
def customer():
    return demo_bundle()['customers'][0].copy()


@pytest.mark.parametrize('code,changes,reason', [
    ('GP01', {'vehicle_type': 'ev'}, 'ALREADY_HAS_EV'),
    ('GP01', {'vehicle_type': 'none'}, 'NO_EV_PURCHASE_USE_CASE'),
    ('GP01', {'vehicle_type': None}, 'MISSING_VEHICLE_TYPE'),
    ('GP01', {'conflict_flags': ['VEHICLE_SOLD']}, 'VEHICLE_SOLD'),
    ('GP02', {'home_status': 'renter'}, 'NOT_HOMEOWNER'),
    ('GP02', {'epc_rating': 'A'}, 'NO_IDENTIFIED_HOME_UPGRADE'),
    ('GP02', {'epc_rating': None}, 'MISSING_HOME_UPGRADE_EVIDENCE'),
    ('GP02', {'green_home_use_case': 'solar_installation'}, 'UNSUPPORTED_GREEN_HOME_USE_CASE'),
    ('GP03', {'home_status': 'renter'}, 'NOT_HOMEOWNER'),
    ('GP03', {'has_solar': True}, 'ALREADY_HAS_SOLAR'),
    ('GP03', {'has_solar': 'false'}, 'ALREADY_HAS_SOLAR'),
    ('GP03', {'has_solar': None}, 'MISSING_HAS_SOLAR'),
    ('GP03', {'property_type': 'flat'}, 'PROPERTY_REQUIRES_SUITABILITY_REVIEW'),
    ('GP04', {'statement_pref': 'digital'}, 'ALREADY_DIGITAL'),
    ('GP04', {'statement_pref': None}, 'MISSING_STATEMENT_PREF'),
    ('GP04', {'requires_paper_statements': True}, 'PAPER_STATEMENTS_REQUIRED'),
    ('GP03', {'marketing_consent': False}, 'NO_MARKETING_CONSENT'),
    ('GP03', {'age': 17}, 'UNDER_18'),
    ('GP03', {'age': True}, 'UNDER_18'),
    ('GP03', {'age': float('nan')}, 'MISSING_AGE'),
    ('GP03', {'conflict_flags': 'HOME_STATUS_CONFLICT'}, 'INVALID_CONFLICT_FLAGS'),
])
def test_suppression(customer, code, changes, reason):
    customer.update(changes)
    result = check_eligibility(customer, code)
    assert not result.eligible_to_show
    assert reason in result.suppression_reasons


def test_ev_replacement_and_charging_not_conflicting(customer):
    customer.update(vehicle_type='ev', ev_replacement_use_case=True, spend_fuel_recent_m=80)
    assert check_eligibility(customer, 'GP01').eligible_to_show
    customer['conflict_flags'] = ['VEHICLE_FUEL_CONFLICT']
    assert not check_eligibility(customer, 'GP01').eligible_to_show
    assert check_eligibility(customer, 'GP03').eligible_to_show


def test_training_ignores_consent_but_not_home_rules(customer):
    customer['marketing_consent'] = False
    assert check_eligibility(customer, 'GP03', purpose='training').eligible_to_show
    customer['home_status'] = 'renter'
    assert not check_eligibility(customer, 'GP03', purpose='training').eligible_to_show
    assert check_eligibility(customer, 'GP04').eligible_to_show


def test_explicit_home_use_case_and_first_ev(customer):
    customer.update(epc_rating=None, green_home_use_case='heat_pump', vehicle_type='none', ev_purchase_use_case=True)
    assert check_eligibility(customer, 'GP02').eligible_to_show
    assert check_eligibility(customer, 'GP01').eligible_to_show


def test_pairs_never_depend_on_scores_or_labels(customer):
    before = eligible_customer_product_pairs([customer])
    customer.update(propensity=1, took_up=True, kg_co2e_per_year=999)
    assert before == eligible_customer_product_pairs([customer])
    assert len(before) == 4
    with pytest.raises(ValueError):
        check_eligibility(customer, 'GP99')
