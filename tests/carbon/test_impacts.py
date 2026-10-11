"""P2 impact engine: supported examples match hand calculations; missing inputs stay unavailable. Owner: P2.

The expected numbers below are worked by hand from config/emission_factors.csv,
data/reference/pvgis/solar_yield_by_region.csv and config/solar_self_consumption.csv,
not by calling the engine's own helpers.
"""
import pytest

from src.carbon import engine
from src.recommend.engine import clean_impact

GRID_PLUS_TD = 0.13096 + 0.01299   # DESNZ 2026 kg CO2e/kWh, generation + T&D
HOUSE = {'has_solar': False, 'property_type': 'semi_detached', 'region': 'London',
         'est_electricity_kwh_year': 3200.0}
GAS_HOME = {'main_heating_fuel': 'gas', 'est_gas_kwh_year': 11500.0, 'green_home_use_case': None}


def test_solar_reference_design_matches_hand_calculation():
    out = engine.solar_impact(HOUSE)
    # 3.5 kWp x 1018.84 = 3,565.94 kWh -> MCS table 7-13 (use 3,000-3,499), row 3,300-3,599 -> 23%
    self_used = 3.5 * 1018.84 * 0.23
    assert out['status'] == 'ESTIMATED'
    assert out['energy_kwh_per_year'] == pytest.approx(self_used, abs=0.05)
    assert out['kg_co2e_per_year'] == pytest.approx(self_used * GRID_PLUS_TD, abs=1e-3)
    assert any('exported electricity gets no credit' in a for a in out['assumptions'])


def test_solar_share_rises_with_household_use():
    small = engine.solar_impact({**HOUSE, 'est_electricity_kwh_year': 1600.0})
    large = engine.solar_impact({**HOUSE, 'est_electricity_kwh_year': 5800.0})
    assert small['energy_kwh_per_year'] == pytest.approx(3.5 * 1018.84 * 0.15, abs=0.05)
    assert large['energy_kwh_per_year'] == pytest.approx(3.5 * 1018.84 * 0.33, abs=0.05)


@pytest.mark.parametrize('change, reason', [
    ({'has_solar': True}, 'EXISTING_OR_UNKNOWN_SOLAR_INSTALLATION'),
    ({'has_solar': None}, 'EXISTING_OR_UNKNOWN_SOLAR_INSTALLATION'),
    ({'property_type': 'flat'}, 'NO_DOCUMENTED_ROOF_FOR_REFERENCE_DESIGN'),
    ({'property_type': None}, 'NO_DOCUMENTED_ROOF_FOR_REFERENCE_DESIGN'),
    ({'region': None}, 'MISSING_REGIONAL_SOLAR_YIELD'),
    ({'est_electricity_kwh_year': None}, 'MISSING_ELECTRICITY_USE'),
    ({'est_electricity_kwh_year': float('nan')}, 'MISSING_ELECTRICITY_USE'),
    ({'est_electricity_kwh_year': 9000.0}, 'OUTSIDE_MCS_SELF_CONSUMPTION_RANGE'),
    ({'est_electricity_kwh_year': 900.0}, 'OUTSIDE_MCS_SELF_CONSUMPTION_RANGE'),
])
def test_solar_missing_or_unsupported_inputs_are_unavailable(change, reason):
    out = engine.solar_impact({**HOUSE, **change})
    assert out == {'status': 'INSUFFICIENT_DATA', 'factor_edition': None, 'reason': reason}


def test_heat_pump_switch_matches_hand_calculation():
    out = engine.green_home_impact(GAS_HOME)
    heat_pump_kwh = 11500 * 0.84 / 2.80                  # 3,450 kWh electricity added
    avoided = 11500 * 0.18231 - heat_pump_kwh * GRID_PLUS_TD
    assert out['status'] == 'ESTIMATED'
    assert out['kg_co2e_per_year'] == pytest.approx(avoided, abs=1e-3)
    assert out['energy_kwh_per_year'] == pytest.approx(11500 - heat_pump_kwh, abs=0.05)
    # Gas displaced and electricity added are both stated, not just the net figure.
    assert any('3,450 kWh' in a and '11,500 kWh' in a for a in out['assumptions'])


def test_heat_pump_dual_fuel_baseline_is_flagged():
    out = engine.green_home_impact({**GAS_HOME, 'energy_from_dual_fuel': True})
    assert any('dual-fuel' in a for a in out['assumptions'])


@pytest.mark.parametrize('change, reason', [
    ({'green_home_use_case': 'insulation'}, 'NO_DOCUMENTED_SAVING_FOR_THIS_RETROFIT'),
    ({'main_heating_fuel': 'oil'}, 'NO_DOCUMENTED_GAS_HEATING_BASELINE'),
    ({'main_heating_fuel': 'electric'}, 'NO_DOCUMENTED_GAS_HEATING_BASELINE'),
    ({'main_heating_fuel': None}, 'NO_DOCUMENTED_GAS_HEATING_BASELINE'),
    ({'est_gas_kwh_year': None}, 'MISSING_GAS_USE'),
    ({'est_gas_kwh_year': 0.0}, 'MISSING_GAS_USE'),
])
def test_heat_pump_missing_or_unsupported_inputs_are_unavailable(change, reason):
    out = engine.green_home_impact({**GAS_HOME, **change})
    assert out['status'] == 'INSUFFICIENT_DATA' and out['reason'] == reason


def test_paperless_lower_bound_matches_hand_calculation():
    out = engine.paperless_impact({'paper_statements_12m': 12})
    assert out['status'] == 'ESTIMATED'
    assert out['paper_sheets_per_year'] == 12
    assert out['kg_co2e_per_year'] == pytest.approx(12 * 1 * 0.005 * 1343.59114 / 1000, abs=1e-3)


@pytest.mark.parametrize('statements, reason', [
    (None, 'MISSING_PAPER_STATEMENT_COUNT'), (0, 'NO_PAPER_STATEMENTS_OBSERVED')])
def test_paperless_without_statements_is_unavailable_not_zero(statements, reason):
    assert engine.paperless_impact({'paper_statements_12m': statements})['reason'] == reason


def test_every_estimate_passes_p5_impact_validation():
    customer = {**HOUSE, **GAS_HOME, 'vehicle_type': 'petrol', 'est_annual_km': 9000.0,
                'num_vehicles': 1, 'paper_statements_12m': 12}
    impacts = engine.compute_impacts(customer)
    assert {code: i['status'] for code, i in impacts.items()} == dict.fromkeys(impacts, 'ESTIMATED')
    for impact in impacts.values():
        assert clean_impact(impact)['status'] == 'ESTIMATED'


def test_every_unavailable_reason_has_customer_text():
    import re
    from pathlib import Path
    codes = set(re.findall(r"unavailable\('([A-Z_]+)'\)", Path(engine.__file__).read_text(encoding='utf-8')))
    assert codes <= set(engine.UNAVAILABLE_REASONS), codes - set(engine.UNAVAILABLE_REASONS)


def test_implausibly_high_gas_use_is_not_used_for_a_heat_pump_estimate():
    # Above twice Ofgem's high benchmark (2 x 17,000 kWh) the bill-derived baseline is not trusted.
    assert engine.green_home_impact({**GAS_HOME, 'est_gas_kwh_year': 34000.0})['status'] == 'ESTIMATED'
    out = engine.green_home_impact({**GAS_HOME, 'est_gas_kwh_year': 71064.0})
    assert out['reason'] == 'GAS_USE_OUTSIDE_PLAUSIBLE_RANGE'


def test_p5_keeps_the_reason_an_impact_is_unavailable():
    cleaned = clean_impact(engine.solar_impact({**HOUSE, 'property_type': 'flat'}))
    assert cleaned['status'] == 'INSUFFICIENT_DATA'
    assert cleaned['reason'] == 'NO_DOCUMENTED_ROOF_FOR_REFERENCE_DESIGN'
    assert 'reason' not in clean_impact(None)
