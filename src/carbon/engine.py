"""FR4 deterministic carbon engine. Owner: P2. Pure Python, no ML, no LLM.

Framing: GHG Protocol project / avoided-emissions accounting. If an input is
missing, return {"status": "INSUFFICIENT_DATA"}: never guess.

Output contract = P5's impact input (docs/integration.md), one dict per product:
  status            "ESTIMATED" | "INSUFFICIENT_DATA"
  kg_co2e_per_year  float >= 0
  energy_kwh_per_year, paper_sheets_per_year   optional floats >= 0
  method            non-empty str, e.g. "GHG Protocol avoided emissions; BEV vs current petrol car"
  assumptions       non-empty list[str] (baseline goes here too, plus price periods)
  horizon_years     > 0 (settings.carbon.horizon_years)
  factor_edition    REQUIRED on ESTIMATED: exactly settings.carbon.factor_edition
                    ("DESNZ GHG Conversion Factors 2026"); null on INSUFFICIENT_DATA (D7)
P5 rejects an ESTIMATED impact that lacks method, assumptions, a valid number or
factor_edition, and raises on mixed or unconfigured editions (docs/integration.md).
Physical inputs (est_annual_km, est_*_kwh_year) come from src.carbon.conversions.

Formulas (annual; factors from src.carbon.factors.get):
  EV         : km/yr x (ICE factor[vehicle_type] - (bev_car_avg_per_km + bev_car_avg_td_per_km))
  Solar      : reference 3.5 kWp design x PVGIS yield[region] x MCS MGD 003 self-consumed
               share (by generation and the customer's electricity use) x (grid + T&D factor).
               Exported electricity gets no credit.
  Green Home : one documented intervention, gas boiler -> air-source heat pump.
               heat = gas kWh x boiler in-situ efficiency; heat pump kWh = heat / SPF;
               avoided = gas kWh x gas factor - heat pump kWh x (grid + T&D factor)
  Paperless  : statements/yr x sheets/statement x sheet mass x paper factor / 1000
               (postage excluded unless a documented factor is sourced: D16)
Solar and Green Home are illustrative designs applied to the customer's own bills.
They are not site surveys, and they are not added together for one customer.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from math import isfinite
import csv
import json

from src.carbon import factors
from src.common import config
from src.common.records import to_records

INSUFFICIENT = {"status": "INSUFFICIENT_DATA"}

# Plain-English text for each INSUFFICIENT_DATA reason, shown to customers and staff.
UNAVAILABLE_REASONS = {
    'NO_DOCUMENTED_ICE_REPLACEMENT_BASELINE': 'There is no petrol, diesel or hybrid car on record to compare an electric car against.',
    'MISSING_ROAD_FUEL_MILEAGE': 'We could not estimate your annual mileage from fuel spending.',
    'SINGLE_REPLACED_VEHICLE_MILEAGE_NOT_ESTABLISHED': 'Your household has more than one car, so we cannot tell how far the replaced car travels.',
    'MISSING_EMISSIONS_FACTOR': 'A required emissions factor is missing.',
    'EXISTING_OR_UNKNOWN_SOLAR_INSTALLATION': 'You may already have solar panels, so a new-system estimate would not be reliable.',
    'NO_DOCUMENTED_ROOF_FOR_REFERENCE_DESIGN': 'We do not know if your home has a suitable roof (for example, it is a flat).',
    'MISSING_REGIONAL_SOLAR_YIELD': 'We do not have solar yield data for your region.',
    'MISSING_ELECTRICITY_USE': 'We could not estimate your annual electricity use from your bills.',
    'OUTSIDE_MCS_SELF_CONSUMPTION_RANGE': 'Your electricity use is outside the range the MCS self-consumption tables cover (1,500 to 5,999 kWh a year).',
    'NO_DOCUMENTED_SAVING_FOR_THIS_RETROFIT': 'We only have a documented saving for a heat pump, not for this type of home upgrade.',
    'NO_DOCUMENTED_GAS_HEATING_BASELINE': 'Your home is not heated by gas, so the heat pump estimate does not apply.',
    'MISSING_GAS_USE': 'We could not estimate your annual gas use from your bills.',
    'GAS_USE_OUTSIDE_PLAUSIBLE_RANGE': 'Your estimated gas use is unusually high, so we have not used it for an estimate.',
    'MISSING_PAPER_STATEMENT_COUNT': 'We do not know how many paper statements you receive.',
    'NO_PAPER_STATEMENTS_OBSERVED': 'No paper statements were sent to you in the last 12 months.',
    'MISSING_DOCUMENTED_PAPER_ASSUMPTION': 'A required paper assumption is missing.',
    # Optional products (src/recommend/optional.py): no sourced method, so no number is shown.
    'NO_DOCUMENTED_INVESTMENT_IMPACT_METHOD': 'Measuring the emissions linked to investments (PCAF) needs fund holdings data we do not have.',
    'NO_EMISSIONS_CHANGE_FROM_HOME_CHARGING': 'A home charger uses the same grid electricity as public charging, so the benefit is cost and convenience, not lower emissions.',
    'NO_DOCUMENTED_APPLIANCE_SAVING': 'Savings depend on which appliances are replaced, and we have no documented figure to use.',
    'NO_ELIGIBLE_GREEN_LOAN_USE_CASE': 'No eligible green loan use case was found.',
}
# A spend-derived gas estimate above twice Ofgem's high consumption benchmark is treated as a
# data problem (business use, wrong tariff, misread bills), not a household baseline.
GAS_PLAUSIBILITY_MULTIPLE = 2


def unavailable(reason: str) -> dict:
    return {'status': 'INSUFFICIENT_DATA', 'factor_edition': None, 'reason': reason}


def _number(value) -> bool:
    return type(value) in {int, float} and isfinite(value) and value >= 0


def _estimate(method: str, assumptions: list[str], **metrics) -> dict:
    return {'status': 'ESTIMATED', 'method': method, 'assumptions': assumptions,
            'factor_edition': config.settings()['carbon']['factor_edition'],
            'horizon_years': config.settings()['carbon']['horizon_years'], **metrics}


def ev_impact(features: Mapping) -> dict:
    vehicle, km = features.get('vehicle_type'), features.get('est_annual_km')
    if vehicle not in {'petrol', 'diesel', 'hybrid'}:
        return unavailable('NO_DOCUMENTED_ICE_REPLACEMENT_BASELINE')
    if not _number(km) or km == 0:
        return unavailable('MISSING_ROAD_FUEL_MILEAGE')
    if features.get('num_vehicles') != 1:
        return unavailable('SINGLE_REPLACED_VEHICLE_MILEAGE_NOT_ESTABLISHED')
    try:
        baseline = factors.get(f'{vehicle}_car_avg_per_km')
        bev = factors.get('bev_car_avg_per_km')
        losses = factors.get('bev_car_avg_td_per_km')
    except factors.MissingFactorError:
        return unavailable('MISSING_EMISSIONS_FACTOR')
    return _estimate('GHG Protocol avoided emissions: current ICE vehicle versus average BEV',
        [f'Baseline: {vehicle} car, {km:g} estimated km/year; the replacement BEV travels the same distance.',
         'Mileage derives from road-fuel spend, average pump prices and derived fuel economy; charging is excluded.',
         f'Factors: {baseline.factor_id}, {bev.factor_id}, {losses.factor_id}; {baseline.source} {baseline.edition}.',
         'Use-phase estimate only; excludes manufacturing, financing, rebound and fuel upstream emissions.'],
        kg_co2e_per_year=round(max(0.0, km * (baseline.value - bev.value - losses.value)), 3))


HOUSE_TYPES = {'detached', 'semi_detached', 'terraced', 'bungalow'}


@lru_cache(maxsize=1)
def regional_solar_yield() -> dict[str, float]:
    """kWh per kWp per year by region (PVGIS v5.3, one city per region)."""
    path = config.REFERENCE_DIR / 'pvgis' / 'solar_yield_by_region.csv'
    with path.open(encoding='utf-8') as handle:
        return {row['region']: float(row['kwh_per_kwp_year']) for row in csv.DictReader(handle)}


@lru_cache(maxsize=1)
def _self_consumption_rows() -> tuple[tuple, ...]:
    with (config.CONFIG_DIR / 'solar_self_consumption.csv').open(encoding='utf-8') as handle:
        return tuple((float(r['gen_min_kwh']), float(r['gen_max_kwh']) + 1, float(r['use_min_kwh']),
                      float(r['use_max_kwh']) + 1, float(r['self_consumption_share']), r['mcs_table'])
                     for r in csv.DictReader(handle))


def self_consumption_share(generation_kwh: float, use_kwh: float) -> tuple[float, str] | None:
    """MCS MGD 003 PV-only share for 'in half the day' occupancy, or None outside the table."""
    for gen_lo, gen_hi, use_lo, use_hi, share, table in _self_consumption_rows():
        if gen_lo <= generation_kwh < gen_hi and use_lo <= use_kwh < use_hi:
            return share, table
    return None


def solar_impact(features: Mapping) -> dict:
    # A reference design applied to the customer's own electricity use, never a survey.
    if features.get('has_solar') is not False:
        return unavailable('EXISTING_OR_UNKNOWN_SOLAR_INSTALLATION')
    if features.get('property_type') not in HOUSE_TYPES:
        return unavailable('NO_DOCUMENTED_ROOF_FOR_REFERENCE_DESIGN')
    region, use = features.get('region'), features.get('est_electricity_kwh_year')
    yields = regional_solar_yield()
    if region not in yields:
        return unavailable('MISSING_REGIONAL_SOLAR_YIELD')
    if not _number(use) or use == 0:
        return unavailable('MISSING_ELECTRICITY_USE')
    try:
        size = factors.get('solar_system_size_kwp')
        grid = factors.get('grid_electricity_per_kwh')
        losses = factors.get('grid_electricity_td_per_kwh')
    except factors.MissingFactorError:
        return unavailable('MISSING_EMISSIONS_FACTOR')
    generation = size.value * yields[region]
    lookup = self_consumption_share(generation, use)
    if lookup is None:
        return unavailable('OUTSIDE_MCS_SELF_CONSUMPTION_RANGE')
    share, table = lookup
    self_used = generation * share
    return _estimate('GHG Protocol avoided emissions: grid imports displaced by a reference rooftop PV design',
        [f'Illustrative design: {size.value:g} kWp south-facing system ({size.source} {size.edition} typical '
         'installation). Roof suitability, shading and orientation are not surveyed.',
         f'Generation {generation:,.0f} kWh/year = {size.value:g} kWp x {yields[region]:g} kWh/kWp/year '
         f'(PVGIS v5.3 for {region}, 14% system loss, 35 degree pitch).',
         f'Self-consumed share {share:.0%} from MCS MGD 003 table {table} (occupancy unknown, so "in half '
         f'the day"; no battery) for estimated electricity use {use:,.0f} kWh/year.',
         f'Only the {self_used:,.0f} kWh/year used at home displaces grid imports ({grid.factor_id} + '
         f'{losses.factor_id}, {grid.source} {grid.edition}); exported electricity gets no credit.',
         'First-year use-phase estimate; excludes panel manufacturing, installation and degradation.'],
        energy_kwh_per_year=round(self_used, 1),
        kg_co2e_per_year=round(self_used * (grid.value + losses.value), 3))


def green_home_impact(features: Mapping) -> dict:
    # EPC alone does not establish a saving. The one documented intervention is a gas
    # boiler -> air-source heat pump switch, applied to the customer's own gas use.
    if features.get('green_home_use_case') not in {None, 'heat_pump'}:
        return unavailable('NO_DOCUMENTED_SAVING_FOR_THIS_RETROFIT')
    if features.get('main_heating_fuel') != 'gas':
        return unavailable('NO_DOCUMENTED_GAS_HEATING_BASELINE')
    gas = features.get('est_gas_kwh_year')
    if not _number(gas) or gas == 0:
        return unavailable('MISSING_GAS_USE')
    try:
        high = factors.get('gas_tdcv_high_2023_kwh')
    except factors.MissingFactorError:
        return unavailable('MISSING_EMISSIONS_FACTOR')
    if gas > GAS_PLAUSIBILITY_MULTIPLE * high.value:
        return unavailable('GAS_USE_OUTSIDE_PLAUSIBLE_RANGE')
    try:
        efficiency = factors.get('gas_boiler_in_situ_efficiency')
        spf = factors.get('heat_pump_spf_ashp')
        gas_factor = factors.get('natural_gas_per_kwh')
        grid = factors.get('grid_electricity_per_kwh')
        losses = factors.get('grid_electricity_td_per_kwh')
    except factors.MissingFactorError:
        return unavailable('MISSING_EMISSIONS_FACTOR')
    heat = gas * efficiency.value
    electricity = heat / spf.value
    avoided = gas * gas_factor.value - electricity * (grid.value + losses.value)
    assumptions = [
        'Illustrative intervention: replace the gas boiler with an air-source heat pump for heating and hot '
        'water. A heat-loss survey would set the real design.',
        f'Baseline: {gas:,.0f} kWh/year of gas from bills in the feature window, all treated as heating and '
        'hot water (cooking gas is not separated).',
        f'Heat delivered = gas x {efficiency.value:.0%} boiler in-situ efficiency ({efficiency.source}); heat '
        f'pump electricity = heat / SPF {spf.value:g} ({spf.source}, median).',
        f'Gas displaced {gas:,.0f} kWh minus heat pump electricity added {electricity:,.0f} kWh = '
        f'{gas - electricity:,.0f} kWh/year less purchased energy; emissions use {gas_factor.factor_id}, '
        f'{grid.factor_id} and {losses.factor_id} ({grid.source} {grid.edition}).',
        'Excludes refrigerant leakage, manufacturing, installation, fabric upgrades and rebound.']
    if features.get('energy_from_dual_fuel'):
        assumptions.append('Gas use is apportioned from a dual-fuel bill (D6), so the baseline is less certain.')
    return _estimate('GHG Protocol avoided emissions: gas boiler versus air-source heat pump',
        assumptions, energy_kwh_per_year=round(max(0.0, gas - electricity), 1),
        kg_co2e_per_year=round(max(0.0, avoided), 3))


def paperless_impact(features: Mapping) -> dict:
    statements = features.get('paper_statements_12m')
    if not _number(statements):
        return unavailable('MISSING_PAPER_STATEMENT_COUNT')
    if statements == 0:
        return unavailable('NO_PAPER_STATEMENTS_OBSERVED')
    try:
        sheets = factors.get('sheets_per_statement')
        mass = factors.get('paper_sheet_mass_kg')
        paper = factors.get('paper_per_tonne')
    except factors.MissingFactorError:
        return unavailable('MISSING_DOCUMENTED_PAPER_ASSUMPTION')
    count = statements * sheets.value
    return _estimate('Avoided paper production from switching statements to digital',
        [f'Baseline: {statements:g} paper statements in the 12-month feature window, {sheets.value:g} '
         f'sheet/statement ({sheets.source}: the minimum, so the estimate is a lower bound).',
         f'Sheet mass {mass.value:g} kg; production factor {paper.value:g} kg CO2e/tonne ({paper.source} {paper.edition}).',
         'All paper statements replaced; excludes postage and digital-service emissions.'],
        paper_sheets_per_year=count, kg_co2e_per_year=round(count * mass.value * paper.value / 1000, 3))


_BY_PRODUCT = {"GP01": ev_impact, "GP02": green_home_impact, "GP03": solar_impact, "GP04": paperless_impact}


def compute_impacts(features: Mapping) -> dict[str, dict]:
    """All four products for one customer, keyed GP01-GP04 (the bundle's `impacts[customer_id]`)."""
    return {code: fn(features) for code, fn in _BY_PRODUCT.items()}


def main() -> None:
    import pandas as pd
    features = pd.read_parquet(config.PROCESSED_DIR / 'features.parquet')
    impacts = {row['customer_id']: compute_impacts(row) for row in to_records(features)}
    path = config.PROCESSED_DIR / 'impacts.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(impacts, allow_nan=False))
    counts = {code: sum(rows[code]['status'] == 'ESTIMATED' for rows in impacts.values()) for code in _BY_PRODUCT}
    print(f'Impacts for {len(impacts)} customers -> {path}; estimated coverage {counts}')


if __name__ == '__main__':
    main()
