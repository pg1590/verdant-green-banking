"""Small deterministic integration fixtures, NOT P1 data or P3/P2 outputs."""
import json
from pathlib import Path


def demo_bundle() -> dict:
    common = dict(age=38, region='Yorkshire and The Humber', income_band='Q3',
                  home_status='owner_mortgage', property_type='semi_detached',
                  epc_rating='E', has_solar=False, vehicle_type='petrol',
                  statement_pref='paper', marketing_consent=True,
                  data_as_of='2026-06-30', feature_cutoff_date='2026-06-30',
                  data_completeness=0.95, conflict_flags=[], data_flags='')
    overrides = [
        ('C10001', {}, 'Homeowner and commuter'),
        ('C10002', {'home_status': 'renter', 'region': 'London'}, 'Renter with high electricity spend'),
        ('C10003', {'vehicle_type': 'ev', 'statement_pref': 'digital'}, 'Existing EV and digital customer'),
        ('C10004', {'has_solar': True, 'region': 'Scotland'}, 'Already has solar'),
        ('C10005', {'home_status': None, 'has_solar': None, 'vehicle_type': None,
                    'statement_pref': None, 'data_completeness': 0.2}, 'Missing data'),
        ('C10006', {'vehicle_type': 'ev', 'ev_replacement_use_case': True,
                    'conflict_flags': ['VEHICLE_FUEL_CONFLICT']}, 'Conflicting road-fuel evidence'),
        ('C10007', {'vehicle_type': 'none', 'conflict_flags': ['VEHICLE_SOLD'],
                    'region': 'Wales'}, 'Sold vehicle; updated circumstances'),
        ('C10008', {'marketing_consent': False}, 'No marketing consent'),
        ('C10009', {'requires_paper_statements': True, 'region': 'North West'}, 'Needs paper statements'),
        ('C10010', {'property_type': 'flat', 'epc_rating': 'B', 'region': 'London'}, 'Property suitability review'),
    ]
    customers, predictions, impacts = [], {}, {}
    signals = {
        'GP01': (.84, .87, ['HIGH_FUEL_SPEND', 'HIGH_ANNUAL_MILEAGE'], {'avg_monthly_fuel_spend_gbp': 210, 'est_annual_km': 18400}),
        'GP02': (.72, .85, ['LOW_EPC_RATING'], {'epc_rating': 'E'}),
        'GP03': (.68, .82, ['HIGH_ELECTRICITY_SPEND'], {'avg_monthly_electricity_spend_gbp': 140}),
        'GP04': (.91, .94, ['PAPER_STATEMENTS', 'DIGITAL_ENGAGEMENT'], {'paper_statements_per_year': 12, 'app_logins_last3m': 42}),
    }
    # Deliberately no emissions numbers: only P2 supplies sourced impact estimates.
    for cid, changes, scenario in overrides:
        customer = dict(common, customer_id=cid, scenario=scenario, **changes)
        customers.append(customer)
        predictions[cid] = {}
        impacts[cid] = {}
        for code, (propensity, confidence, reasons, evidence) in signals.items():
            ev = dict(evidence)
            if code == 'GP02':
                ev['epc_rating'] = customer['epc_rating']
            if code == 'GP04':
                ev['statement_pref'] = customer['statement_pref']
            predictions[cid][code] = dict(propensity=propensity, confidence=confidence,
                reason_codes=reasons, evidence=ev, model_version='mock_contract_v1', data_mode='DEMO_MOCK')
            impacts[cid][code] = {'status': 'INSUFFICIENT_DATA'}
    return {'schema_version': '1.0', 'data_mode': 'DEMO_MOCK', 'customers': customers,
            'predictions': predictions, 'impacts': impacts, 'portfolio_history': []}


def write_demo(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(demo_bundle(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return target
