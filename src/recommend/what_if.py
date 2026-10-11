"""What-if: replay one customer with changed circumstances (brief: "changed customer circumstances").

A relationship manager changes profile facts (sold the car, became a homeowner, installed solar,
switched to digital statements). The same components then run again, each doing its own job:
P5 rules decide eligibility, P3's frozen models rescore propensity and confidence, and P2's engine
recomputes impact. Nothing is saved. Transaction-derived features (spend, mileage, energy use,
conflict flags) stay as observed, because a profile edit does not create new transactions.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

from src.carbon.engine import compute_impacts
from src.schemas import tables as T
from .engine import build_recommendations

USE_CASES = (None, 'insulation', 'heat_pump', 'efficient_heating', 'energy_efficiency_retrofit')
EDITABLE = {
    'vehicle_type': T.VEHICLE_TYPES, 'num_vehicles': T.VEHICLE_COUNTS, 'home_status': T.HOME_STATUS,
    'property_type': T.PROPERTY_TYPES, 'epc_rating': T.EPC_RATINGS, 'main_heating_fuel': T.HEATING_FUELS,
    'has_solar': (True, False), 'statement_pref': T.STATEMENT_PREFS, 'marketing_consent': (True, False),
    'ev_replacement_use_case': (True, False, None), 'ev_purchase_use_case': (True, False, None),
    'green_home_use_case': USE_CASES, 'requires_paper_statements': (True, False, None),
}


@lru_cache(maxsize=1)
def _models():
    """P3's frozen model files, loaded once; None if the ML stack is unavailable."""
    try:
        from src.features.snapshots import frozen_bundles
        from src.models.config import load_config
        cfg = load_config()
        return cfg, frozen_bundles(cfg)
    except Exception:
        return None


def _rescore(record: Mapping) -> dict | None:
    models = _models()
    if models is None:
        return None
    import pandas as pd
    from src.models.predict import score, to_p5_predictions
    cfg, bundles = models
    return to_p5_predictions(score(pd.DataFrame([dict(record)]), cfg, bundles))[record['customer_id']]


def _view(row: Mapping) -> dict:
    impact = row['impact']
    return {'eligible_to_show': row['eligible_to_show'], 'propensity': row['propensity'],
            'confidence': row['confidence'], 'reason_codes': row['reason_codes'],
            'suppression_reasons': row['suppression_reasons'], 'impact_status': impact['status'],
            'kg_co2e_per_year': impact.get('kg_co2e_per_year'), 'impact_reason': impact.get('reason')}


def what_if(customer: Mapping, predictions: Mapping, impacts: Mapping, changes: Mapping,
            catalogue: dict) -> dict:
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        raise ValueError(f'Only profile facts can change: {sorted(unknown)} are not editable')
    for field, value in changes.items():
        if value not in EDITABLE[field]:
            raise ValueError(f'{field}: {value!r} is not one of {list(EDITABLE[field])}')
    notes = ['Simulation only: nothing is saved and the customer record is unchanged.',
             'Spending, mileage, energy use and conflict flags stay as observed in the transactions.']
    after = {**customer, **changes}
    if after.get('vehicle_type') == 'none' and after.get('num_vehicles') != 0:
        after['num_vehicles'] = 0
        notes.append('No vehicle means 0 cars, so the car count was set to 0.')
    if after.get('vehicle_type') not in (None, 'none') and after.get('num_vehicles') == 0:
        after['num_vehicles'] = 1
        notes.append('A vehicle type needs at least 1 car, so the car count was set to 1.')
    new_predictions = _rescore(after)
    if new_predictions is None:
        new_predictions = predictions
        notes.append('Model files are unavailable, so propensity and confidence were not rescored.')
    before_rows = {r['product_code']: r for r in build_recommendations(customer, predictions, impacts, catalogue)}
    after_rows = {r['product_code']: r for r in build_recommendations(after, new_predictions, compute_impacts(after), catalogue)}
    products = []
    for code, product in catalogue['products'].items():
        b, a = _view(before_rows[code]), _view(after_rows[code])
        products.append({'product_code': code, 'product_name': product['name'], 'before': b, 'after': a,
                         'changed': [k for k in b if b[k] != a[k]]})
    return {'customer_id': customer['customer_id'], 'scope': 'SIMULATION_NOT_SAVED',
            'changes': {k: after[k] for k in changes}, 'rescored': new_predictions is not predictions,
            'notes': notes, 'products': products}
