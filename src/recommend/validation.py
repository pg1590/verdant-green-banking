"""Validate team exports against P1's canonical schema, without guessing aliases."""
from pathlib import Path
from functools import lru_cache

import yaml
from src.schemas import tables as T
from .engine import probability, valid_date

ENUM_FIELDS = {
    'region': T.REGIONS, 'income_band': T.INCOME_BANDS,
    'home_status': T.HOME_STATUS, 'property_type': T.PROPERTY_TYPES,
    'epc_rating': T.EPC_RATINGS, 'vehicle_type': T.VEHICLE_TYPES,
    'statement_pref': T.STATEMENT_PREFS, 'main_heating_fuel': T.HEATING_FUELS,
}
FORBIDDEN_FIELDS = set(T.LATENT_TRAITS) | {'persona', 'took_up', 'label_window_start', 'label_window_end', 'true_category'}


@lru_cache(maxsize=1)
def known_data_flags() -> frozenset:
    path = Path(__file__).resolve().parents[2] / 'config' / 'reason_codes.yaml'
    with path.open(encoding='utf-8') as handle:
        return frozenset(yaml.safe_load(handle)['data_flags'])


@lru_cache(maxsize=1)
def configured_factor_edition() -> str:
    path = Path(__file__).resolve().parents[2] / 'config' / 'settings.yaml'
    with path.open(encoding='utf-8') as handle:
        edition = yaml.safe_load(handle)['carbon']['factor_edition']
    if not isinstance(edition, str) or not edition.strip():
        raise ValueError('settings.carbon.factor_edition must be a nonempty string')
    return edition


def validate_team_customer(customer: dict) -> None:
    cid = customer.get('customer_id', '<unknown>')
    leaked = FORBIDDEN_FIELDS & customer.keys()
    if leaked:
        raise ValueError(f'{cid}: labels/hidden traits must not enter customer bundles: {sorted(leaked)}')
    for field, allowed in ENUM_FIELDS.items():
        value = customer.get(field)
        if value is not None and (not isinstance(value, str) or value not in allowed):
            raise ValueError(f'{cid}: invalid canonical {field}: {value!r}')
    if not probability(customer.get('data_completeness')):
        raise ValueError(f'{cid}: TEAM_OUTPUTS requires data_completeness in [0, 1]')
    cutoff = customer.get('feature_cutoff_date')
    if not valid_date(cutoff):
        raise ValueError(f'{cid}: TEAM_OUTPUTS requires ISO feature_cutoff_date')
    as_of = customer.get('data_as_of')
    if as_of is not None and (not valid_date(as_of) or as_of != cutoff):
        raise ValueError(f'{cid}: data_as_of must match feature_cutoff_date for this snapshot')
    for field in ('conflict_flags', 'data_flags'):
        if field not in customer:
            raise ValueError(f'{cid}: TEAM_OUTPUTS requires {field}; export [] or empty string explicitly when none')
    conflicts = customer['conflict_flags']
    if not isinstance(conflicts, list) or any(not isinstance(x, str) or x not in T.CONFLICT_FLAGS for x in conflicts):
        raise ValueError(f'{cid}: invalid conflict_flags code/list')
    flags = customer.get('data_flags', '')
    if not isinstance(flags, str):
        raise ValueError(f'{cid}: data_flags must be comma-separated codes')
    known = known_data_flags()
    if any(flag.strip() not in known for flag in flags.split(',') if flag.strip()):
        raise ValueError(f'{cid}: unknown data_flags code')
    # Optional P1 Customer 360 display fields.
    held = customer.get('products_held')
    if held is not None and (not isinstance(held, list) or any(x not in T.HELD_PRODUCT_TYPES for x in held)):
        raise ValueError(f'{cid}: products_held must be a list of canonical product types')
    for field in ('n_transactions_window', 'active_months_window'):
        value = customer.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f'{cid}: {field} must be a nonnegative integer')
