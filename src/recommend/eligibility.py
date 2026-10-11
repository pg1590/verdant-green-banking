"""P5-owned shared eligibility interface; P3 starts here.

P3: eligible_customer_product_pairs() filters historical training cohorts.
P1: supply canonical raw customer attributes alongside cutoff features.
Full handoff and examples: docs/integration.md. Rules: config/products.yaml.
This module never reads model scores, labels or carbon to decide eligibility.
"""
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

CATALOGUE_PATH = Path(__file__).resolve().parents[2] / 'config' / 'products.yaml'


def load_catalogue(path: str | Path = CATALOGUE_PATH) -> dict:
    with Path(path).open(encoding='utf-8') as handle:
        catalogue = yaml.safe_load(handle)
    if set(catalogue['products']) != {'GP01', 'GP02', 'GP03', 'GP04'}:
        raise ValueError('Catalogue must contain GP01–GP04 exactly')
    for code, product in catalogue['products'].items():
        for rule in product['rules']:
            if rule['op'] not in {'gte', 'in', 'eq'}:
                raise ValueError(f'Unsupported rule operator for {code}')
    return catalogue


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, float) and not isfinite(value)
    )


@dataclass(frozen=True)
class EligibilityResult:
    product_code: str
    eligible_to_show: bool
    suppression_reason: str | None
    suppression_reasons: tuple[str, ...]
    policy_version: str
    eligibility_scope: str = 'DISPLAY_ONLY_NOT_CREDIT_APPROVAL'

    def to_dict(self) -> dict:
        result = asdict(self)
        result['suppression_reasons'] = list(self.suppression_reasons)
        return result


def check_eligibility(customer: Mapping[str, Any], product_code: str,
                      catalogue: dict | None = None, *, purpose: str = 'recommendation') -> EligibilityResult:
    """purpose='training' ignores marketing consent, never structural eligibility.

    Supply the customer snapshot as of feature_cutoff_date for training. Labels,
    propensity, future purchases and impact never influence this function.
    """
    if purpose not in {'recommendation', 'training'}:
        raise ValueError('purpose must be recommendation or training')
    cat = catalogue if catalogue is not None else load_catalogue()
    if product_code not in cat['products']:
        raise ValueError(f'Unknown product code: {product_code}')
    product = cat['products'][product_code]
    reasons = []
    for rule in product['rules']:
        when = rule.get('when')
        if when and customer.get(when['field']) != when['eq']:
            continue
        skip = rule.get('unless_present')
        if skip and not is_missing(customer.get(skip)):
            continue
        value = customer.get(rule['field'])
        if is_missing(value):
            if not rule.get('optional', False):
                reasons.append(rule.get('missing_reason', f'MISSING_{rule["field"].upper()}'))
            continue
        expected = rule['value']
        if rule['op'] == 'gte':
            valid = type(value) in {int, float} and isfinite(value) and value >= expected
        elif rule['op'] == 'eq':
            valid = type(value) is type(expected) and value == expected
        else:
            valid = any(type(value) is type(item) and value == item for item in expected)
        if not valid:
            reasons.append(rule['reason'])
    conflicts = customer.get('conflict_flags', [])
    if conflicts is None:
        conflicts = []
    if not isinstance(conflicts, (list, tuple)) or any(not isinstance(x, str) for x in conflicts):
        reasons.append('INVALID_CONFLICT_FLAGS')
    else:
        relevant = {
            'GP01': {'VEHICLE_FUEL_CONFLICT', 'VEHICLE_SOLD', 'VEHICLE_STATUS_CONFLICT'},
            'GP02': {'HOME_STATUS_CONFLICT', 'EPC_CONFLICT'},
            'GP03': {'HOME_STATUS_CONFLICT', 'SOLAR_STATUS_CONFLICT'},
            'GP04': {'STATEMENT_PREF_CONFLICT'},
        }[product_code]
        reasons.extend(sorted(set(conflicts) & relevant))
    if purpose == 'recommendation' and product['kind'] == 'finance':
        consent = customer.get('marketing_consent')
        if consent is not True:
            reasons.append('MISSING_MARKETING_CONSENT' if is_missing(consent) else 'NO_MARKETING_CONSENT')
    unique = tuple(dict.fromkeys(reasons))
    return EligibilityResult(product_code, not unique, unique[0] if unique else None, unique, cat['policy_version'])


def eligible_customer_product_pairs(customers: Iterable[Mapping[str, Any]],
                                     catalogue: dict | None = None) -> list[dict]:
    """One row per customer/product, ready to merge into P3's training features.

    Pass DataFrame.to_dict('records') for pandas input. Preserve ineligible rows
    in audit outputs; filter eligible_to_show == True before fitting each model.
    """
    cat = catalogue if catalogue is not None else load_catalogue()
    rows = []
    for customer in customers:
        if is_missing(customer.get('customer_id')):
            raise ValueError('customer_id is required')
        for code in cat['products']:
            result = check_eligibility(customer, code, cat, purpose='training').to_dict()
            rows.append({'customer_id': customer['customer_id'], **result})
    return rows
