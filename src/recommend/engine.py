"""Join P1/P2/P3 outputs, apply eligibility first and rank by propensity × confidence."""
from datetime import date
from math import isfinite
from typing import Any, Mapping

from .eligibility import check_eligibility, load_catalogue


def probability(value: Any) -> bool:
    return type(value) in {int, float} and isfinite(value) and 0 <= value <= 1


def valid_date(value: Any) -> bool:
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def clean_impact(value: Mapping | None) -> dict:
    unavailable = {'kg_co2e_per_year': None, 'energy_kwh_per_year': None,
                   'paper_sheets_per_year': None, 'method': None, 'factor_edition': None, 'assumptions': [],
                   'horizon_years': 1, 'status': 'INSUFFICIENT_DATA'}
    if not isinstance(value, Mapping) or value.get('status') != 'ESTIMATED':
        reason = value.get('reason') if isinstance(value, Mapping) else None
        # Keep P2's reason code so the API and UI can say why no estimate exists.
        return {**unavailable, 'reason': reason} if isinstance(reason, str) and reason.strip() else unavailable
    metrics = ['kg_co2e_per_year', 'energy_kwh_per_year', 'paper_sheets_per_year']
    numbers = [value.get(k) for k in metrics]
    if not any(type(x) in {int, float} and isfinite(x) for x in numbers):
        return unavailable
    if any(x is not None and (type(x) not in {int, float} or not isfinite(x) or x < 0) for x in numbers):
        return unavailable
    if not isinstance(value.get('method'), str) or not value['method'].strip():
        return unavailable
    if not isinstance(value.get('assumptions'), list) or not value['assumptions'] or any(not isinstance(x, str) for x in value['assumptions']):
        return unavailable
    edition = value.get('factor_edition')
    if not isinstance(edition, str) or not edition.strip():
        return unavailable
    horizon = value.get('horizon_years')
    if type(horizon) not in {int, float} or not isfinite(horizon) or horizon <= 0:
        return unavailable
    return {**unavailable, **{k: value[k] for k in unavailable if k in value}, 'status': 'ESTIMATED'}


def build_recommendations(customer: Mapping, predictions: Mapping[str, Mapping],
                          impacts: Mapping[str, Mapping] | None = None,
                          catalogue: dict | None = None) -> list[dict]:
    cat = catalogue if catalogue is not None else load_catalogue()
    impacts = impacts or {}
    if not isinstance(customer.get('customer_id'), str) or not customer['customer_id'].strip():
        raise ValueError('customer_id is required')
    rows = []
    for code, product in cat['products'].items():
        eligibility = check_eligibility(customer, code, cat)
        structural = check_eligibility(customer, code, cat, purpose='training')
        pred = predictions.get(code, {})
        if not isinstance(pred, Mapping):
            pred = {}
        reasons = list(eligibility.suppression_reasons)
        valid = probability(pred.get('propensity')) and probability(pred.get('confidence'))
        reason_codes = pred.get('reason_codes', [])
        evidence = pred.get('evidence', {})
        if not valid:
            reasons.append('MISSING_OR_INVALID_MODEL_OUTPUT')
        if not isinstance(reason_codes, list) or not reason_codes or any(not isinstance(x, str) or not x.strip() for x in reason_codes):
            reasons.append('MISSING_OR_INVALID_REASON_CODES')
            reason_codes = []
        if not isinstance(evidence, Mapping) or not evidence:
            reasons.append('MISSING_OR_INVALID_EVIDENCE')
            evidence = {}
        timestamp = customer.get('feature_cutoff_date', customer.get('data_as_of'))
        if not valid_date(timestamp):
            reasons.append('MISSING_OR_INVALID_DATA_TIMESTAMP')
        model_version = pred.get('model_version')
        if not isinstance(model_version, str) or not model_version.strip():
            reasons.append('MISSING_MODEL_VERSION')
        confidence = pred.get('confidence') if valid else None
        completeness = customer.get('data_completeness')
        if completeness is not None:
            if not probability(completeness):
                reasons.append('INVALID_DATA_COMPLETENESS')
            elif confidence is not None:
                # P3 may already include completeness. Cap rather than multiply twice.
                confidence = min(confidence, completeness)
                if completeness == 0:
                    reasons.append('INSUFFICIENT_CUSTOMER_DATA')
        if confidence == 0:
            reasons.append('ZERO_CONFIDENCE')
        unique = list(dict.fromkeys(reasons))
        shown = not unique
        rows.append({
            'customer_id': customer['customer_id'], 'opportunity': product['opportunity'],
            'product_code': code, 'product_name': product['name'],
            'propensity': pred.get('propensity') if valid else None,
            'confidence': confidence, 'model_confidence': pred.get('confidence') if valid else None,
            'reason_codes': reason_codes, 'evidence': dict(evidence),
            'data_flags': customer.get('data_flags', ''),
            'impact': clean_impact(impacts.get(code)),
            'eligible_to_show': shown, 'structurally_eligible': structural.eligible_to_show,
            'suppression_reason': unique[0] if unique else None, 'suppression_reasons': unique,
            'data_timestamp': timestamp, 'model_version': model_version,
            'policy_version': cat['policy_version'], 'eligibility_scope': eligibility.eligibility_scope,
            'ranking_score': round(pred['propensity'] * confidence, 8) if shown else None,
            'rank': None, 'action': product['action'] if shown else None,
            'data_mode': pred.get('data_mode', 'TEAM_OUTPUTS'),
        })
    rows.sort(key=lambda r: (not r['eligible_to_show'], -(r['ranking_score'] or 0), r['product_code']))
    for rank, row in enumerate((r for r in rows if r['eligible_to_show']), start=1):
        row['rank'] = rank
    return rows
