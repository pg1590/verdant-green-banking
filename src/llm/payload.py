"""The minimised payload the LLM is allowed to see, built from ONE final eligible row
(src/recommend/engine.py). Owner: P4.

Never sent: customer_id, raw propensity/confidence, other products, free-text evidence.
Numbers are pre-formatted display strings so every number the LLM may repeat is checkable.
"""
from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from typing import Any

from . import phrases

TOPICS = {
    "EV_FINANCE": "switching to an electric vehicle",
    "GREEN_HOME": "making your home more energy efficient",
    "SOLAR_FINANCE": "generating your own electricity with solar panels",
    "PAPERLESS_BANKING": "switching to paperless statements",
}
# Confidence -> wording band (the LLM never sees the raw score).
CERTAINTY_BANDS = [(0.75, "high"), (0.5, "medium"), (0.0, "low")]
EPC_LETTERS = "ABCDEFG"

# Text evidence must look like a short categorical value ("petrol", "semi_detached").
# Anything else (free text, instructions) is withheld from the LLM to block prompt injection.
_SAFE_CATEGORICAL = re.compile(r"^[A-Za-z0-9 _/&'-]{1,40}$")


def fmt_number(value: float) -> str:
    """Whole numbers with thousands separators; one decimal place below 10."""
    if abs(value) < 10 and value != int(value):
        return f"{value:,.1f}"
    return f"{round(value):,}"


def fmt_date(value: str) -> str:
    d = dt.date.fromisoformat(str(value)[:10])
    return f"{d.day} {d.strftime('%B %Y')}"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_safe_evidence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_SAFE_CATEGORICAL.match(value))
    return isinstance(value, (int, float, bool))


def withheld_evidence_keys(row: Mapping) -> list[str]:
    return [k for k, v in (row.get("evidence") or {}).items() if v is not None and not is_safe_evidence(v)]


def format_evidence(key: str, value: Any) -> dict:
    label = phrases.evidence_label(key)
    if isinstance(value, bool) or ((key.startswith(("has_", "is_"))) and value in (0, 1)):
        display = "yes" if value else "no"
    elif key == "epc_rating_ord" and _is_number(value) and 1 <= value <= 7:
        display = EPC_LETTERS[int(value) - 1]
    elif key == 'income_band_ord' and _is_number(value) and 1 <= value <= 5:
        display = f'Q{int(value)}'
    elif key == 'num_vehicles' and _is_number(value) and value >= 2:
        display = '2 or more'  # the schema caps household cars at 2 = "two or more"
    elif _is_number(value):
        if key.endswith("_gbp") or key.startswith('spend_') or key == 'energy_spend_per_person_m':
            display = f"£{fmt_number(value)}"
        elif key.endswith("_km") or key in {'est_annual_km', 'km_per_vehicle'}:
            display = f"{fmt_number(value)} km"
        elif key.endswith("_kwh") or key in {'est_electricity_kwh_year', 'est_gas_kwh_year', 'electricity_kwh_per_person'}:
            display = f"{fmt_number(value)} kWh"
        elif key.endswith("_pct"):
            display = f"{fmt_number(value)}%"
        else:
            display = fmt_number(value)
    else:
        display = str(value).replace("_", " ")
    return {"label": label, "value": display}


def certainty_band(confidence: float | None) -> str:
    if not _is_number(confidence):
        return "low"
    return next(name for floor, name in CERTAINTY_BANDS if confidence >= floor)


def impact_display(impact: Mapping | None) -> dict:
    impact = impact or {}
    if impact.get("status") != "ESTIMATED":
        return {"status": "INSUFFICIENT_DATA", "text": "Not enough data to estimate the impact yet"}
    parts = []
    if _is_number(impact.get("kg_co2e_per_year")):
        parts.append(f"about {fmt_number(impact['kg_co2e_per_year'])} kg CO2e per year")
    if _is_number(impact.get("energy_kwh_per_year")):
        parts.append(f"about {fmt_number(impact['energy_kwh_per_year'])} kWh of energy per year")
    if _is_number(impact.get("paper_sheets_per_year")):
        parts.append(f"about {fmt_number(impact['paper_sheets_per_year'])} sheets of paper per year")
    if not parts:
        return {"status": "INSUFFICIENT_DATA", "text": "Not enough data to estimate the impact yet"}
    horizon = impact.get("horizon_years", 1)
    return {"status": "ESTIMATED", "text": " and ".join(parts), "is_estimate": True,
            "horizon": f"{fmt_number(horizon)} year" + ("" if horizon == 1 else "s"),
            "method": impact.get("method") or "method not stated"}


def topic(row: Mapping) -> str:
    """What the estimate is about. Green Home is sized for one modelled intervention (P2),
    so the wording names it instead of implying a general retrofit."""
    method = str((row.get("impact") or {}).get("method") or "").lower()
    if row.get("opportunity") == "GREEN_HOME" and "heat pump" in method:
        return "replacing your gas boiler with a heat pump"
    return TOPICS.get(row.get("opportunity", ""), "this greener option")


def build_payload(row: Mapping) -> dict:
    """The only data the customer-insight LLM call sees (data minimisation)."""
    opp = row.get("opportunity", "")
    return {
        "product": row.get("product_name") or opp.replace("_", " ").title(),
        "topic": topic(row),
        "reasons": [{"code": c, "meaning": phrases.clause(c)} for c in row.get("reason_codes") or []],
        "evidence": [format_evidence(k, v) for k, v in (row.get("evidence") or {}).items() if is_safe_evidence(v)],
        "estimated_impact": impact_display(row.get("impact")),
        "certainty": certainty_band(row.get("confidence")),
        "data_as_of": fmt_date(row["data_timestamp"]) if row.get("data_timestamp") else "unknown",
        'data_quality_flags': row.get('data_flags', ''),
        'factor_edition': (row.get('impact') or {}).get('factor_edition'),
    }
