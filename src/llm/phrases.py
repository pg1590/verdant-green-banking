"""Customer-facing wording for reason codes and evidence keys. Owner: P4.

P3 owns the reason-code dictionary (config/reason_codes.yaml): which codes exist and when they fire.
P4 owns how a code is *worded* to a customer and which words prove the LLM explained it.
Codes missing here still work: wording falls back to the dictionary's customer_text/label, then to
the code name, so a new P3 code never breaks the LLM layer.
"""
from __future__ import annotations

import re

from src.common import config

# code -> (clause used in sentences and sent to the LLM, keywords that count as "explained")
# Keywords match at a word start: "renovat" matches "renovation".
PHRASES: dict[str, tuple[str, list[str]]] = {
    'INCOME_PROFILE': ('your recorded income profile contributes to this suggestion', ['income', 'profile']),
    'HIGH_ENERGY_SPEND_PER_PERSON': ('your energy spending per household member is high', ['energy', 'spending', 'household']),
    'DIGITAL_WEB_ENGAGEMENT': ('you already use online banking', ['online', 'web', 'digital']),
    # GP01 EV finance
    "HIGH_FUEL_SPEND": ("your monthly spending on fuel or vehicle charging is high", ["fuel", "charging"]),
    "RECENT_HIGH_FUEL_SPEND": ("your fuel spending has been high in recent months", ["fuel", "petrol", "diesel"]),
    "HIGH_ANNUAL_MILEAGE": ("you drive a high mileage each year", ["mileage", "miles", "km", "kilometre", "drive", "driving"]),
    "EXISTING_VEHICLE": ("you currently run a petrol, diesel or hybrid car", ["vehicle", "car"]),
    "EXISTING_VEHICLE_LOAN": ("you have an existing vehicle loan", ["loan", "finance"]),
    "HIGH_AUTOMOTIVE_SPEND": ("you have regular vehicle running costs", ["running cost", "vehicle", "car", "maintenance", "servic"]),
    "HIGH_MILEAGE_PER_VEHICLE": ("each car in your household covers a high mileage", ["mileage", "miles", "km", "kilometre", "drive", "driving"]),
    "MULTIPLE_VEHICLES": ("your household runs more than one car", ["car", "vehicle"]),
    # GP02 green home / GP03 solar
    "HOMEOWNER": ("you own your home", ["own", "homeowner"]),
    "EXISTING_MORTGAGE": ("you have a mortgage with us", ["mortgage"]),
    "HAS_MORTGAGE": ("you have a mortgage with us", ["mortgage"]),
    "LOW_EPC_RATING": ("your home's energy rating has room to improve", ["epc", "energy rating", "rating", "efficien"]),
    "POOR_EPC_RATING": ("your home's energy rating has room to improve", ["epc", "energy rating", "rating", "efficien"]),
    "GAS_HEATING": ("your home is heated by gas or oil", ["gas", "oil", "heating", "boiler"]),
    "HIGH_GAS_SPEND": ("your gas use is on the high side", ["gas", "heating"]),
    "HIGH_HOME_IMPROVEMENT_SPEND": ("you've recently spent on home improvements", ["improvement", "renovat", "diy", "upgrade"]),
    "HOME_IMPROVEMENT_ACTIVITY": ("you regularly spend on home improvements", ["improvement", "renovat", "diy", "upgrade"]),
    "HIGH_ELECTRICITY_SPEND": ("your electricity use is on the high side", ["electricity", "electric", "power"]),
    "RECENT_HIGH_ELECTRICITY_SPEND": ("your electricity spending has been high in recent months", ["electricity", "electric", "power"]),
    "HIGH_ELECTRICITY_PER_PERSON": ("your electricity use per household member is high", ["electricity", "electric", "power"]),
    "SUITABLE_PROPERTY_TYPE": ("your home is a house with its own roof", ["house", "roof", "bungalow", "property"]),
    "SUITABLE_PROPERTY": ("your home is a house with its own roof", ["house", "roof", "bungalow", "property"]),
    "NO_EXISTING_SOLAR": ("you don't have solar panels yet", ["solar", "panel"]),
    "HAS_SAVINGS": ("you hold savings with us", ["saving"]),
    # GP04 paperless
    "PAPER_STATEMENTS": ("you still receive paper statements", ["paper", "statement", "post"]),
    "DIGITAL_ENGAGEMENT": ("you already bank digitally", ["app", "online", "digital"]),
    "HIGH_APP_USAGE": ("you use our mobile app regularly", ["app", "online", "digital", "mobile"]),
    "HIGH_WEB_USAGE": ("you use online banking regularly", ["online", "web", "digital", "internet"]),
}

# Evidence keys whose generic label would read badly.
EVIDENCE_LABELS = {
    'income_band_ord': 'recorded household income band',
    'energy_spend_per_person_m': 'average monthly energy spend per household member',
    'is_gas_or_oil_heated': 'gas or oil heating',
    'spend_fuel_m': 'average monthly fuel and vehicle charging spend',
    'spend_fuel_recent_m': 'average monthly fuel and charging spend in the last three months',
    'est_electricity_kwh_year': 'estimated annual electricity use',
    'est_gas_kwh_year': 'estimated annual gas use',
    'avg_app_logins_m': 'average monthly app logins',
    'paper_statements_12m': 'paper statements in the feature year',
    "est_annual_km": "estimated annual mileage",
    "km_per_vehicle": "estimated annual mileage per car",
    "num_vehicles": "cars in the household",
    "electricity_kwh_per_person": "estimated annual electricity use per household member",
    "epc_rating_ord": "EPC rating",
    "epc_rating": "EPC rating",
    "app_logins_last3m": "app logins in the last 3 months",
    "avg_monthly_app_logins": "average monthly app logins",
    "avg_monthly_web_sessions": "average monthly online banking sessions",
    "has_combustion_vehicle": "petrol, diesel or hybrid vehicle",
    "is_house": "lives in a house",
    "statement_pref": "statement preference",
}
_ABBREVIATIONS = {"avg": "average", "est": "estimated", "num": "number of", "pct": "percent", "m": "monthly",
                  "epc": "EPC", "ev": "EV", "co2e": "CO2e", "last3m": "in the last 3 months"}
_GENERIC_WORDS = {"high", "low", "recent", "existing", "has", "with", "your", "you", "the", "and", "spend"}


def _dictionary() -> dict:
    """Reason-code entries in either format: Sajiv's nested {"reason_codes": {...}} or P3's flat map."""
    raw = config.reason_codes() or {}
    return raw.get("reason_codes", raw) if isinstance(raw, dict) else {}


def short_text(code: str) -> str:
    """Noun phrase for the card's "Why you're seeing this" line."""
    entry = _dictionary().get(code) or {}
    text = entry.get("customer_text") or entry.get("label")
    if text:
        return text[0].lower() + text[1:]
    return code.lower().replace("_", " ")


def clause(code: str) -> str:
    """Full clause for sentences ("We noticed <clause>"); also what the LLM sees."""
    if code in PHRASES:
        return PHRASES[code][0]
    return "your data shows " + short_text(code)


def keywords(code: str) -> list[str]:
    if code in PHRASES:
        return PHRASES[code][1]
    words = re.findall(r"[a-z]+", short_text(code)) + code.lower().split("_")
    return sorted({w for w in words if len(w) >= 4 and w not in _GENERIC_WORDS}) or [code.lower()]


def evidence_label(key: str) -> str:
    if key in EVIDENCE_LABELS:
        return EVIDENCE_LABELS[key]
    words = [w for w in key.split("_") if w and w not in {"gbp", "km", "kwh", "pct", "count", "miles"}]
    return " ".join(_ABBREVIATIONS.get(w, w) for w in words)
