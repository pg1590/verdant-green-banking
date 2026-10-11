"""Raw-table and feature-table contracts (definitions in docs/data.md).

Changing a column here is a contract change: agree it at a sync, P5 approves.
`validate_table` is used by tests and should be called by every stage that
writes a table, so schema drift fails loudly instead of silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CATEGORIES = (
    "Fuel", "Electricity", "Gas", "Public Transport",
    "Home Improvement", "Automotive", "Utilities", "Other",
)
PRODUCT_CODES = ("GP01", "GP02", "GP03", "GP04")
OPPORTUNITY_BY_PRODUCT = {
    "GP01": "EV_FINANCE",
    "GP02": "GREEN_HOME",
    "GP03": "SOLAR_FINANCE",
    "GP04": "PAPERLESS_BANKING",
}
REGIONS = (
    "North East", "North West", "Yorkshire and The Humber", "East Midlands",
    "West Midlands", "East of England", "London", "South East", "South West",
    "Wales", "Scotland", "Northern Ireland",
)
INCOME_BANDS = ("Q1", "Q2", "Q3", "Q4", "Q5")  # ONS household income quintiles
HOME_STATUS = ("owner_mortgage", "owner_outright", "renter")
PROPERTY_TYPES = ("detached", "semi_detached", "terraced", "bungalow", "flat")  # maisonettes count as flats
URBAN_RURAL = ("urban", "town", "rural")
HOUSEHOLD_SIZES = (1, 2, 3, 4, 5)  # 5 = five or more
VEHICLE_COUNTS = (0, 1, 2)         # 2 = two or more
EPC_RATINGS = ("A", "B", "C", "D", "E", "F", "G")
VEHICLE_TYPES = ("none", "petrol", "diesel", "hybrid", "ev")
HEATING_FUELS = ("gas", "electric", "oil", "other")
STATEMENT_PREFS = ("paper", "digital")
CHANNELS = ("card", "direct_debit", "standing_order", "online_transfer")
# Conflict codes read by src/recommend/eligibility.py (P5). Each suppresses only
# the products it affects; see docs/integration.md.
CONFLICT_FLAGS = (
    "VEHICLE_FUEL_CONFLICT", "VEHICLE_SOLD", "VEHICLE_STATUS_CONFLICT",
    "HOME_STATUS_CONFLICT", "EPC_CONFLICT", "SOLAR_STATUS_CONFLICT", "STATEMENT_PREF_CONFLICT",
)
HELD_PRODUCT_TYPES = (
    "current_account", "mortgage", "vehicle_loan", "personal_loan",
    "credit_card", "savings", "investments",
)

# Hidden persona traits that drive take-up in the generator. They are written to
# data/raw/latent_traits.parquet for diagnostics ONLY and must never become
# features (D1). tests/test_contracts.py enforces this.
LATENT_TRAITS = ("eco_mindedness", "digital_affinity", "financial_headroom", "change_readiness")
# Everything in latent_traits.parquet: the traits plus the persona that set them.
HIDDEN_COLUMNS = (*LATENT_TRAITS, "persona")


@dataclass(frozen=True)
class Col:
    dtype: str  # str | int | float | bool | date | list
    nullable: bool = False
    allowed: tuple | None = None
    description: str = ""


# All profile attributes are a snapshot AS OF feature_cutoff_date (D3).
TABLES: dict[str, dict[str, Col]] = {
    "customers": {
        "customer_id": Col("str"),
        "age": Col("int"),
        "region": Col("str", allowed=REGIONS),
        "urban_rural": Col("str", allowed=URBAN_RURAL),
        "household_size": Col("int", allowed=HOUSEHOLD_SIZES),
        "num_vehicles": Col("int", allowed=VEHICLE_COUNTS, description="household cars; vehicle_type = the main one"),
        "income_band": Col("str", allowed=INCOME_BANDS),
        "home_status": Col("str", nullable=True, allowed=HOME_STATUS),
        "property_type": Col("str", nullable=True, allowed=PROPERTY_TYPES),
        "epc_rating": Col("str", nullable=True, allowed=EPC_RATINGS),
        "main_heating_fuel": Col("str", nullable=True, allowed=HEATING_FUELS),
        "has_solar": Col("bool", nullable=True),
        "vehicle_type": Col("str", nullable=True, allowed=VEHICLE_TYPES),
        "statement_pref": Col("str", nullable=True, allowed=STATEMENT_PREFS),
        "marketing_consent": Col("bool"),
        "data_as_of": Col("date"),
    },
    "products_held": {
        "customer_id": Col("str"),
        "product_type": Col("str", allowed=HELD_PRODUCT_TYPES),
        "open_date": Col("date"),
        "close_date": Col("date", nullable=True, description="set when closed, e.g. car sold"),
        "balance": Col("float"),
    },
    "transactions": {
        "txn_id": Col("str"),
        "customer_id": Col("str"),
        "date": Col("date"),
        "amount": Col("float", description="GBP, positive = money out"),
        "merchant_name": Col("str"),
        "mcc": Col("int", nullable=True),
        "description": Col("str"),
        "channel": Col("str", allowed=CHANNELS),
        "true_category": Col("str", allowed=CATEGORIES, description="FR2 evaluation ONLY"),
    },
    "digital_activity": {
        "customer_id": Col("str"),
        "month": Col("date", description="first day of month"),
        "app_logins": Col("int"),
        "web_sessions": Col("int"),
        "paper_statements_sent": Col("int"),
    },
    "labels": {
        "customer_id": Col("str"),
        "product_code": Col("str", allowed=PRODUCT_CODES),
        "took_up": Col("int", allowed=(0, 1)),
        "label_window_start": Col("date"),
        "label_window_end": Col("date"),
    },
    "splits": {
        "customer_id": Col("str"),
        "split": Col("str", allowed=("train", "val", "test")),
    },
}

# Feature store: one row per customer (docs/data.md "Feature definitions"). Owner: P1.
FEATURES: dict[str, Col] = {
    "customer_id": Col("str"),
    "feature_cutoff_date": Col("date"),
    # profile snapshot (as of cutoff)
    "age": Col("int", description="fairness analysis only; excluded from models (D9)"),
    "region": Col("str", allowed=REGIONS),
    "urban_rural": Col("str", allowed=URBAN_RURAL),
    "household_size": Col("int", allowed=HOUSEHOLD_SIZES),
    "num_vehicles": Col("int", allowed=VEHICLE_COUNTS),
    "income_band": Col("str", allowed=INCOME_BANDS),
    "home_status": Col("str", nullable=True, allowed=HOME_STATUS),
    "property_type": Col("str", nullable=True, allowed=PROPERTY_TYPES),
    "epc_rating": Col("str", nullable=True, allowed=EPC_RATINGS),
    "main_heating_fuel": Col("str", nullable=True, allowed=HEATING_FUELS),
    "has_solar": Col("bool", nullable=True),
    "vehicle_type": Col("str", nullable=True, allowed=VEHICLE_TYPES),
    "statement_pref": Col("str", nullable=True, allowed=STATEMENT_PREFS),
    "marketing_consent": Col("bool"),
    # spend: 12-month monthly average in GBP, by FR2 category
    **{f"spend_{c.lower().replace(' ', '_')}_m": Col("float") for c in CATEGORIES},
    "spend_total_m": Col("float", description="all categories, GBP/month"),
    "fuel_share_of_spend": Col("float", description="spend_fuel_m / spend_total_m"),
    "energy_spend_per_person_m": Col("float", description="(electricity + gas incl. dual-fuel share) / household_size"),
    # recency (D5: changed circumstances)
    "spend_fuel_recent_m": Col("float", description="monthly avg over last `recent_months`"),
    "spend_fuel_prior_m": Col("float", description="monthly avg over the months before that"),
    "months_since_last_fuel_txn": Col("int", nullable=True),
    # derived physical quantities (P2 conversion factors)
    "est_annual_km": Col("float", nullable=True),
    "km_per_vehicle": Col("float", nullable=True, description="est_annual_km / num_vehicles"),
    "est_electricity_kwh_year": Col("float", nullable=True),
    "est_gas_kwh_year": Col("float", nullable=True),
    "energy_from_dual_fuel": Col("bool", description="D6: elec/gas apportioned from a dual-fuel bill"),
    # holdings (open at cutoff)
    "has_mortgage": Col("bool"),
    "has_vehicle_loan": Col("bool"),
    "vehicle_loan_closed_in_window": Col("bool"),
    "has_investments": Col("bool"),
    "has_savings": Col("bool"),
    "has_credit_card": Col("bool"),
    # digital engagement
    "avg_app_logins_m": Col("float"),
    "avg_web_sessions_m": Col("float"),
    "paper_statements_12m": Col("int"),
    # data quality
    "data_completeness": Col("float", description="0-1, share of expected fields/months present"),
    "data_flags": Col("str", description="comma-separated codes from reason_codes.yaml data_flags"),
    # Inputs to P5's eligibility rules (config/products.yaml). conflict_flags must be
    # a list of CONFLICT_FLAGS codes; VEHICLE_FUEL_CONFLICT comes from ROAD-fuel spend,
    # never from EV charging (which is also in the Fuel category).
    "conflict_flags": Col("list", allowed=CONFLICT_FLAGS),
    "ev_replacement_use_case": Col("bool", nullable=True, description="EV owner replacing another vehicle"),
    "ev_purchase_use_case": Col("bool", nullable=True, description="no vehicle today, explicit first-EV need"),
    "green_home_use_case": Col("str", nullable=True,
                               allowed=("insulation", "heat_pump", "efficient_heating", "energy_efficiency_retrofit")),
    "solar_use_case": Col("str", nullable=True, allowed=("new_rooftop_solar_installation",)),
    "requires_paper_statements": Col("bool", nullable=True, description="e.g. accessibility need"),
}

# Columns that must never be model inputs.
NON_MODEL_COLUMNS = (
    "customer_id", "feature_cutoff_date", "data_flags", "conflict_flags",
    "ev_replacement_use_case", "ev_purchase_use_case", "green_home_use_case",
    "solar_use_case", "requires_paper_statements",
)

# outputs/test_predictions.csv (submission item I): one row per test customer x product.
TEST_PREDICTION_COLUMNS = (
    "customer_id", "product_code", "propensity", "confidence",
    "reason_codes", "eligible_to_show", "took_up", "model_version",
)

_DTYPE_CHECKS = {
    "str": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_numeric_dtype,
    "bool": pd.api.types.is_bool_dtype,
    "date": pd.api.types.is_datetime64_any_dtype,
    "list": pd.api.types.is_object_dtype,
}


def validate_table(df: pd.DataFrame, name: str) -> list[str]:
    """Return a list of contract violations for `df` (empty list = valid).

    `name` is a key of TABLES, or "features".
    """
    schema = FEATURES if name == "features" else TABLES[name]
    errors = []
    missing = set(schema) - set(df.columns)
    extra = set(df.columns) - set(schema)
    if missing:
        errors.append(f"{name}: missing columns {sorted(missing)}")
    if extra:
        errors.append(f"{name}: unexpected columns {sorted(extra)}")
    for col, spec in schema.items():
        if col not in df.columns:
            continue
        s = df[col]
        if not spec.nullable and s.isna().any():
            errors.append(f"{name}.{col}: {int(s.isna().sum())} nulls in non-nullable column")
        if not _DTYPE_CHECKS[spec.dtype](s):
            errors.append(f"{name}.{col}: dtype {s.dtype} is not {spec.dtype}")
        if spec.dtype == "list":
            if not s.map(lambda v: isinstance(v, list)).all():
                errors.append(f"{name}.{col}: every value must be a list")
            elif spec.allowed is not None:
                bad = {x for v in s for x in v} - set(spec.allowed)
                if bad:
                    errors.append(f"{name}.{col}: values not allowed {sorted(bad)[:5]}")
        elif spec.allowed is not None:
            bad = set(s.dropna().unique()) - set(spec.allowed)
            if bad:
                errors.append(f"{name}.{col}: values not allowed {sorted(map(str, bad))[:5]}")
    return errors
