"""Take-up label generation (rules in config/label_rules.yaml). Owner: P1.

Used by the mock generator now and by P1's real generator later: both pass the
TRUE customer state at the cutoff (`truth`), the hidden traits and a seed.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

from src.common import config
from src.common.records import to_records
from src.recommend import check_eligibility, load_catalogue

# truth columns: customer_id, age, urban_rural, household_size, num_vehicles, income_band,
# home_status, property_type, epc_rating, main_heating_fuel, has_solar, vehicle_type,
# statement_pref, marketing_consent, has_vehicle_loan, has_mortgage, fuel_spend_m,
# fuel_per_vehicle_m, heating_spend_m, electricity_spend_m, electricity_per_person_m,
# energy_spend_per_person_m, app_logins_m  (spend in GBP/month over the feature window)


def _standardise(s: pd.Series, within: np.ndarray) -> pd.Series:
    """z-score using the ELIGIBLE customers' mean/sd, so a driver's coefficient means
    the same thing in every product (e.g. fuel spend among car owners, not everyone)."""
    s = s.astype(float)
    ref = s[within]
    sd = ref.std()
    return (s - ref.mean()) / sd if sd > 0 else s * 0.0


# Raw driver values; continuous ones (more than two distinct values) are standardised
# within each product's eligible population in generate_labels().
DRIVERS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "log_fuel_spend": lambda d: np.log1p(d["fuel_spend_m"]),
    "log_heating_spend": lambda d: np.log1p(d["heating_spend_m"]),
    "log_electricity_spend": lambda d: np.log1p(d["electricity_spend_m"]),
    "log_app_logins": lambda d: np.log1p(d["app_logins_m"]),
    "income": lambda d: d["income_band"].str[1].astype(int),
    "has_vehicle_loan": lambda d: d["has_vehicle_loan"].astype(float),
    "has_mortgage": lambda d: d["has_mortgage"].astype(float),
    "diesel_or_petrol": lambda d: d["vehicle_type"].isin(["petrol", "diesel"]).astype(float),
    "epc_poor": lambda d: d["epc_rating"].isin(["E", "F", "G"]).astype(float),
    "gas_or_oil_heating": lambda d: d["main_heating_fuel"].isin(["gas", "oil"]).astype(float),
    "detached_or_semi": lambda d: d["property_type"].isin(["detached", "semi_detached"]).astype(float),
    "age_65_plus": lambda d: (d["age"] >= 65).astype(float),
    "younger": lambda d: -d["age"].astype(float),
    "log_fuel_per_vehicle": lambda d: np.log1p(d["fuel_per_vehicle_m"]),
    "log_energy_per_person": lambda d: np.log1p(d["energy_spend_per_person_m"]),
    "log_electricity_per_person": lambda d: np.log1p(d["electricity_per_person_m"]),
    "two_plus_cars": lambda d: (d["num_vehicles"] >= 2).astype(float),
    "urban": lambda d: (d["urban_rural"] == "urban").astype(float),
    "not_urban": lambda d: (d["urban_rural"] != "urban").astype(float),
}


@lru_cache
def label_rules() -> dict:
    with open(config.CONFIG_DIR / "label_rules.yaml") as f:
        return yaml.safe_load(f)


def _driver(name: str, truth: pd.DataFrame, latent: pd.DataFrame, eligible: np.ndarray) -> pd.Series:
    if name.startswith("latent_"):
        return latent[name.removeprefix("latent_")].astype(float)  # already N(0, 1)
    raw = DRIVERS[name](truth).astype(float)
    return _standardise(raw, eligible) if raw.nunique() > 2 else raw


def _solve_intercept(linear: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + linear)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_labels(truth: pd.DataFrame, latent: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Return (labels table, diagnostics {product: {intercept, eligible, rate}})."""
    rules = label_rules()
    cal = config.settings()["calendar"]
    rng = np.random.default_rng(seed + rules["seed_offset"])
    truth = truth.set_index("customer_id")
    latent = latent.set_index("customer_id").reindex(truth.index)
    snapshots = to_records(truth.reset_index().assign(conflict_flags=[[] for _ in range(len(truth))]))
    catalogue = load_catalogue()  # once: check_eligibility re-reads the YAML when not given one
    frames, diag = [], {}
    for code, spec in rules["products"].items():
        eligible = np.array([check_eligibility(c, code, catalogue, purpose="training").eligible_to_show
                             for c in snapshots])
        linear = sum(coef * _driver(name, truth, latent, eligible)
                     for name, coef in spec["coefs"].items()).fillna(0).to_numpy()
        linear = linear + rng.normal(0, spec["noise_sd"], len(truth))
        intercept = _solve_intercept(linear[eligible], spec["target_rate"]) if eligible.any() else 0.0
        p = 1 / (1 + np.exp(-(intercept + linear)))
        took_up = ((rng.random(len(truth)) < p) & eligible).astype(int)
        diag[code] = {"intercept": round(intercept, 3), "eligible": int(eligible.sum()),
                      "rate_among_eligible": round(float(took_up[eligible].mean()), 4) if eligible.any() else None}
        frames.append(pd.DataFrame({
            "customer_id": truth.index, "product_code": code, "took_up": took_up,
            "label_window_start": pd.Timestamp(cal["label_window_start"]),
            "label_window_end": pd.Timestamp(cal["label_window_end"]),
        }))
    return pd.concat(frames, ignore_index=True), diag
