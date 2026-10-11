"""Physical quantities derived from categorised spend with P2's conversion factors. Owner: P2.

The feature schema (src/schemas/tables.py) lists four "derived physical quantities
(P2 conversion factors)": est_annual_km, est_electricity_kwh_year, est_gas_kwh_year and
energy_from_dual_fuel. derive_quantities() computes exactly those columns from the
FR2 output, so P1 can merge them into the feature store on customer_id with no schema
change, and the carbon engine reads the same numbers.

Why this lives with P2: FR2 puts EV charging in Fuel (team decision) and dual-fuel
bills in Utilities (D6). Only the categoriser's flags (is_ev_charging, is_dual_fuel)
can undo that for the physical quantities:
- est_annual_km uses ROAD fuel only (Fuel and not EV charging), so an EV owner's
  charging is never read as litres of petrol.
- Electricity and gas kWh include each customer's share of a dual-fuel bill (D6) and
  remove the standing charge before dividing by the unit rate.

Helper columns (not in the feature schema; P1 may use them inside build_features):
  road_fuel_spend_m    GBP/month of road fuel, e.g. for VEHICLE_FUEL_CONFLICT
  ev_charging_spend_m  GBP/month of EV charging
  months_observed      months with any transaction in the feature window
Missing or unusable inputs give None (NaN), never a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.carbon import factors
from src.common import config
from .prices import window_energy_prices

DERIVED_COLUMNS = ("est_annual_km", "est_electricity_kwh_year", "est_gas_kwh_year", "energy_from_dual_fuel")
HELPER_COLUMNS = ("road_fuel_spend_m", "ev_charging_spend_m", "months_observed")

# vehicle_type -> (pump price factor, fuel economy factor). EV / none / unknown have no
# road-fuel baseline, so their mileage stays unknown rather than guessed.
_ROAD_FUEL = {
    "petrol": ("petrol_price_gbp_per_litre", "petrol_car_km_per_litre"),
    "diesel": ("diesel_price_gbp_per_litre", "diesel_car_km_per_litre"),
    "hybrid": ("petrol_price_gbp_per_litre", "hybrid_car_km_per_litre"),
}

ASSUMPTIONS = (
    "Annual quantities use the fixed feature window in P1; standalone observed-month annualisation is explicit.",
    "Mileage = road-fuel spend / mean pump price over the feature window x DESNZ-derived fuel economy; "
    "EV charging is excluded.",
    "kWh = (spend - Ofgem standing charge) / Ofgem price-cap unit rate (direct debit, GB average).",
    "Dual-fuel bills are split between electricity and gas in proportion to the Ofgem medium TDCV cost "
    "of each fuel (D6).",
    "Energy conversion uses day-weighted historical Ofgem tariffs in config/energy_prices.csv "
    "within the feature window; GB direct-debit averages are a proxy, not individual tariffs.",
)


def feature_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    cal = config.settings()["calendar"]
    return pd.Timestamp(cal["feature_window_start"]), pd.Timestamp(cal["feature_cutoff_date"])


def _v(factor_id: str) -> float:
    return factors.get(factor_id).value


def dual_fuel_electricity_share(start=None, cutoff=None) -> float:
    """Share of a dual-fuel bill that is electricity, from Ofgem medium TDCV costs (D6)."""
    prices = window_energy_prices(start, cutoff)
    elec = _v("electricity_tdcv_medium_2023_kwh") * prices['elec_unit'] + prices['elec_standing_year']
    gas = _v("gas_tdcv_medium_2023_kwh") * prices['gas_unit'] + prices['gas_standing_year']
    return elec / (elec + gas)


def _kwh(annual_gbp: pd.Series, fuel: str, prices=None) -> pd.Series:
    prices = prices or window_energy_prices()
    kwh = (annual_gbp - prices[f'{fuel}_standing_year']) / prices[f'{fuel}_unit']
    # no bill, or a bill below the standing charge (partial data): unknown, not zero
    return kwh.where((annual_gbp > 0) & (kwh > 0))


def derive_quantities(categorised: pd.DataFrame, customers: pd.DataFrame,
                      start: pd.Timestamp | None = None, cutoff: pd.Timestamp | None = None,
                      *, annualise_observed: bool = True) -> pd.DataFrame:
    """One row per customer in `customers` with DERIVED_COLUMNS + HELPER_COLUMNS.

    categorised: FR2 output (customer_id, date, amount, category, is_ev_charging, is_dual_fuel).
    customers:   customer_id and vehicle_type, as of the feature cutoff (D3).
    Only transactions inside [start, cutoff] are used (default: settings.calendar).
    """
    default_start, default_cutoff = feature_window()
    start = default_start if start is None else pd.Timestamp(start)
    cutoff = default_cutoff if cutoff is None else pd.Timestamp(cutoff)
    dates = pd.to_datetime(categorised["date"])
    t = categorised.loc[(dates >= start) & (dates <= cutoff)].copy()
    t["date"] = pd.to_datetime(t["date"])

    fuel = t["category"] == "Fuel"
    kinds = {
        "road_fuel": fuel & ~t["is_ev_charging"].astype(bool),
        "ev_charging": fuel & t["is_ev_charging"].astype(bool),
        "electricity": t["category"] == "Electricity",
        "gas": t["category"] == "Gas",
        "dual_fuel": (t["category"] == "Utilities") & t["is_dual_fuel"].astype(bool),
    }
    t["kind"] = np.select(list(kinds.values()), list(kinds.keys()), default="")
    spend = (t[t["kind"] != ""].pivot_table(index="customer_id", columns="kind", values="amount",
                                             aggfunc="sum", fill_value=0.0)
             .reindex(columns=list(kinds), fill_value=0.0))
    months = t.groupby("customer_id")["date"].apply(lambda d: d.dt.to_period("M").nunique())

    out = customers[["customer_id", "vehicle_type"]].drop_duplicates("customer_id").set_index("customer_id")
    out = out.join(spend).join(months.rename("months_observed"))
    out[list(kinds)] = out[list(kinds)].fillna(0.0)
    out["months_observed"] = out["months_observed"].fillna(0).astype(int)
    months_in_window = (cutoff.year - start.year) * 12 + cutoff.month - start.month + 1
    denominator = out['months_observed'] if annualise_observed else months_in_window
    scale = (12 / denominator)
    if annualise_observed:
        scale = scale.where(out['months_observed'] > 0)
    else:
        scale = pd.Series(scale, index=out.index).where(out['months_observed'] > 0)
    annual = out[list(kinds)].mul(scale, axis=0)  # NaN for customers with no data

    km = pd.Series(np.nan, index=out.index)
    for vtype, (price_id, economy_id) in _ROAD_FUEL.items():
        rows = (out["vehicle_type"] == vtype) & (annual["road_fuel"] > 0)
        km[rows] = annual.loc[rows, "road_fuel"] / _v(price_id) * _v(economy_id)

    prices = window_energy_prices(start, cutoff)
    share = dual_fuel_electricity_share(start, cutoff)
    elec_gbp = annual["electricity"] + share * annual["dual_fuel"]
    gas_gbp = annual["gas"] + (1 - share) * annual["dual_fuel"]

    result = pd.DataFrame({
        "customer_id": out.index,
        "est_annual_km": km.to_numpy(),
        "est_electricity_kwh_year": _kwh(elec_gbp, 'elec', prices).to_numpy(),
        "est_gas_kwh_year": _kwh(gas_gbp, 'gas', prices).to_numpy(),
        "energy_from_dual_fuel": (out["dual_fuel"] > 0).to_numpy(),
        "road_fuel_spend_m": (annual["road_fuel"] / 12).to_numpy(),
        "ev_charging_spend_m": (annual["ev_charging"] / 12).to_numpy(),
        "months_observed": out["months_observed"].to_numpy(),
    })
    return result.reset_index(drop=True)
