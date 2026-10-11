"""FR1 Customer 360 + feature store. Owner: P1.

v0: works on the mock sample (data/sample) or P1's generated data (data/raw)
and writes data/processed/features.parquet, one row per customer, matching
src.schemas.tables.FEATURES. Definitions are in docs/data.md ("Feature
definitions"); thresholds are in config/settings.yaml `features:`.

Leakage rules (D3, D5):
- Only transactions/activity dated in [feature_window_start, feature_cutoff_date].
- products_held: held if open_date <= cutoff and (close_date null or > cutoff).
- Never reads latent_traits, labels or true_category.
- Spend features use PREDICTED categories (FR2), so categoriser errors propagate
  honestly. Until P2's model exists, the rules baseline is used.

P5 eligibility inputs (docs/integration.md): data_completeness, conflict_flags
(VEHICLE_FUEL_CONFLICT comes from ROAD fuel only, never EV charging) and the
optional use-case flags, which v0 leaves null (no generation rule agreed yet).

Dual fuel (D6): Utilities rows matching categories.yaml dual_fuel keywords are
split into electricity and gas spend in proportion to the expected bill at Ofgem
medium TDCV (unit rate x kWh + standing charge x 365); energy_from_dual_fuel is
set and ENERGY_APPORTIONED_FROM_DUAL_FUEL added to data_flags.

    python -m src.features.build [--source data/sample] [--out data/processed/features.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.carbon import factors
from src.carbon.conversions import derive_quantities, dual_fuel_electricity_share
from src.carbon.prices import window_energy_prices
from src.categorisation.rules import categorise_rules, is_ev_charging
from src.common import config
from src.schemas import tables as T

PROFILE_FIELDS = ("home_status", "property_type", "epc_rating", "main_heating_fuel",
                  "has_solar", "vehicle_type", "statement_pref")
RAW_TABLES = ("customers", "products_held", "transactions", "digital_activity")
TXN_COLUMNS = ("customer_id", "date", "amount")
CATEGORY_COLUMNS = ("category", "is_ev_charging", "is_dual_fuel")


def load_tables(source: Path, txn_columns: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    """Raw tables. `txn_columns` limits the transaction columns read: the text columns and
    txn_id are most of the memory on large runs, and the features only need TXN_COLUMNS."""
    return {name: pd.read_parquet(source / f"{name}.parquet",
                                  columns=list(txn_columns) if txn_columns and name == "transactions" else None)
            for name in RAW_TABLES}


def read_categorised(path: Path) -> pd.DataFrame:
    """P2's predictions indexed by txn_id, reading only the columns the features use
    (the file repeats every transaction column)."""
    return pd.read_parquet(path, columns=['txn_id', *CATEGORY_COLUMNS]).set_index('txn_id')


def read_categories(source: Path, categorised: Path) -> pd.DataFrame:
    """P2's CATEGORY_COLUMNS for each row of source/transactions.parquet, in that order.

    P2 writes the raw rows in order, so the IDs are compared as Arrow arrays: no Python
    strings, sets or string index for millions of IDs. A file in another order is still
    accepted and realigned by txn_id.
    """
    raw_ids = pq.read_table(source / "transactions.parquet", columns=["txn_id"])["txn_id"]
    if pq.read_table(categorised, columns=["txn_id"])["txn_id"].equals(raw_ids):
        if pc.count_distinct(raw_ids, mode="all").as_py() != len(raw_ids):
            raise ValueError('Categorised transaction IDs must match the raw source exactly')
        return pd.read_parquet(categorised, columns=list(CATEGORY_COLUMNS))
    by_id = read_categorised(categorised)
    ids = raw_ids.to_pandas()
    if by_id.index.has_duplicates or set(by_id.index) != set(ids):
        raise ValueError('Categorised transaction IDs must match the raw source exactly')
    return by_id.loc[ids].reset_index(drop=True)


def default_source() -> Path:
    raw = config.RAW_DIR / "customers.parquet"
    return config.RAW_DIR if raw.exists() else config.SAMPLE_DIR


def _f(factor_id: str) -> float:
    return factors.get(factor_id).value


def _monthly(txns: pd.DataFrame, mask: pd.Series, ids: pd.Index, months: int) -> pd.Series:
    return (txns.loc[mask].groupby("customer_id")["amount"].sum() / months).reindex(ids, fill_value=0.0)


def dual_fuel_share(start=None, cutoff=None) -> float:
    """Electricity's share of a dual-fuel bill: expected bill at the pre-July-2026 Ofgem
    medium consumption (2,700 / 11,500 kWh) at feature-window prices."""
    return dual_fuel_electricity_share(start, cutoff)


def build_features(tables: dict[str, pd.DataFrame], categories: pd.Series | pd.DataFrame,
                   cutoff: pd.Timestamp | None = None, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """`categories` is one predicted FR2 category per row of tables["transactions"].

    `cutoff`/`start` default to the canonical window; trend snapshots pass their own
    12-month window (src/features/snapshots.py). Nothing after `cutoff` is read.
    """
    s = config.settings()
    cal, thr = s["calendar"], s["features"]
    cutoff = pd.Timestamp(cutoff or cal["feature_cutoff_date"])
    start = pd.Timestamp(start or cal["feature_window_start"])
    n_months = (cutoff.year - start.year) * 12 + cutoff.month - start.month + 1
    recent_start = (cutoff + pd.Timedelta(1, unit='D')) - pd.DateOffset(months=cal["recent_months"])
    n_recent, n_prior = cal["recent_months"], n_months - cal["recent_months"]

    cust = tables["customers"].set_index("customer_id")
    ids = cust.index
    # Only these transaction columns are used below; copying the text columns (merchant,
    # description, channel) doubled peak memory on large runs. Rows outside the window are
    # dropped before the categories are added, so they are never copied.
    txns = tables['transactions']
    in_window = ((txns["date"] >= start) & (txns["date"] <= cutoff)).to_numpy()
    base = txns.loc[in_window, list(TXN_COLUMNS)]
    if isinstance(categories, pd.DataFrame):
        t = base.assign(**{k: categories[k].to_numpy()[in_window] for k in CATEGORY_COLUMNS})
    else:
        from src.categorisation.rules import explain_rules
        metadata = explain_rules(txns)
        t = base.assign(category=categories.to_numpy()[in_window],
            is_ev_charging=metadata['is_ev_charging'].to_numpy()[in_window],
            is_dual_fuel=metadata['is_dual_fuel'].to_numpy()[in_window])
    out = pd.DataFrame(index=ids)
    out["feature_cutoff_date"] = cutoff

    for col in ("age", "region", "urban_rural", "household_size", "num_vehicles", "income_band",
                *PROFILE_FIELDS, "marketing_consent"):
        out[col] = cust[col]

    # Spend by FR2 category, GBP per month over the window.
    for cat in T.CATEGORIES:
        out[f"spend_{cat.lower().replace(' ', '_')}_m"] = _monthly(t, t["category"] == cat, ids, n_months)

    out["spend_total_m"] = _monthly(t, t["amount"].notna(), ids, n_months)
    out["fuel_share_of_spend"] = (out["spend_fuel_m"] / out["spend_total_m"]).where(out["spend_total_m"] > 0, 0.0)
    fuel = t["category"] == "Fuel"
    road = fuel & ~t['is_ev_charging']
    recent = t["date"] >= recent_start
    out["spend_fuel_recent_m"] = _monthly(t, fuel & recent, ids, n_recent)
    out["spend_fuel_prior_m"] = _monthly(t, fuel & ~recent, ids, n_prior)
    last_fuel = t.loc[fuel].groupby("customer_id")["date"].max().reindex(ids)
    out["months_since_last_fuel_txn"] = ((cutoff - last_fuel).dt.days // 30).astype("Int64")
    road_m = _monthly(t, road, ids, n_months)
    road_recent_m = _monthly(t, road & recent, ids, n_recent)
    road_prior_m = _monthly(t, road & ~recent, ids, n_prior)

    # km/yr from ROAD fuel for petrol/diesel/hybrid; unknown for EV, none or missing.
    vt = out["vehicle_type"]
    # Energy: single-fuel bills plus the apportioned share of dual-fuel bills (D6).
    dual = (t["category"] == "Utilities") & t['is_dual_fuel']
    dual_annual = _monthly(t, dual, ids, n_months) * 12
    share = dual_fuel_share(start, cutoff)
    elec_annual = out["spend_electricity_m"] * 12 + dual_annual * share
    gas_annual = out["spend_gas_m"] * 12 + dual_annual * (1 - share)
    no_gas_expected = out["main_heating_fuel"].isin(["electric", "oil", "other"])
    out["energy_spend_per_person_m"] = (elec_annual + gas_annual) / 12 / out["household_size"]
    physical = derive_quantities(t, tables['customers'], start=start, cutoff=cutoff,
                                 annualise_observed=False).set_index('customer_id').reindex(ids)
    for field in ('est_annual_km', 'est_electricity_kwh_year', 'energy_from_dual_fuel'):
        out[field] = physical[field]
    out['est_gas_kwh_year'] = physical['est_gas_kwh_year'].where(
        physical['est_gas_kwh_year'].notna(), np.where(no_gas_expected, 0.0, np.nan))
    out['km_per_vehicle'] = (out['est_annual_km'] / out['num_vehicles']).where(out['num_vehicles'] > 0)
    road_m = physical['road_fuel_spend_m']

    # Holdings open at the cutoff.
    ph = tables["products_held"]
    open_at = ph[(ph["open_date"] <= cutoff) & (ph["close_date"].isna() | (ph["close_date"] > cutoff))]
    held = open_at.groupby("customer_id")["product_type"].agg(set).reindex(ids)
    held = held.apply(lambda v: v if isinstance(v, set) else set())
    for col, ptype in [("has_mortgage", "mortgage"), ("has_vehicle_loan", "vehicle_loan"),
                       ("has_investments", "investments"), ("has_savings", "savings"),
                       ("has_credit_card", "credit_card")]:
        out[col] = held.apply(lambda v, p=ptype: p in v).astype(bool)
    closed = ph[(ph["product_type"] == "vehicle_loan") & ph["close_date"].between(start, cutoff)]
    out["vehicle_loan_closed_in_window"] = ids.isin(closed["customer_id"])

    # Digital engagement over the window.
    da = tables["digital_activity"]
    da = da[(da["month"] >= start) & (da["month"] <= cutoff)].groupby("customer_id")
    out["avg_app_logins_m"] = da["app_logins"].mean().reindex(ids, fill_value=0.0)
    out["avg_web_sessions_m"] = da["web_sessions"].mean().reindex(ids, fill_value=0.0)
    out["paper_statements_12m"] = da["paper_statements_sent"].sum().reindex(ids, fill_value=0).astype(int)
    da_recent = tables["digital_activity"]
    da_recent = da_recent[(da_recent["month"] >= recent_start) & (da_recent["month"] <= cutoff)]
    paper_recent = da_recent.groupby("customer_id")["paper_statements_sent"].sum().reindex(ids, fill_value=0)

    # Data completeness: 7 profile fields + share of window months with any transaction.
    months_seen = t.assign(m=t["date"].dt.to_period("M")).groupby("customer_id")["m"].nunique().reindex(ids, fill_value=0)
    present = pd.concat([out[f].notna() for f in PROFILE_FIELDS], axis=1).sum(axis=1)
    out["data_completeness"] = ((present + months_seen / n_months) / (len(PROFILE_FIELDS) + 1)).round(4)

    # Conflict flags (P5 codes) and data-quality flags.
    ice = vt.isin(["petrol", "diesel", "hybrid"])
    stopped = (road_prior_m >= thr["road_fuel_conflict_gbp_m"]) & (road_recent_m < thr["fuel_stopped_gbp_m"])
    conflicts = {
        # An EV household with a second car legitimately buys road fuel.
        "VEHICLE_FUEL_CONFLICT": (vt == "ev") & (out["num_vehicles"] <= 1) & (road_m >= thr["road_fuel_conflict_gbp_m"]),
        "VEHICLE_SOLD": (vt == "none") & (stopped | out["vehicle_loan_closed_in_window"]),
        "VEHICLE_STATUS_CONFLICT": (ice & stopped) | ((vt == "none") & (road_recent_m >= thr["road_fuel_conflict_gbp_m"])),
        "HOME_STATUS_CONFLICT": (out["home_status"] == "renter") & out["has_mortgage"],
        "STATEMENT_PREF_CONFLICT": (out["statement_pref"] == "digital") & (paper_recent > 0),
    }
    out["conflict_flags"] = [[c for c, m in conflicts.items() if m[cid]] for cid in ids]
    fuel_prior = out["spend_fuel_prior_m"]
    data_flags = {
        "LOW_DATA_COMPLETENESS": out["data_completeness"] < thr["low_completeness"],
        "CONFLICTING_DATA": out["conflict_flags"].str.len() > 0,
        "RECENT_CIRCUMSTANCE_CHANGE": (fuel_prior >= thr["road_fuel_conflict_gbp_m"])
                                      & (out["spend_fuel_recent_m"] < thr["circumstance_drop_ratio"] * fuel_prior),
        "ENERGY_APPORTIONED_FROM_DUAL_FUEL": out["energy_from_dual_fuel"],
    }
    out["data_flags"] = [",".join(f for f, m in data_flags.items() if m[cid]) for cid in ids]

    # Optional use-case flags: no agreed generation rule yet, so unknown.
    for col in ("ev_replacement_use_case", "ev_purchase_use_case", "requires_paper_statements"):
        out[col] = pd.array([pd.NA] * len(ids), dtype="boolean")
    for col in ("green_home_use_case", "solar_use_case"):
        out[col] = pd.Series([None] * len(ids), index=ids, dtype=object)

    out = out.reset_index().rename(columns={"index": "customer_id"})
    out = out[list(T.FEATURES)]
    errors = T.validate_table(out, "features")
    if errors:
        raise ValueError("feature table violates contract:\n" + "\n".join(errors))
    return out


CUSTOMER_360_COLUMNS = ("products_held", "n_transactions_window", "active_months_window",
                        "last_transaction_date")


def customer_360(tables: dict[str, pd.DataFrame], features: pd.DataFrame) -> pd.DataFrame:
    """FR1: one view per customer = feature row + open products + activity counts.

    Counts cover the feature window [feature_window_start, cutoff] only. These display
    fields are not model inputs (models use the explicit lists in config/model.yaml).
    """
    cutoff = pd.Timestamp(features["feature_cutoff_date"].iloc[0])
    start = pd.Timestamp(config.settings()["calendar"]["feature_window_start"])
    ph = tables["products_held"]
    open_at = ph[(ph["open_date"] <= cutoff) & (ph["close_date"].isna() | (ph["close_date"] > cutoff))]
    products = open_at.groupby("customer_id")["product_type"].agg(lambda s: sorted(set(s)))
    txns = tables["transactions"]
    txns = txns[(txns["date"] >= start) & (txns["date"] <= cutoff)]
    by_customer = txns.groupby("customer_id")
    view = features.set_index("customer_id")
    view["products_held"] = products.reindex(view.index).apply(lambda v: v if isinstance(v, list) else [])
    view["n_transactions_window"] = by_customer.size().reindex(view.index, fill_value=0).astype(int)
    view["active_months_window"] = (by_customer["date"].apply(lambda d: d.dt.to_period("M").nunique())
                                    .reindex(view.index, fill_value=0).astype(int))
    last = by_customer["date"].max().reindex(view.index)
    view["last_transaction_date"] = last.dt.strftime("%Y-%m-%d").where(last.notna(), None)
    return view.reset_index()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build the feature store (FR1).")
    p.add_argument("--source", type=Path, default=None, help="folder of raw parquet tables")
    p.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "features.parquet")
    p.add_argument('--categorised', type=Path, help='P2 predictions for this exact raw transaction table')
    args = p.parse_args(argv)
    source = args.source or default_source()
    if args.categorised:
        categories = read_categories(source, args.categorised)
        tables = load_tables(source, TXN_COLUMNS)
    else:
        # the rules read the text columns, and every column is written out below
        tables = load_tables(source)
        categories = categorise_rules(tables['transactions'])
    features = build_features(tables, categories)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    # FR1 Customer 360: feature snapshot + holdings + activity, one row per customer.
    customer_360(tables, features).to_parquet(args.out.parent / "customer_360.parquet", index=False)
    if not args.categorised:
        tables['transactions'].drop(columns=['true_category'], errors='ignore').assign(
            category=categories.to_numpy()).to_parquet(args.out.parent / 'transactions_categorised.parquet', index=False)
    flags = features["conflict_flags"].explode().value_counts().to_dict()
    print(f"features: {len(features)} customers x {features.shape[1]} columns from {source} -> {args.out}")
    print(f"conflict flags: {flags or 'none'}")


if __name__ == "__main__":
    main()
