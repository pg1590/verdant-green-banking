"""Calibration report: generated national totals vs official targets
(config/calibration_targets.yaml). Writes reports/calibration_report.md.

    python scripts/calibration_report.py [--source data/raw]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.carbon import factors  # noqa: E402
from src.categorisation.rules import is_ev_charging  # noqa: E402
from src.common import config  # noqa: E402
from src.data_gen.generate import AGE_BANDS, scenarios, targets  # noqa: E402

SPEND_GROUPS = {  # report row -> (true_category filter, ONS lines)
    "Electricity (incl. dual-fuel share)": (None, ["electricity"]),
    "Gas (incl. dual-fuel share)": (None, ["gas"]),
    "Road fuel": ("Fuel", ["fuel"]),
    "Public transport": ("Public Transport", ["rail", "bus_coach"]),
    "Home improvement": ("Home Improvement", ["home_maintenance"]),
    "Automotive": ("Automotive", ["vehicle_parts", "vehicle_servicing", "vehicle_insurance_or_other_motoring"]),
}


def share_rows(name: str, generated: pd.Series, target: dict) -> list[str]:
    gen = generated.value_counts(normalize=True)
    rows = []
    for key, t in target.items():
        g = float(gen.get(key, gen.get(str(key), 0.0)))
        rows.append(f"| {name} | {key} | {t:.1%} | {g:.1%} | {g - t:+.1%} |")
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, default=config.RAW_DIR)
    p.add_argument("--out", type=Path, default=config.REPORTS_DIR / "calibration_report.md")
    args = p.parse_args(argv)
    tg = targets()
    cust = pd.read_parquet(args.source / "customers.parquet")
    lat = pd.read_parquet(args.source / "latent_traits.parquet")
    tx = pd.read_parquet(args.source / "transactions.parquet")
    scenario_ids = set(scenarios())  # hand-designed edge cases: not part of the population
    cust = cust[~cust["customer_id"].isin(scenario_ids)]
    lat = lat[~lat["customer_id"].isin(scenario_ids)]
    tx = tx[~tx["customer_id"].isin(scenario_ids)]
    cal = config.settings()["calendar"]
    cutoff = pd.Timestamp(cal["feature_cutoff_date"])
    n, n_months = len(cust), 12
    band = pd.cut(cust["age"], [17, 24, 34, 44, 54, 64, 74, 200], labels=list(AGE_BANDS))

    lines = [f"# Calibration report\n\nGenerated from `{args.source.relative_to(config.ROOT) if args.source.is_relative_to(config.ROOT) else args.source}` ({n:,} customers, {len(tx):,} transactions) "
             "vs official targets in `config/calibration_targets.yaml`. Shares are of all customers unless "
             "stated; record fields (with injected missing values) are measured on non-missing rows.\n",
             "## Shares\n", "| Attribute | Value | Target | Generated | Difference |", "| --- | --- | --- | --- | --- |"]
    lines += share_rows("region", cust["region"], tg["region"])
    lines += share_rows("urban_rural", cust["urban_rural"], tg["urban_rural"])
    lines += share_rows("household_size", cust["household_size"], {int(k): v for k, v in tg["household_size"].items()})
    lines += share_rows("age band", band.astype(str), tg["age_band"])
    lines += share_rows("home_status", cust["home_status"].dropna(), tg["home_status"])
    lines += share_rows("income_band", cust["income_band"], tg["income_band"])
    lines += share_rows("property_type", cust["property_type"].dropna(), tg["property_type"])
    lines += share_rows("epc_rating", cust["epc_rating"].dropna(), tg["epc_rating"])
    lines += share_rows("main_heating_fuel", cust["main_heating_fuel"].dropna(), tg["main_heating_fuel"])
    lines += share_rows("num_vehicles", cust["num_vehicles"], {int(k): v for k, v in tg["num_vehicles"].items()})
    cars = cust.loc[cust["num_vehicles"] > 0, "vehicle_type"].dropna()
    lines += share_rows("vehicle_type (car households)", cars[cars != "none"], tg["vehicle_type_of_cars"])
    solar = cust["has_solar"].dropna().astype(bool)
    lines.append(f"| has_solar | true | {tg['has_solar']:.1%} | {solar.mean():.1%} | {solar.mean() - tg['has_solar']:+.1%} |")

    # Spend per household per month over the feature window (true categories).
    win = tx[(tx["date"] <= cutoff)]
    text = (win["merchant_name"] + " " + win["description"]).str.upper()
    dual = (win["true_category"] == "Utilities") & text.str.contains("DUAL FUEL|GAS & ELEC|GAS AND ELEC")
    from src.features.build import dual_fuel_share
    share = dual_fuel_share()
    dual_m = win.loc[dual, "amount"].sum() / n / n_months
    weekly = tg["weekly_spend_gbp"]
    lines += ["\n## Spend per household per month (all households, feature window)\n",
              "| Category | ONS target (GBP) | Generated (GBP) | Ratio |", "| --- | --- | --- | --- |"]
    for label, (cat, ons) in SPEND_GROUPS.items():
        target = sum(weekly[x] for x in ons) * 52 / 12
        if cat is None:
            base = win.loc[win["true_category"] == ("Electricity" if "Electricity" in label else "Gas"), "amount"].sum()
            gen = base / n / n_months + (share if "Electricity" in label else 1 - share) * dual_m
        else:
            mask = win["true_category"] == cat
            if cat == "Fuel":
                mask &= ~is_ev_charging(win)
            gen = win.loc[mask, "amount"].sum() / n / n_months
        lines.append(f"| {label} | {target:,.2f} | {gen:,.2f} | {gen / target:.2f} |")

    # Mileage check: road-fuel spend -> litres -> miles per car (car households).
    road = win[(win["true_category"] == "Fuel") & ~is_ev_charging(win)]
    per_cust = road.groupby("customer_id")["amount"].sum()
    ice = cust[cust["vehicle_type"].isin(["petrol", "diesel", "hybrid"]) & (cust["num_vehicles"] > 0)]
    litres = per_cust.reindex(ice["customer_id"]).fillna(0).to_numpy() / factors.get("petrol_price_gbp_per_litre").value
    miles = litres * factors.get("petrol_car_km_per_litre").value / 1.609 / ice["num_vehicles"].to_numpy()
    lines += ["\n## Mileage check\n",
              f"Mean implied miles per car (combustion-car households, from road-fuel spend): "
              f"**{pd.Series(miles).mean():,.0f}** (median {pd.Series(miles).median():,.0f}) vs NTS0901 average "
              f"**{tg['annual_miles_per_car']:,}** "
              "(spend is calibrated to ONS, so this is a consistency check, not a target)."]

    lines += ["\n## Personas\n", "| Persona | Share |", "| --- | --- |"]
    lines += [f"| {k} | {v:.1%} |" for k, v in lat["persona"].value_counts(normalize=True).items()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
