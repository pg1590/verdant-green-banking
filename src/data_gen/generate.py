"""Synthetic data generator v1. Owner: P1.

Writes the raw tables (src/schemas/tables.py) for N customers, seeded and
reproducible. Design (schema doc + docs/data.md):

1. Personas (config/personas.yaml) give relative weights per attribute; those
   weights are raked (iterative proportional fitting) so the NATIONAL mix matches
   config/calibration_targets.yaml exactly while each persona keeps its tilt.
   Tenure comes from the real EHS tenure-by-age table, region from national shares.
2. Hidden traits = persona mean + individual noise, partly linked to income, age
   and urban living, then standardised. Never features (latent_traits.parquet).
3. Spending depends on household size, property, heating, EPC, cars and area;
   national per-category means are then scaled to ONS Family Spending.
4. Record errors are injected AFTER the truth snapshot (sold car, EV record with a
   petrol car, missing fields); labels use the truth (src/data_gen/labels.py).

    python -m src.data_gen.generate [--n 10000] [--out data/raw]
"""
from __future__ import annotations

import argparse
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common import config
from src.data_gen.labels import generate_labels
from src.data_gen.splits import split_customers
from src.schemas import tables as T

AGE_BANDS = {"18-24": (18, 24), "25-34": (25, 34), "35-44": (35, 44), "45-54": (45, 54),
             "55-64": (55, 64), "65-74": (65, 74), "75+": (75, 90)}
TOWNS = ["LEEDS", "MANCHESTER", "BRISTOL", "CARDIFF", "GLASGOW", "BIRMINGHAM", "LONDON", "NEWCASTLE",
         "NOTTINGHAM", "SHEFFIELD", "BELFAST", "NORWICH", "EXETER", "YORK", "READING", "LIVERPOOL"]
ICE = ("petrol", "diesel", "hybrid")

# group -> list of (merchant template, mcc or None, true_category, channel, description)
MERCHANTS = {
    "fuel": [("SHELL {n} {town}", 5541, "Fuel", "card", "CARD PAYMENT"),
             ("BP CONNECT {town}", 5541, "Fuel", "card", "CONTACTLESS"),
             ("ESSO {town}", 5542, "Fuel", "card", "CARD PAYMENT"),
             ("TESCO PFS {n}", None, "Fuel", "card", "CARD PAYMENT"),
             ("SAINSBURYS PETROL {town}", 5542, "Fuel", "card", "CONTACTLESS"),
             ("TEXACO {town}", 5541, "Fuel", "card", "CARD PAYMENT")],
    "fuel_shop": [("SHELL {n} {town}", 5541, "Other", "card", "SHOP PURCHASE"),
                  ("BP CONNECT {town}", 5541, "Other", "card", "SHOP PURCHASE")],
    "ev_charge": [("POD POINT CHARGE", 5552, "Fuel", "card", "CONTACTLESS"),
                  ("OCTOPUS EV CHARGE", None, "Fuel", "card", "APP PAYMENT"),
                  ("IONITY {town}", 5552, "Fuel", "card", "APP PAYMENT"),
                  ("GRIDSERVE {town}", 5552, "Fuel", "card", "CONTACTLESS")],
    "electricity": [("OCTOPUS ENERGY ELEC", 4900, "Electricity", "direct_debit", "DD ELECTRICITY"),
                    ("EDF ENERGY ELECTRIC", 4900, "Electricity", "direct_debit", "DIRECT DEBIT"),
                    ("E.ON NEXT ELECTRICITY", 4900, "Electricity", "direct_debit", "DIRECT DEBIT")],
    "gas": [("BRITISH GAS", 4900, "Gas", "direct_debit", "DD GAS ACCOUNT"),
            ("SSE GAS SUPPLY", 4900, "Gas", "direct_debit", "DIRECT DEBIT")],
    "dual_fuel": [("EDF ENERGY DUAL FUEL", 4900, "Utilities", "direct_debit", "DD GAS & ELEC"),
                  ("OVO ENERGY", 4900, "Utilities", "direct_debit", "DUAL FUEL DD"),
                  ("OCTOPUS ENERGY", 4900, "Utilities", "direct_debit", "GAS AND ELEC DD")],
    "heating_oil": [("BOILERJUICE HEATING OIL", None, "Utilities", "card", "HEATING OIL"),
                    ("CERTAS ENERGY", None, "Utilities", "card", "HEATING OIL DELIVERY")],
    "utilities": [("THAMES WATER", 4900, "Utilities", "direct_debit", "DIRECT DEBIT"),
                  ("BT BROADBAND", 4814, "Utilities", "direct_debit", "DIRECT DEBIT"),
                  ("{town} CITY COUNCIL", 9311, "Utilities", "direct_debit", "COUNCIL TAX"),
                  ("VIRGIN MEDIA", 4899, "Utilities", "direct_debit", "DIRECT DEBIT")],
    "public_transport": [("TFL TRAVEL CH", 4111, "Public Transport", "card", "CONTACTLESS"),
                         ("TRAINLINE.COM", 4112, "Public Transport", "card", "ONLINE"),
                         ("STAGECOACH BUS", 4131, "Public Transport", "card", "CONTACTLESS"),
                         ("NATIONAL EXPRESS", 4131, "Public Transport", "card", "ONLINE")],
    "home_improvement": [("B&Q {town}", 5200, "Home Improvement", "card", "CARD PAYMENT"),
                         ("WICKES {town}", 5211, "Home Improvement", "card", "CARD PAYMENT"),
                         ("SCREWFIX DIRECT", None, "Home Improvement", "card", "ONLINE"),
                         ("TOOLSTATION {town}", None, "Home Improvement", "card", "CARD PAYMENT")],
    "automotive": [("HALFORDS {town}", 5533, "Automotive", "card", "CARD PAYMENT"),
                   ("KWIK FIT {town}", 7538, "Automotive", "card", "CARD PAYMENT"),
                   ("CAR INSURANCE DD", None, "Automotive", "direct_debit", "DIRECT DEBIT"),
                   ("NCP PARKING {town}", 7523, "Automotive", "card", "CONTACTLESS")],
    "other": [("TESCO STORES {n}", 5411, "Other", "card", "CONTACTLESS"),
              ("SAINSBURYS {town}", 5411, "Other", "card", "CONTACTLESS"),
              ("AMAZON UK MARKETPLACE", 5999, "Other", "card", "ONLINE"),
              ("UBER *TRIP", 4121, "Other", "card", "APP PAYMENT"),
              ("PRET A MANGER {town}", 5814, "Other", "card", "CONTACTLESS"),
              ("BOOTS {town}", 5912, "Other", "card", "CARD PAYMENT")],
}
# group -> (mean GBP per txn, txns per month, what the customer multiplier scales)
BASE_RATES = {
    "fuel": (55, 2.5, "count"), "fuel_shop": (6, 0.8, "count"), "ev_charge": (12, 3, "count"),
    "electricity": (75, 1, "amount"), "gas": (85, 1, "amount"), "dual_fuel": (160, 1, "amount"),
    "heating_oil": (300, 0.3, "count"), "utilities": (60, 3, "amount"),
    "public_transport": (8, 4, "count"), "home_improvement": (45, 0.5, "count"),
    "automotive": (40, 1.3, "count"), "other": (25, 18, "count"),
}
# ONS weekly spend line(s) each generated group is scaled to (national mean per household).
ONS_LINES = {"fuel": ["fuel"], "public_transport": ["rail", "bus_coach"],
             "utilities": ["water_and_dwelling_services", "telecoms", "council_tax"],
             "home_improvement": ["home_maintenance"],
             "automotive": ["vehicle_parts", "vehicle_servicing", "vehicle_insurance_or_other_motoring"]}

# Used only while config/calibration_targets.yaml still has nulls; flagged loudly.
PLACEHOLDERS = {
    "num_vehicles": {0: 0.22, 1: 0.43, 2: 0.35},
    "urban_rural": {"urban": 0.35, "town": 0.47, "rural": 0.18},
    "vehicle_type_of_cars": {"petrol": 0.58, "diesel": 0.30, "hybrid": 0.07, "ev": 0.05},
}


@lru_cache
def _yaml(name: str) -> dict:
    with open(config.CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def scenarios() -> dict[str, dict]:
    """config/scenarios.yaml with the `base` block merged into every scenario."""
    raw = _yaml("scenarios.yaml")
    base = raw["base"]
    out = {}
    for sid, spec in raw["scenarios"].items():
        out[sid] = {**spec, "truth": {**base["truth"], **spec.get("truth", {})},
                    "behaviour": {**base["behaviour"], **spec.get("behaviour", {})},
                    "record": spec.get("record", {})}
    return out


def targets() -> dict:
    t = dict(_yaml("calibration_targets.yaml"))
    for key, value in PLACEHOLDERS.items():
        if t.get(key) is None:
            warnings.warn(f"calibration target {key!r} missing: using PLACEHOLDER values", stacklevel=2)
            t[key] = value
    return t


def rake(shares: np.ndarray, weights: np.ndarray, target: np.ndarray, iters: int = 300) -> np.ndarray:
    """Iterative proportional fitting: row-stochastic P (persona x category) with
    shares @ P == target, starting from the personas' relative weights."""
    P = weights / weights.sum(1, keepdims=True)
    for _ in range(iters):
        implied = shares @ P
        P = P * np.divide(target, implied, out=np.ones_like(target), where=implied > 0)
        P = P / P.sum(1, keepdims=True)
    return P


def _draw(rng, P: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """One categorical draw per customer from row `rows[i]` of P."""
    cum = P[rows].cumsum(1)
    return (rng.random(len(rows))[:, None] > cum).sum(1).clip(max=P.shape[1] - 1)


def _attribute(rng, personas: dict, idx: np.ndarray, key: str, cats: list, target: dict,
               multiplier_key: str | None = None, floor: float = 0.05) -> np.ndarray:
    """Draw one attribute: persona weights (or national x persona multiplier), raked to target."""
    names = list(personas)
    shares = np.bincount(idx, minlength=len(names)) / len(idx)
    tvec = np.array([target[c] for c in cats], dtype=float)
    if multiplier_key:
        W = np.array([[tvec[j] * personas[p].get(multiplier_key, {}).get(c, 1.0) for j, c in enumerate(cats)]
                      for p in names])
    else:
        W = np.array([[personas[p][key].get(c, 0.0) for c in cats] for p in names]) + floor
    return np.array(cats, dtype=object)[_draw(rng, rake(shares, W, tvec), idx)]


def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() or 1.0)


def generate(n_customers: int, seed: int, with_scenarios: bool = True) -> dict[str, pd.DataFrame]:
    """`n_customers` normal customers plus the fixed scenario customers
    (config/scenarios.yaml), which get no labels and no split."""
    rng = np.random.default_rng(seed)
    cal = config.settings()["calendar"]
    cutoff = pd.Timestamp(cal["feature_cutoff_date"])
    months = pd.date_range(cal["data_start"], cal["data_end"], freq="MS")
    n_feat = int((months <= cutoff).sum())
    tg = targets()
    pc = _yaml("personas.yaml")
    personas = pc["personas"]
    names = list(personas)
    sc = scenarios() if with_scenarios else {}
    n_normal = n_customers
    n = n_normal + len(sc)
    ids = [f"C{10001 + i}" for i in range(n_normal)] + list(sc)
    sc_rows = {n_normal + j: spec for j, spec in enumerate(sc.values())}  # row index -> scenario

    # 1. Persona, then persona-tilted attributes raked to national targets.
    pidx = rng.choice(len(names), n, p=np.array([personas[p]["share"] for p in names]) / sum(
        personas[p]["share"] for p in names))
    persona = np.array(names, dtype=object)[pidx]
    band = _attribute(rng, personas, pidx, "age_band", list(AGE_BANDS), tg["age_band"])
    age = np.array([rng.integers(AGE_BANDS[b][0], AGE_BANDS[b][1] + 1) for b in band])
    hh_size = _attribute(rng, personas, pidx, "household_size", [1, 2, 3, 4, 5],
                         {int(k): v for k, v in tg["household_size"].items()}).astype(int)
    n_veh = _attribute(rng, personas, pidx, "num_vehicles", [0, 1, 2],
                       {int(k): v for k, v in tg["num_vehicles"].items()}).astype(int)
    urban = _attribute(rng, personas, pidx, "urban_rural", list(T.URBAN_RURAL), tg["urban_rural"])
    prop = _attribute(rng, personas, pidx, "property_type", list(T.PROPERTY_TYPES), tg["property_type"])
    income = _attribute(rng, personas, pidx, "income_band", list(T.INCOME_BANDS), tg["income_band"])
    heating = _attribute(rng, personas, pidx, None, list(T.HEATING_FUELS), tg["main_heating_fuel"],
                         multiplier_key="heating_multiplier")
    epc = _attribute(rng, personas, pidx, None, list(tg["epc_rating"]), tg["epc_rating"],
                     multiplier_key="epc_multiplier")
    region = rng.choice(list(tg["region"]), n, p=np.array(list(tg["region"].values())) / sum(tg["region"].values()))
    tenure_tab = tg["home_status_by_age_band"]
    home = np.empty(n, dtype=object)
    for b in AGE_BANDS:
        m = band == b
        opts = list(tenure_tab[b])
        p = np.array([tenure_tab[b][o] for o in opts])
        home[m] = rng.choice(opts, m.sum(), p=p / p.sum())

    # Main vehicle type among households with a car, raked to licensed-car shares.
    vehicle = np.full(n, "none", dtype=object)
    has_car = n_veh > 0
    if has_car.any():
        vt = tg["vehicle_type_of_cars"]
        vehicle[has_car] = _attribute(rng, personas, pidx[has_car], None, list(vt), vt,
                                      multiplier_key="vehicle_type_multiplier")

    # Scenario customers: force their true attributes (before traits and behaviour).
    for i, spec in sc_rows.items():
        tr = spec["truth"]
        pidx[i] = names.index(tr["persona"])
        persona[i] = tr["persona"]
        age[i] = tr["age"]
        band[i] = next(b for b, (lo, hi) in AGE_BANDS.items() if lo <= tr["age"] <= hi)
        region[i], urban[i], hh_size[i], n_veh[i] = tr["region"], tr["urban_rural"], tr["household_size"], tr["num_vehicles"]
        income[i], home[i], prop[i], epc[i] = tr["income_band"], tr["home_status"], tr["property_type"], tr["epc_rating"]
        heating[i], vehicle[i] = tr["main_heating_fuel"], tr["vehicle_type"]

    # 2. Hidden traits: persona mean + noise, partial links, standardised.
    links = pc["trait_links"]
    inc_z, age_z, is_urban = _z([int(q[1]) for q in income]), _z(age), (urban == "urban").astype(float)
    latent = {"customer_id": ids, "persona": persona}
    for trait in T.LATENT_TRAITS:
        mean = np.array([personas[p]["traits"][trait] for p in persona])
        link = links.get(trait, {})
        raw = (mean + rng.normal(0, 1, n) + link.get("income_z", 0) * inc_z
               + link.get("age_z", 0) * age_z + link.get("urban", 0) * is_urban)
        latent[trait] = _z(raw)
    latent = pd.DataFrame(latent)
    eco, digital, headroom = (latent[t].to_numpy() for t in ("eco_mindedness", "digital_affinity", "financial_headroom"))

    # Solar: owner-occupied houses only, persona tilt, scaled to the national share.
    can_solar = (home != "renter") & (prop != "flat")
    tilt = np.array([personas[p].get("solar_multiplier", 1.0) for p in persona]) * np.exp(0.4 * eco) * can_solar
    scale = tg["has_solar"] * n / tilt.sum()
    has_solar = rng.random(n) < np.clip(tilt * scale, 0, 1)

    a = tg["assumptions"]
    paper_score = -digital + 0.3 * age_z + rng.normal(0, 0.5, n)
    statement = np.where(paper_score >= np.quantile(paper_score, 1 - a["paper_statement_share"]), "paper", "digital")
    consent = rng.random(n) < a["marketing_consent_share"]
    for i, spec in sc_rows.items():
        has_solar[i] = spec["truth"]["has_solar"]
        statement[i] = spec["truth"]["statement_pref"]
        consent[i] = spec["truth"]["marketing_consent"]

    customers = pd.DataFrame({
        "customer_id": ids, "age": age.astype(int), "region": region.astype(object),
        "urban_rural": urban, "household_size": hh_size, "num_vehicles": n_veh,
        "income_band": income, "home_status": home, "property_type": prop, "epc_rating": epc,
        "main_heating_fuel": heating, "has_solar": pd.array(has_solar, dtype="boolean"),
        "vehicle_type": vehicle, "statement_pref": statement.astype(object),
        "marketing_consent": consent, "data_as_of": cutoff,
    })
    truth = customers.copy()

    # 3. Record errors, injected after the truth snapshot.
    normal = np.arange(n) < n_normal  # random record errors never touch scenario customers
    single_ice = np.flatnonzero(np.isin(vehicle, ICE) & (n_veh == 1) & normal)
    sold_car = rng.choice(single_ice, size=min(len(single_ice), max(1, n_normal // 200)), replace=False)  # 0.5%
    sold_month = {int(i): int(rng.integers(3, n_feat - 1)) for i in sold_car}
    for frame in (customers, truth):
        frame.loc[sold_car, ["vehicle_type", "num_vehicles"]] = ["none", 0]
    pool = np.setdiff1d(single_ice, sold_car)
    conflicting = rng.choice(pool, size=min(len(pool), max(1, n_normal // 300)), replace=False)
    customers.loc[conflicting, "vehicle_type"] = "ev"           # record says EV; really a petrol/diesel car
    for col, rate in [("epc_rating", 0.08), ("property_type", 0.03), ("main_heating_fuel", 0.05),
                      ("vehicle_type", 0.03), ("has_solar", 0.03)]:
        customers.loc[(rng.random(n) < rate) & normal, col] = None
    for i, spec in sc_rows.items():
        b = spec["behaviour"]
        if b.get("sold_month"):  # last month with fuel; the car is sold after it
            sold_month[i] = int(months.get_loc(pd.Timestamp(f"{b['sold_month']}-01"))) + 1
            for frame in (customers, truth):
                frame.loc[i, ["vehicle_type", "num_vehicles"]] = ["none", 0]
        for col, value in spec["record"].items():
            customers.loc[i, col] = value

    # 4. Products held.
    held = [(cid, "current_account", pd.Timestamp("2015-01-01"), pd.NaT, 1500.0) for cid in ids]
    loan_p = a["vehicle_loan_share_of_car_households"]
    for i, cid in enumerate(ids):
        opened = pd.Timestamp("2016-01-01") + pd.Timedelta(int(rng.integers(0, 3300)), unit='D')
        if home[i] == "owner_mortgage":
            held.append((cid, "mortgage", opened, pd.NaT, float(rng.integers(60_000, 350_000))))
        wants_loan = (sc_rows[i]["behaviour"]["vehicle_loan"] if i in sc_rows
                      else n_veh[i] > 0 and rng.random() < loan_p)
        if wants_loan:
            if i in sold_month:
                held.append((cid, "vehicle_loan", opened, months[sold_month[i]], 0.0))
            else:
                held.append((cid, "vehicle_loan", opened, pd.NaT, float(rng.integers(2_000, 25_000))))
        for ptype, p in [("credit_card", 0.5 + 0.05 * int(income[i][1])), ("savings", 0.45 + 0.1 * headroom[i]),
                         ("investments", 1 / (1 + np.exp(-(-1.8 + 0.8 * headroom[i] + 0.3 * inc_z[i]))))]:
            if rng.random() < p:
                held.append((cid, ptype, opened, pd.NaT, float(rng.integers(100, 20_000))))
    products_held = pd.DataFrame(held, columns=["customer_id", "product_type", "open_date", "close_date", "balance"])

    # 5. Transactions (vectorised per customer x group).
    size_e = 0.6 + 0.2 * hh_size
    prop_f = pd.Series(prop).map({"detached": 1.25, "semi_detached": 1.05, "terraced": 0.95,
                                  "bungalow": 1.0, "flat": 0.75}).to_numpy()
    epc_f = pd.Series(epc).map({"B": 0.7, "C": 0.85, "D": 1.0, "E": 1.2, "F": 1.4, "G": 1.6}).to_numpy()
    area_drive = pd.Series(urban).map({"urban": 0.7, "town": 1.0, "rural": 1.4}).to_numpy()
    area_pt = pd.Series(urban).map({"urban": 2.0, "town": 1.0, "rural": 0.4}).to_numpy()
    income_f = np.array([0.6 + 0.2 * (int(q[1]) - 1) for q in income])
    dual = (heating == "gas") & (rng.random(n) < 0.5)
    for i, spec in sc_rows.items():
        dual[i] = spec["behaviour"]["dual_fuel"] and heating[i] == "gas"
    town = rng.choice(TOWNS, n)
    days_in = np.array([m.days_in_month for m in months])
    cols = {k: [] for k in ("customer_id", "date", "amount", "merchant_name", "mcc", "description",
                            "channel", "true_category", "group")}
    for i in range(n):
        main = vehicle[i]  # true main vehicle (a sold car still drove until its sale month)
        g: dict[str, float] = {"other": income_f[i] * (0.7 + 0.15 * hh_size[i]),
                               "utilities": 0.8 + 0.1 * hh_size[i]}
        if main in ICE or (main == "ev" and n_veh[i] == 2):
            per_car = 0.7 if main == "ev" else (1.0 if n_veh[i] == 1 else 1.7)
            level = sc_rows[i]["behaviour"]["fuel_level"] if i in sc_rows else rng.lognormal(0, 0.35)
            g["fuel"] = per_car * area_drive[i] * level * (0.65 if main == "hybrid" else 1.0)
            g["fuel_shop"] = 1.0
        if main == "ev":
            g["ev_charge"] = area_drive[i] * rng.lognormal(0, 0.3)
        if n_veh[i] > 0:
            g["automotive"] = 0.8 * n_veh[i]
        g["public_transport"] = area_pt[i] * (2.0 if n_veh[i] == 0 else 0.5) * np.exp(0.3 * eco[i])
        elec_level = sc_rows[i]["behaviour"]["elec_level"] if i in sc_rows else rng.lognormal(0, 0.25)
        elec = size_e[i] * prop_f[i] * elec_level * (2.2 if heating[i] == "electric" else 1.0)
        gas = (0.75 + 0.12 * hh_size[i]) * prop_f[i] * epc_f[i] * rng.lognormal(0, 0.25)
        if dual[i]:
            g["dual_fuel"] = (75 * elec + 85 * gas) / 160
        else:
            g["electricity"] = elec
            if heating[i] == "gas":
                g["gas"] = gas
        if heating[i] == "oil":
            g["heating_oil"] = prop_f[i] * epc_f[i]
        if home[i] != "renter":
            g["home_improvement"] = float(np.exp(0.3 * eco[i] + 0.2 * headroom[i]))
        else:
            g["home_improvement"] = 0.2
        for group, mult in g.items():
            mean_amt, rate, scales = BASE_RATES[group]
            lam = rate * (mult if scales == "count" else 1.0)
            counts = rng.poisson(lam, len(months))
            if i in sold_month and group in ("fuel", "fuel_shop", "automotive"):
                counts[sold_month[i]:] = 0
            k = int(counts.sum())
            if not k:
                continue
            m_idx = np.repeat(np.arange(len(months)), counts)
            day = (rng.random(k) * days_in[m_idx]).astype(int)
            amt_mean = mean_amt * (mult if scales == "amount" else 1.0)
            amount = np.round(rng.lognormal(np.log(amt_mean), 0.35 if scales == "count" else 0.15, k), 2)
            opts = MERCHANTS[group]
            pick = rng.integers(len(opts), size=k)
            drop_mcc = rng.random(k) < 0.05
            nums = rng.integers(1000, 9999, size=k)
            cols["customer_id"].extend([ids[i]] * k)
            cols["date"].extend(months[m_idx] + pd.to_timedelta(day, unit="D"))
            cols["amount"].extend(amount.tolist())
            for j in range(k):
                tmpl, mcc, cat, channel, desc = opts[pick[j]]
                cols["merchant_name"].append(tmpl.format(n=nums[j], town=town[i]))
                cols["mcc"].append(None if (mcc is None or drop_mcc[j]) else mcc)
                cols["description"].append(desc)
                cols["channel"].append(channel)
                cols["true_category"].append(cat)
            cols["group"].extend([group] * k)
    tx = pd.DataFrame(cols)
    tx = _scale_spend(tx, n, n_feat, cutoff, tg.get("weekly_spend_gbp"))
    tx = tx.sort_values(["customer_id", "date"], ignore_index=True)
    tx.insert(0, "txn_id", [f"T{j:09d}" for j in range(len(tx))])
    tx["mcc"] = tx["mcc"].astype("Int64")

    # 6. Digital activity.
    lam = np.exp(1.8 + 0.6 * digital)
    digital_activity = pd.DataFrame({
        "customer_id": np.repeat(ids, len(months)),
        "month": np.tile(months, n),
        "app_logins": rng.poisson(np.repeat(lam, len(months))),
        "web_sessions": rng.poisson(np.repeat(lam / 3, len(months))),
        "paper_statements_sent": np.repeat((statement == "paper").astype(int), len(months)),
    })

    # 7. Labels from the TRUE state.
    labels, diagnostics = generate_labels(_truth_drivers(truth, tx, digital_activity, products_held, ids, cutoff, n_feat),
                                          latent.drop(columns="persona"), seed)
    labels = labels[~labels["customer_id"].isin(list(sc))].reset_index(drop=True)  # scenarios: no labels
    labels.attrs["diagnostics"] = diagnostics
    tx = tx.drop(columns="group")
    return {"customers": customers, "products_held": products_held, "transactions": tx,
            "digital_activity": digital_activity, "labels": labels,
            "splits": split_customers(ids[:n_normal], seed), "latent_traits": latent}


def _scale_spend(tx: pd.DataFrame, n: int, n_feat: int, cutoff, weekly: dict | None) -> pd.DataFrame:
    """Scale each group's amounts so the national mean monthly spend per household
    (over the feature window, all households) matches ONS Family Spending."""
    if not weekly:
        warnings.warn("weekly_spend_gbp target missing: spend NOT scaled to ONS", stacklevel=2)
        return tx
    from src.features.build import dual_fuel_share
    win = tx["date"] <= cutoff
    monthly = tx[win].groupby("group")["amount"].sum() / n / n_feat
    def target(lines):
        return sum(weekly[x] for x in lines if weekly.get(x) is not None) * 52 / 12
    factor = {}
    for group, lines in ONS_LINES.items():
        if monthly.get(group, 0) > 0 and target(lines) > 0:
            factor[group] = target(lines) / monthly[group]
    share = dual_fuel_share()
    elec_now = monthly.get("electricity", 0) + share * monthly.get("dual_fuel", 0)
    gas_now = monthly.get("gas", 0) + (1 - share) * monthly.get("dual_fuel", 0)
    if weekly.get("electricity") and weekly.get("gas") and elec_now and gas_now:
        fe, fg = weekly["electricity"] * 52 / 12 / elec_now, weekly["gas"] * 52 / 12 / gas_now
        factor.update(electricity=fe, gas=fg, dual_fuel=share * fe + (1 - share) * fg)
    tx["amount"] = (tx["amount"] * tx["group"].map(factor).fillna(1.0)).round(2)
    return tx


def _truth_drivers(truth, tx, da, products_held, ids, cutoff, n_feat) -> pd.DataFrame:
    """Truth snapshot + true spend per month over the feature window, for labels."""
    from src.features.build import dual_fuel_share
    win = tx[tx["date"] <= cutoff]

    def monthly(groups):
        m = win[win["group"].isin(groups)].groupby("customer_id")["amount"].sum() / n_feat
        return m.reindex(ids, fill_value=0.0).to_numpy()

    share = dual_fuel_share()
    out = truth.copy()
    ph = products_held
    open_at = ph[(ph["open_date"] <= cutoff) & (ph["close_date"].isna() | (ph["close_date"] > cutoff))]
    for col, ptype in [("has_vehicle_loan", "vehicle_loan"), ("has_mortgage", "mortgage")]:
        out[col] = out["customer_id"].isin(open_at.loc[open_at["product_type"] == ptype, "customer_id"])
    out["fuel_spend_m"] = monthly(["fuel"])
    out["electricity_spend_m"] = monthly(["electricity"]) + share * monthly(["dual_fuel"])
    out["heating_spend_m"] = monthly(["gas", "heating_oil"]) + (1 - share) * monthly(["dual_fuel"])
    out["energy_spend_per_person_m"] = (out["electricity_spend_m"] + out["heating_spend_m"]) / out["household_size"]
    out["electricity_per_person_m"] = out["electricity_spend_m"] / out["household_size"]
    out["fuel_per_vehicle_m"] = out["fuel_spend_m"] / out["num_vehicles"].clip(lower=1)
    out["app_logins_m"] = da[da["month"] <= cutoff].groupby("customer_id")["app_logins"].mean().reindex(ids).to_numpy()
    return out


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if name in T.TABLES:
            errors = T.validate_table(df, name)
            if errors:
                raise ValueError("\n".join(errors))
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def main(argv: list[str] | None = None, default_n: int | None = None, default_out: Path | None = None) -> None:
    s = config.settings()
    p = argparse.ArgumentParser(description="Generate the synthetic dataset (P1).")
    p.add_argument("--n", type=int, default=default_n or s["data"]["n_customers"])
    p.add_argument("--seed", type=int, default=s["seed"])
    p.add_argument("--out", type=Path, default=default_out or config.RAW_DIR)
    args = p.parse_args(argv)
    tables = generate(args.n, args.seed)
    write_tables(tables, args.out)
    print(f"wrote {len(tables)} tables for {args.n} customers + {len(scenarios())} scenario customers to {args.out} "
          f"({len(tables['transactions']):,} transactions)")
    for code, d in tables["labels"].attrs.get("diagnostics", {}).items():
        print(f"  {code}: {d['eligible']} eligible, take-up {d['rate_among_eligible']:.1%}")


if __name__ == "__main__":
    main()
