"""Feature store v0: contract, leakage, dual fuel, conflict flags."""
import pandas as pd
import pytest

from src.categorisation.rules import categorise_rules, is_ev_charging
from src.common import config
from src.data_gen.mock import generate_mock
from src.features.build import CATEGORY_COLUMNS, build_features, dual_fuel_share, read_categories
from src.schemas import tables as T


@pytest.fixture(scope="module")
def raw():
    return generate_mock(80, seed=11)


@pytest.fixture(scope="module")
def features(raw):
    return build_features(raw, categorise_rules(raw["transactions"]))


def test_matches_feature_contract(features, raw):
    assert T.validate_table(features, "features") == []
    assert len(features) == len(raw["customers"])


def test_rules_categoriser_only_returns_allowed_categories(raw):
    assert categorise_rules(raw["transactions"]).isin(T.CATEGORIES).all()


def test_no_leakage_from_after_the_cutoff(raw, features):
    cutoff = pd.Timestamp(config.settings()["calendar"]["feature_cutoff_date"])
    t = raw["transactions"]
    future = t[t["date"] > cutoff].copy()
    assert len(future), "mock should include label-window transactions"
    future["amount"] *= 100  # wildly different future behaviour
    tampered = dict(raw, transactions=pd.concat([t[t["date"] <= cutoff], future], ignore_index=True))
    again = build_features(tampered, categorise_rules(tampered["transactions"]))
    pd.testing.assert_frame_equal(features, again)


def test_dual_fuel_is_apportioned(features):
    dual = features[features["energy_from_dual_fuel"]]
    assert len(dual) and dual["est_electricity_kwh_year"].notna().all()
    assert dual["data_flags"].str.contains("ENERGY_APPORTIONED_FROM_DUAL_FUEL").all()
    assert 0.2 < dual_fuel_share() < 0.6


def test_ev_charging_is_not_road_fuel(raw):
    t = raw["transactions"]
    charging = is_ev_charging(t)
    assert charging.any()
    assert (categorise_rules(t)[charging] == "Fuel").all()


def test_injected_scenarios_raise_conflict_flags(features):
    flags = set(features["conflict_flags"].explode().dropna())
    assert {"VEHICLE_SOLD", "VEHICLE_FUEL_CONFLICT"} <= flags
    sold = features[features["conflict_flags"].apply(lambda f: "VEHICLE_SOLD" in f)]
    assert (sold["vehicle_type"] == "none").all()


def test_km_only_for_combustion_vehicles(features):
    has_km = features["est_annual_km"].notna()
    assert features.loc[has_km, "vehicle_type"].isin(["petrol", "diesel", "hybrid"]).all()


def test_read_categories_matches_raw_order_and_rejects_mismatches(tmp_path):
    pd.DataFrame({"txn_id": ["T1", "T2", "T3"], "customer_id": ["C1"] * 3}) \
        .to_parquet(tmp_path / "transactions.parquet", index=False)
    cats = pd.DataFrame({"txn_id": ["T1", "T2", "T3"], "category": ["Fuel", "Gas", "Other"],
                         "is_ev_charging": [True, False, False], "is_dual_fuel": [False, False, True]})
    cats.to_parquet(tmp_path / "in_order.parquet", index=False)
    cats.iloc[::-1].to_parquet(tmp_path / "reversed.parquet", index=False)
    for name in ("in_order", "reversed"):  # P2's row order, or any other order realigned by txn_id
        got = read_categories(tmp_path, tmp_path / f"{name}.parquet")
        assert list(got.columns) == list(CATEGORY_COLUMNS) and list(got.index) == [0, 1, 2]
        assert got["category"].tolist() == ["Fuel", "Gas", "Other"]
        assert got["is_dual_fuel"].tolist() == [False, False, True]
    cats.iloc[:2].to_parquet(tmp_path / "missing.parquet", index=False)
    pd.concat([cats, cats.iloc[:1]]).to_parquet(tmp_path / "duplicated.parquet", index=False)
    for name in ("missing", "duplicated"):
        with pytest.raises(ValueError, match="must match the raw source"):
            read_categories(tmp_path, tmp_path / f"{name}.parquet")
    # duplicate IDs in the raw table itself are rejected even when the order matches
    dup = tmp_path / "dup"
    dup.mkdir()
    pd.DataFrame({"txn_id": ["T1", "T1"]}).to_parquet(dup / "transactions.parquet", index=False)
    cats.iloc[[0, 0]].to_parquet(dup / "categorised.parquet", index=False)
    with pytest.raises(ValueError, match="must match the raw source"):
        read_categories(dup, dup / "categorised.parquet")
