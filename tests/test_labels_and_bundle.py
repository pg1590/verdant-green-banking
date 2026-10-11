"""Label rules v0, reason-code gating and the mock bundle."""
import numpy as np
import pandas as pd
import pytest

from src.categorisation.rules import categorise_rules
from src.common import config
from src.data_gen.labels import generate_labels, label_rules
from src.data_gen.mock import generate_mock
from src.features.build import build_features
from src.models.reason_codes import evidence_codes, evidence_for, fit_thresholds
from src.recommend.service import RecommendationService
from src.schemas import tables as T


def _truth(n, rng):
    d = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": rng.integers(18, 85, n),
        "urban_rural": rng.choice(list(T.URBAN_RURAL), n),
        "household_size": rng.integers(1, 6, n),
        "num_vehicles": rng.integers(0, 3, n),
        "income_band": rng.choice(list(T.INCOME_BANDS), n),
        "home_status": rng.choice(list(T.HOME_STATUS), n),
        "property_type": rng.choice(list(T.PROPERTY_TYPES), n),
        "epc_rating": rng.choice(list(T.EPC_RATINGS), n),
        "main_heating_fuel": rng.choice(list(T.HEATING_FUELS), n),
        "has_solar": rng.random(n) < 0.1,
        "vehicle_type": rng.choice(list(T.VEHICLE_TYPES), n),
        "statement_pref": rng.choice(list(T.STATEMENT_PREFS), n),
        "marketing_consent": True,
        "has_vehicle_loan": rng.random(n) < 0.3,
        "has_mortgage": rng.random(n) < 0.4,
        "fuel_spend_m": rng.gamma(2, 50, n), "heating_spend_m": rng.gamma(2, 40, n),
        "electricity_spend_m": rng.gamma(2, 35, n), "app_logins_m": rng.gamma(2, 4, n),
    })
    d["fuel_per_vehicle_m"] = d["fuel_spend_m"] / d["num_vehicles"].clip(lower=1)
    d["electricity_per_person_m"] = d["electricity_spend_m"] / d["household_size"]
    d["energy_spend_per_person_m"] = (d["electricity_spend_m"] + d["heating_spend_m"]) / d["household_size"]
    return d


@pytest.fixture(scope="module")
def generated():
    rng = np.random.default_rng(0)
    truth = _truth(3000, rng)
    latent = pd.DataFrame({"customer_id": truth["customer_id"], **{t: rng.normal(0, 1, 3000) for t in T.LATENT_TRAITS}})
    labels, diag = generate_labels(truth, latent, seed=1)
    return truth, latent, labels, diag


def test_labels_match_contract_and_hit_target_rates(generated):
    _, _, labels, diag = generated
    assert T.validate_table(labels, "labels") == []
    for code, spec in label_rules()["products"].items():
        assert abs(diag[code]["rate_among_eligible"] - spec["target_rate"]) < 0.03, code


def test_ruled_out_customers_never_take_up(generated):
    truth, _, labels, _ = generated
    renters = truth.loc[truth["home_status"] == "renter", "customer_id"]
    solar = labels[(labels["product_code"] == "GP03") & labels["customer_id"].isin(renters)]
    assert solar["took_up"].sum() == 0
    digital = truth.loc[truth["statement_pref"] == "digital", "customer_id"]
    assert labels[(labels["product_code"] == "GP04") & labels["customer_id"].isin(digital)]["took_up"].sum() == 0


def test_labels_are_reproducible(generated):
    truth, latent, labels, _ = generated
    again, _ = generate_labels(truth, latent, seed=1)
    pd.testing.assert_frame_equal(labels, again)


def test_hidden_traits_change_labels(generated):
    truth, latent, labels, _ = generated
    flipped = latent.assign(**{t: -latent[t] for t in T.LATENT_TRAITS})
    other, _ = generate_labels(truth, flipped, seed=1)
    assert not labels["took_up"].equals(other["took_up"])


@pytest.fixture(scope="module")
def features():
    raw = generate_mock(80, seed=5)
    return build_features(raw, categorise_rules(raw["transactions"]))


def test_reason_codes_need_evidence(features):
    thresholds = fit_thresholds(features)
    for row in features.to_dict("records"):
        for code in evidence_codes(row, "GP01", thresholds):
            assert "GP01" in config.reason_codes()["reason_codes"][code]["products"]
        low = dict(row, spend_fuel_m=0.0)
        assert "HIGH_FUEL_SPEND" not in evidence_codes(low, "GP01", thresholds)
    assert evidence_for({"spend_fuel_m": 123.456}, ["HIGH_FUEL_SPEND"]) == {"spend_fuel_m": 123.46}


def test_mock_bundle_loads_in_p5_service(features):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "build_mock_bundle", Path(__file__).resolve().parents[1] / "scripts" / "build_mock_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    service = RecommendationService(mod.build_bundle(features))
    assert service.mode == "DEMO_MOCK"
    summary = service.summary()
    assert summary["customers_analysed"] == len(features)
    assert summary["eligible_opportunities"] > 0
