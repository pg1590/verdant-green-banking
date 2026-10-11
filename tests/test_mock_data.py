"""The mock generator honours the table contracts. P1's real generator should
pass the same checks (swap generate_mock for generate)."""
import pandas as pd
import pytest

from src.common import config
from src.data_gen.mock import generate_mock
from src.schemas import tables as T


@pytest.fixture(scope="module")
def mock():
    return generate_mock(60, seed=7)


@pytest.mark.parametrize("name", list(T.TABLES))
def test_tables_match_schema(mock, name):
    assert T.validate_table(mock[name], name) == []


def test_seeded_and_reproducible(mock):
    again = generate_mock(60, seed=7)
    pd.testing.assert_frame_equal(mock["transactions"], again["transactions"])


def test_splits_are_customer_level_and_disjoint(mock):
    s = mock["splits"]
    assert s["customer_id"].is_unique
    from src.data_gen.generate import scenarios
    normal = set(mock["customers"]["customer_id"]) - set(scenarios())
    assert set(s["customer_id"]) == normal  # scenario customers have no split by design


def test_label_window_is_after_feature_cutoff(mock):
    cutoff = pd.Timestamp(config.settings()["calendar"]["feature_cutoff_date"])
    assert (mock["labels"]["label_window_start"] > cutoff).all()
    assert (mock["customers"]["data_as_of"] == cutoff).all()


def test_has_boundary_and_scenario_cases(mock):
    t = mock["transactions"]
    assert (t["true_category"] == "Utilities").any() and t["description"].str.contains("DUAL FUEL|GAS & ELEC").any()
    assert ((t["merchant_name"].str.startswith("SHELL")) & (t["true_category"] == "Other")).any()
    assert t["mcc"].isna().any()
    assert mock["customers"]["epc_rating"].isna().any()
