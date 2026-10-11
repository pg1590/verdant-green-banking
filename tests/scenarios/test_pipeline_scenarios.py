"""Judge-style scenarios (brief slide 16), end to end on GENERATED customers:
P1 generator (config/scenarios.yaml) -> features -> P5 eligibility rules.

Each scenario customer has a fixed ID and an expected outcome per product. These
checks need no trained model: eligibility, suppression and flags are deterministic.
"""
import pytest

from src.categorisation.rules import categorise_rules
from src.common.records import to_records
from src.data_gen.generate import generate, scenarios
from src.features.build import build_features
from src.recommend import check_eligibility, load_catalogue

SCENARIOS = scenarios()


@pytest.fixture(scope="module")
def generated():
    tables = generate(30, seed=21)
    features = build_features(tables, categorise_rules(tables["transactions"]))
    rows = {r["customer_id"]: r for r in to_records(features)}
    return tables, rows


@pytest.mark.parametrize("sid", list(SCENARIOS), ids=[f"{k} {v['name']}" for k, v in SCENARIOS.items()])
def test_scenario(generated, sid):
    _, rows = generated
    spec, row = SCENARIOS[sid], rows[sid]
    catalogue = load_catalogue()
    for product, expected in spec.get("expected", {}).items():
        result = check_eligibility(row, product, catalogue)
        if expected == "shown":
            assert result.eligible_to_show, f"{sid} {product}: suppressed by {result.suppression_reasons}"
        else:
            assert expected in result.suppression_reasons, f"{sid} {product}: got {result.suppression_reasons}"
    flags = set(row["conflict_flags"]) | set(filter(None, row["data_flags"].split(",")))
    for flag in spec.get("flags_present", []):
        assert flag in flags, f"{sid}: expected flag {flag}, got {sorted(flags)}"
    for flag in spec.get("flags_absent", []):
        assert flag not in flags, f"{sid}: unexpected flag {flag}"


def test_scenarios_have_fixed_ids_and_no_labels_or_split(generated):
    tables, rows = generated
    ids = set(SCENARIOS)
    assert ids <= set(tables["customers"]["customer_id"]) and ids <= set(rows)
    assert not ids & set(tables["labels"]["customer_id"])
    assert not ids & set(tables["splits"]["customer_id"])


def test_scenario_records_differ_from_truth_only_where_designed(generated):
    tables, _ = generated
    cust = tables["customers"].set_index("customer_id")
    assert cust.loc["C90003", "vehicle_type"] is None or cust.loc["C90003", "vehicle_type"] != cust.loc["C90003", "vehicle_type"]
    assert cust.loc["C90005", "vehicle_type"] == "ev"
    assert cust.loc["C90001", "epc_rating"] == "E"          # no random missing-data injection on scenarios
