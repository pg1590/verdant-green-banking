"""FR2 model + pipeline stage: output contract, rule overrides and leakage guards. Owner: P2."""
import pandas as pd
import pytest

from src.categorisation import evaluate
from src.categorisation import model as fr2
from src.schemas.tables import CATEGORIES


@pytest.fixture(scope="module")
def hand():
    return evaluate.load_hand_labelled()


@pytest.fixture(scope="module")
def trained(hand):
    return fr2.train(hand, seed=0)


def test_model_inputs_never_include_the_label(hand):
    x = fr2.model_inputs(hand)
    assert "true_category" not in x.columns
    assert not any("true_category" in c for c in x.columns)


def test_categorise_contract(trained, hand):
    out = fr2.categorise(trained, hand.drop(columns=["true_category"]))
    assert list(out.columns) == list(fr2.OUTPUT_COLUMNS)
    assert out.index.equals(hand.index)
    assert out["category"].isin(CATEGORIES).all()
    assert out["category_confidence"].between(0, 1).all()
    assert out["category_source"].str.match(r"^(model|rule:[A-Z_]+)$").all()
    # flags only ever sit on their own category
    assert (out.loc[out["is_ev_charging"], "category"] == "Fuel").all()
    assert (out.loc[out["is_dual_fuel"], "category"] == "Utilities").all()


def test_strong_rules_override_the_model(trained):
    txns = pd.DataFrame({
        "txn_id": ["A", "B"], "customer_id": ["C1", "C1"], "date": pd.to_datetime(["2026-01-01"] * 2),
        "amount": [4.0, 30.0], "merchant_name": ["SHELL 9 YORK", "BP PULSE UK"],
        "mcc": pd.array([5541, 5552], dtype="Int64"), "description": ["SHOP PURCHASE", "CARD PAYMENT"],
        "channel": ["card", "card"],
    })
    out = fr2.categorise(trained, txns)
    assert out["category"].tolist() == ["Other", "Fuel"]
    assert out["category_source"].tolist() == ["rule:FORECOURT_SHOP", "rule:EV_CHARGING"]
    assert out["category_confidence"].tolist() == [1.0, 1.0]
    assert out["is_ev_charging"].tolist() == [False, True]


def test_training_is_deterministic(hand):
    a = fr2.predict(fr2.train(hand, seed=0), hand)
    b = fr2.predict(fr2.train(hand, seed=0), hand)
    assert a.equals(b)


def test_train_rejects_unknown_labels(hand):
    bad = hand.copy()
    bad.loc[bad.index[0], "true_category"] = "Groceries"
    with pytest.raises(ValueError):
        fr2.train(bad)


def test_main_writes_output_without_true_category(tmp_path, hand):
    data = tmp_path / "data"
    data.mkdir()
    hand.drop(columns=["note"]).to_parquet(data / "transactions.parquet", index=False)
    customers = sorted(hand["customer_id"].unique())
    pd.DataFrame({"customer_id": customers,
                  "split": ["train"] * (len(customers) - 2) + ["val", "test"]}) \
        .to_parquet(data / "splits.parquet", index=False)
    out_path, model_path = tmp_path / "categorised.parquet", tmp_path / "categoriser.joblib"
    fr2.main(["--data-dir", str(data), "--out", str(out_path), "--model-path", str(model_path)])
    out = pd.read_parquet(out_path)
    assert "true_category" not in out.columns
    assert set(fr2.OUTPUT_COLUMNS) <= set(out.columns)
    assert len(out) == len(hand) and out["txn_id"].is_unique
    assert fr2.load(model_path).version == fr2.MODEL_VERSION


def test_brand_split_keeps_every_category_trainable(hand):
    held = evaluate.holdout_brands(hand, frac=0.5, seed=0)
    brands = hand["merchant_name"].map(evaluate.brand)
    kept = hand[~brands.isin(held)]
    assert held, "something should be held out"
    assert set(kept["true_category"]) == set(hand["true_category"])


def test_keyword_rules_beat_the_model(trained):
    """F4 regression: a merchant the keyword rules know is never overridden by the model."""
    txns = pd.DataFrame({
        "txn_id": ["K1", "K2", "K3"], "customer_id": ["C1"] * 3, "date": pd.to_datetime(["2026-01-01"] * 3),
        "amount": [11.99, 320.0, 45.0],
        "merchant_name": ["NETFLIX.COM", "BLACK HORSE LTD", "CAR INSURANCE DD"],
        "mcc": pd.array([4899, None, None], dtype="Int64"),
        "description": ["CARD PAYMENT", "DIRECT DEBIT", "DIRECT DEBIT"],
        "channel": ["card", "direct_debit", "direct_debit"],
    })
    out = fr2.categorise(trained, txns)
    assert out["category"].tolist() == ["Other", "Other", "Automotive"]
    assert (out["category_source"] == "rule:KEYWORD").all()


def test_model_only_replaces_a_fallback_when_confident(trained):
    """An unknown merchant with no MCC: the rules default to Other; the model may replace that
    only at >= MODEL_MIN_CONFIDENCE, and a threshold above 1 means the rule fallback always stands."""
    txns = pd.DataFrame({
        "txn_id": ["U1"], "customer_id": ["C1"], "date": pd.to_datetime(["2026-01-01"]), "amount": [60.0],
        "merchant_name": ["ZYXW QUBIT 77"], "mcc": pd.array([None], dtype="Int64"),
        "description": ["CARD PAYMENT"], "channel": ["card"],
    })
    never = fr2.categorise(trained, txns, min_model_confidence=1.01)
    assert never["category"].tolist() == ["Other"] and never["category_source"].tolist() == ["rule:DEFAULT_OTHER"]
    always = fr2.categorise(trained, txns, min_model_confidence=0.0)
    pred = fr2.model_only(trained, txns)["pred"].iloc[0]
    assert always["category"].tolist() == [pred] and always["category_source"].tolist() == ["model"]
    assert 0 < always["category_confidence"].iloc[0] <= 1
