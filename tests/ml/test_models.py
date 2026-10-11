"""P3 propensity subsystem tests. Run: pytest tests/ml

End-to-end tests train on a small dataset from P1's real generator (src.data_gen) and
feature store (src.features.build), so they break if the P1 -> P3 contract drifts.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification

from src.models import evaluate as ev
from src.models.confidence import compute_confidence, quality_multiplier
from src.models.config import check_feature_policy, feature_list, load_config, load_reason_codes, sources
from src.models.contract import validate_scores
from src.models.data import encode
from src.models.reasons import passes_gate, shap_values
from src.schemas import tables as T

# Codes whose model feature is constant inside the product's training population (P5's
# catalogue already requires it), so they can never be a positive driver.
ELIGIBILITY_ONLY = {"HOMEOWNER", "SUITABLE_PROPERTY_TYPE", "NO_EXISTING_SOLAR", "EXISTING_VEHICLE"}


# ---------- schema-independent units ----------

def test_metrics_on_known_signal():
    X, y = make_classification(2000, n_features=5, weights=[0.9], random_state=0)
    m_good = ev.metrics(y, 1 / (1 + np.exp(-X[:, 0] * 3)))
    m_rand = ev.metrics(y, np.random.default_rng(0).random(len(y)))
    assert 0 <= m_good["brier"] <= 1 and 0 <= m_good["ece"] <= 1
    assert m_rand["roc_auc"] == pytest.approx(0.5, abs=0.06)
    lift = ev.lift_table(y, X[:, 0])
    assert len(lift) == 10 and lift["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_perfect_calibration_has_low_ece():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert ev.ece(y, p) < 0.02


def test_confidence_bounds_and_direction():
    ens_agree = np.full((5, 3), 0.4)
    ens_disagree = np.array([[0.1] * 3, [0.7] * 3, [0.2] * 3, [0.6] * 3, [0.4] * 3])
    full = compute_confidence([1, 1, 1], ens_agree)
    assert np.allclose(full, 1.0)
    assert (compute_confidence([0.5] * 3, ens_agree) < full).all()            # missing data
    assert (compute_confidence([1] * 3, ens_disagree) < full).all()           # unstable model
    assert (compute_confidence([1] * 3, ens_agree, [0.5] * 3) < full).all()   # conflicting data
    assert compute_confidence([np.nan], np.full((5, 1), 0.4))[0] == 0.0


def test_confidence_spread_is_measured_in_log_odds():
    """The same relative disagreement must not punish high-propensity customers more."""
    low = np.array([[0.010], [0.012], [0.008], [0.011], [0.009]])
    high = np.array([[0.50], [0.55], [0.45], [0.52], [0.48]])
    assert compute_confidence([1], high)[0] >= compute_confidence([1], low)[0] - 0.05


def test_quality_multiplier_uses_conflicts_and_data_flags():
    gp01 = load_config()["products"]["GP01"]
    q = quality_multiplier(
        [[], ["VEHICLE_FUEL_CONFLICT"], ["EPC_CONFLICT"], [], np.array(["VEHICLE_SOLD"])],
        ["", "", "", "RECENT_CIRCUMSTANCE_CHANGE,LOW_DATA_COMPLETENESS", None],
        gp01["conflict_codes"], gp01["data_flag_penalties"], conflict_penalty=0.5)
    assert q[0] == 1.0 and q[1] == 0.5
    assert q[2] == 1.0                     # EPC conflict is irrelevant to EV finance
    assert q[3] == pytest.approx(gp01["data_flag_penalties"]["RECENT_CIRCUMSTANCE_CHANGE"])
    assert q[4] == 0.5                     # numpy array from parquet


def test_evidence_gate():
    assert passes_gate(210.0, ">=", 120) and not passes_gate(20.0, ">=", 120)
    assert not passes_gate(float("nan"), ">=", 120) and not passes_gate(5.0, ">=", None)


def test_fairness_slices_hide_unreliable_auc():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": ["big"] * 2000 + ["small"] * 40, "p": rng.random(2040)})
    df["y"] = (rng.random(2040) < df["p"]).astype(int)
    out = ev.fairness_slices(df, "g", "y", "p").set_index("group")
    assert out.loc["big", "auc_reliable"] and out.loc["big", "roc_auc_ci95_lo"] < out.loc["big", "roc_auc"]
    small = df[df.g == "small"]
    if small.y.sum() < 10 or (1 - small.y).sum() < 10:
        assert not out.loc["small", "auc_reliable"] and np.isnan(out.loc["small", "roc_auc"])
    assert out.loc["big", "calibration_ratio"] == pytest.approx(1.0, abs=0.1)


# ---------- config follows P1's schema and the feature policy ----------

def test_model_features_follow_schema_and_exclude_age():
    cfg = load_config()
    for code in cfg["products"]:
        for feat in feature_list(cfg, code):
            for src in sources(cfg, feat):
                assert src in T.FEATURES, f"{code}: {feat} reads {src}, not in src/schemas/tables.py"
            assert "age" not in sources(cfg, feat)
    for spec in cfg["slices"].values():
        assert spec["source"] in T.FEATURES


def test_forbidden_inputs_fail_loudly():
    cfg = load_config()
    for bad in ("age", "eco_mindedness", "persona", "conflict_flags", "took_up"):
        broken = copy.deepcopy(cfg)
        broken["products"]["GP01"]["features"][bad] = 0
        with pytest.raises(AssertionError, match="never be a model input"):
            check_feature_policy(broken)
    broken = copy.deepcopy(cfg)
    broken["derived"]["age_years"] = {"source": "age", "map": {}}
    broken["products"]["GP01"]["features"]["age_years"] = 0
    with pytest.raises(AssertionError):
        check_feature_policy(broken)


def test_reason_codes_reachable_from_model_features():
    """Every dictionary code maps to a model feature of its product, except eligibility-only
    codes (constant inside the training population, so never a positive driver)."""
    cfg = load_config()
    compiled = load_reason_codes(cfg)
    for code, entry in compiled.items():
        for p in entry["products"]:
            assert entry["feature"] in cfg["products"][p]["features"], code
    with open(cfg["_root"] / cfg["reasons"]["dictionary"]) as f:
        everything = set(yaml.safe_load(f)["reason_codes"])
    assert everything - set(compiled) <= ELIGIBILITY_ONLY


def test_encode_preserves_missing_and_is_idempotent():
    cfg = load_config()
    df = pd.DataFrame({"income_band": ["Q1", None], "epc_rating": ["G", None],
                       "vehicle_type": ["diesel", None], "household_size": [2, 0],
                       "est_electricity_kwh_year": [3000.0, 2500.0], "age": [25, 70]})
    out = encode(cfg, df)
    assert out["income_band_ord"].tolist()[0] == 1 and np.isnan(out["income_band_ord"][1])
    assert out["epc_rating_ord"][0] == 7 and out["is_ice_vehicle"][0] == 1 and np.isnan(out["is_ice_vehicle"][1])
    assert out["electricity_kwh_per_person"][0] == 1500 and np.isnan(out["electricity_kwh_per_person"][1])
    assert out["age_band"].tolist() == ["18-30", "60+"]
    pd.testing.assert_frame_equal(encode(cfg, out), out)


# ---------- end-to-end on P1's generator ----------

@pytest.fixture(scope="module")
def dataset():
    from src.categorisation.rules import categorise_rules
    from src.data_gen.mock import generate_mock
    from src.features.build import build_features
    raw = generate_mock(3000, seed=7)
    feats = build_features(raw, categorise_rules(raw["transactions"]))
    return feats, raw["labels"], raw["splits"]


@pytest.fixture(scope="module")
def trained(dataset):
    from src.models.data import product_frame
    from src.models.train import fit_product
    cfg = load_config()
    feats, labels, splits = dataset
    bundles = {}
    for code in cfg["products"]:
        df = product_frame(cfg, code, feats, labels, splits)
        bundles[code] = fit_product(cfg, code, df[df["split"] == "train"], fast=True)
    return cfg, feats, bundles


def test_model_selection_on_train_cv(trained):
    cfg, _, bundles = trained
    for b in bundles.values():
        assert b["family"] in cfg["training"]["families"]
        assert b["cv_scores"][b["family"]] == max(b["cv_scores"].values())
        assert b["model_version"] == f"{b['product_code'].lower()}_{b['family']}_{cfg['version']}"


def test_reason_codes_explain_the_scoring_model(trained):
    """SHAP explains the model the calibrator wraps, and sums to its log-odds."""
    from src.models.data import numeric
    _, feats, bundles = trained
    for b in bundles.values():
        assert b["calibrated"].calibrated_classifiers_[0].estimator is b["base"]
        X = numeric(encode(load_config(), feats.head(50)), b["features"])
        sv = shap_values(b["base"], X)
        if b["family"] == "logreg":
            logit = b["base"].decision_function(X) - b["base"][-1].intercept_[0]
        else:
            logit = b["base"].predict_proba(X, raw_score=True) - b["base"].booster_.predict(
                X, pred_contrib=True)[:, -1]
        np.testing.assert_allclose(sv.sum(axis=1), logit, atol=1e-6)


def test_scores_respect_contract(trained):
    from src.models.predict import score
    cfg, feats, bundles = trained
    out = score(feats, cfg, bundles)
    validate_scores(out, load_reason_codes(cfg))
    assert len(out) == len(feats) * len(cfg["products"])


def test_reason_codes_backed_by_evidence_one_per_feature(trained):
    from src.models.predict import score
    cfg, feats, bundles = trained
    out = score(feats.head(300), cfg, bundles).merge(encode(cfg, feats.head(300)), on="customer_id")
    emitted = 0
    for _, r in out.iterrows():
        codes = json.loads(r["reason_codes"])
        d = bundles[r["product_code"]]["reason_dictionary"]
        assert len(codes) <= cfg["reasons"]["max_codes"]
        assert len({d[c]["feature"] for c in codes}) == len(codes), f"two codes for one feature: {codes}"
        for code in codes:
            emitted += 1
            assert passes_gate(r[d[code]["feature"]], d[code]["op"], d[code]["threshold"]), code
        assert set(json.loads(r["evidence"])) == {d[c]["evidence_key"] for c in codes}
    assert emitted


def test_deterministic(trained):
    from src.models.predict import score
    cfg, feats, bundles = trained
    a, b = score(feats.head(200), cfg, bundles), score(feats.head(200), cfg, bundles)
    pd.testing.assert_frame_equal(a, b)


def test_score_requires_confidence_inputs(trained):
    from src.models.predict import score
    cfg, feats, bundles = trained
    for col in ("data_completeness", "conflict_flags", "data_flags"):
        with pytest.raises(ValueError, match=col):
            score(feats.head(5).drop(columns=col), cfg, bundles)
    with pytest.raises(ValueError, match="missing values"):
        score(feats.head(5).assign(data_completeness=np.nan), cfg, bundles)


def test_scenarios(trained):
    """Brief's scenario tests at the propensity layer: changed circumstances, sparse data,
    conflicting data, dual-fuel apportioning."""
    from src.models.predict import score
    cfg, feats, bundles = trained
    base = feats[(feats["vehicle_type"] == "petrol") & feats["spend_fuel_m"].notna()].iloc[[0]].copy()
    base[["spend_fuel_m", "spend_fuel_recent_m", "est_annual_km", "km_per_vehicle", "has_vehicle_loan"]] = \
        [250.0, 250.0, 30000.0, 30000.0, True]
    base = base.assign(data_completeness=1.0, conflict_flags=[[]], data_flags="")
    sold_car = base.assign(customer_id="SOLD", spend_fuel_recent_m=0.0)
    sparse = base.assign(customer_id="SPARSE", data_completeness=0.3, spend_fuel_m=np.nan, est_annual_km=np.nan)
    conflict = base.assign(customer_id="CONFLICT", conflict_flags=[["VEHICLE_FUEL_CONFLICT"]])
    dual = base.assign(customer_id="DUAL", data_flags="ENERGY_APPORTIONED_FROM_DUAL_FUEL")
    s = score(pd.concat([base, sold_car, sparse, conflict, dual]), cfg, bundles).set_index(
        ["product_code", "customer_id"])
    b = base["customer_id"].iloc[0]
    if "spend_fuel_recent_m" not in bundles["GP01"]["sign_violations"]:
        assert s.loc[("GP01", "SOLD"), "propensity"] <= s.loc[("GP01", b), "propensity"]
    assert s.loc[("GP01", "SPARSE"), "confidence"] < s.loc[("GP01", b), "confidence"]
    assert "HIGH_FUEL_SPEND" not in json.loads(s.loc[("GP01", "SPARSE"), "reason_codes"])
    assert s.loc[("GP01", "CONFLICT"), "confidence"] < s.loc[("GP01", b), "confidence"]
    assert s.loc[("GP02", "DUAL"), "confidence"] < s.loc[("GP02", b), "confidence"]
    assert s.loc[("GP01", "DUAL"), "confidence"] == s.loc[("GP01", b), "confidence"]


def test_p5_service_accepts_p3_predictions(trained):
    """P3's mapping loads into P5's RecommendationService as TEAM_OUTPUTS, and only
    catalogue-eligible products with evidence-backed reasons are shown."""
    from src.common.records import to_records
    from src.models.predict import score, to_p5_predictions
    from src.recommend.service import RecommendationService
    cfg, feats, bundles = trained
    sample = feats.head(200)
    svc = RecommendationService({"schema_version": "1.0", "data_mode": "TEAM_OUTPUTS",
                                 "customers": to_records(sample), "impacts": {},
                                 "predictions": to_p5_predictions(score(sample, cfg, bundles))})
    rows = [r for cid in svc.customers for r in svc.recommendations(cid, include_suppressed=True)["recommendations"]]
    assert len(rows) == 200 * 4
    shown = [r for r in rows if r["eligible_to_show"]]
    assert shown, "no recommendation survived P5's checks"
    assert all(r["reason_codes"] and r["structurally_eligible"] for r in shown)


def test_dataset_mismatch_fails_loudly(dataset):
    from src.models.data import check_same_dataset
    cfg = load_config()
    feats, labels, splits = dataset
    check_same_dataset(cfg, feats, labels, splits)
    with pytest.raises(AssertionError, match="have a split"):
        check_same_dataset(cfg, feats, labels, splits.head(100))
    with pytest.raises(AssertionError, match="have a label"):
        check_same_dataset(cfg, feats, labels[labels.customer_id.isin(feats.customer_id.head(100))], splits)
    with pytest.raises(AssertionError, match="duplicate"):
        check_same_dataset(cfg, pd.concat([feats, feats.head(1)]), labels, splits)


def test_small_run_with_scenario_customers_passes_dataset_check():
    """The 13 judge-scenario customers have no split or labels by design; a 200-customer
    run used to fail the 99% overlap (13/213 unsplit)."""
    from src.categorisation.rules import categorise_rules
    from src.data_gen.generate import generate, scenarios
    from src.features.build import build_features
    from src.models.data import check_same_dataset
    raw = generate(200, seed=5)
    feats = build_features(raw, categorise_rules(raw["transactions"]))
    assert set(scenarios()) <= set(feats["customer_id"])
    check_same_dataset(load_config(), feats, raw["labels"], raw["splits"])


def test_cli_train_retrain_and_one_shot_test(dataset, tmp_path):
    """CLI end to end in a temp dir: train, retrain one product (reports merged), held-out
    test run writes exactly the agreed columns and refuses to run twice without --force."""
    from src.models import run_test, train
    feats, labels, splits = dataset
    for name, df in (("features", feats), ("labels", labels), ("splits", splits)):
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    cfg = yaml.safe_load(open(load_config()["_root"] / "config" / "model.yaml"))
    cfg["paths"] = {**{k: str(tmp_path / f"{k}.parquet") for k in ("features", "labels", "splits")},
                    "models_dir": str(tmp_path / "models"), "reports_dir": str(tmp_path / "reports"),
                    "scores": str(tmp_path / "scores.parquet"), "p5_predictions": str(tmp_path / "p5.json"),
                    "test_predictions": str(tmp_path / "out" / "test_predictions.csv")}
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    train.main(["--fast", "--config", str(cfg_path)])
    manifest = json.loads((tmp_path / "models" / "gp01_propensity_v3.manifest.json").read_text())
    assert set(manifest["input_sha1"]) >= {"features", "labels", "splits", "model.yaml", "reason_codes.yaml"}
    assert manifest["model_version"].startswith("gp01_") and manifest["reason_thresholds"]
    train.main(["--fast", "--config", str(cfg_path), "--products", "GP04"])
    metrics = json.loads((tmp_path / "reports" / "metrics_val.json").read_text())
    assert set(metrics) == {"GP01", "GP02", "GP03", "GP04"}
    assert set(pd.read_csv(tmp_path / "reports" / "fairness_val.csv")["product_code"]) == set(metrics)

    run_test.main(["--config", str(cfg_path)])
    out = pd.read_csv(tmp_path / "out" / "test_predictions.csv")
    assert tuple(out.columns) == T.TEST_PREDICTION_COLUMNS
    assert len(out) == (splits.split == "test").sum() * 4
    with pytest.raises(SystemExit, match="already evaluated"):
        run_test.main(["--config", str(cfg_path)])
    run_test.main(["--config", str(cfg_path), "--force"])
    assert "force=True" in (tmp_path / "out" / "test_runs.log").read_text()
