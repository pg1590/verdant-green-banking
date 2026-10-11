"""FR3 inference: score every customer for every product. Owner: P3.

    python -m src.models.predict     # -> data/processed/scores.parquet + p3_predictions.json

    from src.models.predict import predict, score
    predict(features_df)   # P5's prediction input {customer_id: {GP0X: {...}}}
    score(features_df)     # long format, see contract.SCORE_COLUMNS

Every customer is scored for every product; P5 suppresses at display time.
"""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from .confidence import compute_confidence, quality_multiplier
from .config import bundle_stem, feature_list, load_config, load_reason_codes, path
from .contract import validate_scores
from .data import eligible_mask, encode, numeric
from .reasons import explain


def load_bundles(cfg: dict) -> dict:
    """Load every product's bundle; refuse stale models (features or model config changed)."""
    d = path(cfg, "models_dir")
    expected_hash = hashlib.sha1(path(cfg, "features").read_bytes()).hexdigest()[:12]
    bundles = {}
    for code in cfg["products"]:
        stem = bundle_stem(cfg, code)
        manifest = json.loads((d / f"{stem}.manifest.json").read_text())
        if manifest["train_data_sha1"] != expected_hash:
            raise ValueError(f"{code}: features changed since training; retrain before exporting predictions")
        bundle = joblib.load(d / f"{stem}.pkl")
        if bundle["features"] != feature_list(cfg, code):
            raise ValueError(f"{code}: model configuration changed; retrain before inference")
        bundles[code] = bundle
    return bundles


def score(features: pd.DataFrame, cfg: dict | None = None, bundles: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    bundles = bundles or load_bundles(cfg)
    features = encode(cfg, features).reset_index(drop=True)
    c, ccfg = cfg["columns"], cfg["confidence"]
    # Confidence inputs are mandatory: defaulting them would silently report full confidence.
    required = [c["id"], c["cutoff"], c["completeness"], c["conflict_flags"], c["data_flags"]]
    absent = [col for col in required if col not in features.columns]
    if absent:
        raise ValueError(f"feature table lacks {absent}; P1 always exports them (src/schemas/tables.py)")
    if features[c["completeness"]].isna().any():
        raise ValueError(f"{c['completeness']} has missing values; it must be 0-1 for every customer")
    completeness = features[c["completeness"]]
    cutoff = pd.to_datetime(features[c["cutoff"]]).dt.date.astype(str)
    conflicts, data_flags = features[c["conflict_flags"]], features[c["data_flags"]]

    frames = []
    for code, b in bundles.items():
        pcfg = cfg["products"][code]
        X = numeric(features, b["features"])
        propensity = b["calibrated"].predict_proba(X)[:, 1]
        ens = np.vstack([m.predict_proba(X)[:, 1] for m in b["ensemble"]])
        quality = quality_multiplier(conflicts, data_flags, pcfg["conflict_codes"],
                                     pcfg.get("data_flag_penalties") or {}, ccfg["conflict_penalty"])
        confidence = compute_confidence(completeness, ens, quality, ccfg["std_norm"])
        codes, evidence = explain(b["base"], X, code, b["reason_dictionary"], cfg["reasons"]["max_codes"])
        frames.append(pd.DataFrame({
            "customer_id": features[c["id"]].astype(str).to_numpy(),
            "product_code": code,
            "opportunity": pcfg["opportunity"],
            "propensity": np.round(propensity, 4),
            "confidence": np.round(confidence, 4),
            "reason_codes": [json.dumps(x) for x in codes],
            "evidence": [json.dumps(x) for x in evidence],
            "in_model_population": eligible_mask(cfg, features, code).to_numpy(),
            "model_version": b["model_version"],
            "feature_cutoff_date": np.asarray(cutoff),
        }))
    out = pd.concat(frames, ignore_index=True)
    validate_scores(out, load_reason_codes(cfg))
    return out


def to_p5_predictions(scores: pd.DataFrame) -> dict:
    """P5 bundle v1 `predictions` block: {customer_id: {GP0X: {propensity, confidence,
    reason_codes, evidence, model_version}}} with plain-Python JSON values (docs/integration.md)."""
    out: dict = {}
    for r in scores.itertuples(index=False):
        out.setdefault(r.customer_id, {})[r.product_code] = {
            "propensity": float(r.propensity), "confidence": float(r.confidence),
            "reason_codes": json.loads(r.reason_codes), "evidence": json.loads(r.evidence),
            "model_version": r.model_version}
    return out


def predict(features: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """P5's prediction input for every customer x product (see module docstring)."""
    return to_p5_predictions(score(features))


def main() -> None:
    cfg = load_config()
    features = pd.read_parquet(path(cfg, "features"))
    out = score(features, cfg)
    dest = path(cfg, "scores")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"wrote {len(out)} rows -> {dest}")
    p5 = path(cfg, "p5_predictions")
    p5.write_text(json.dumps(to_p5_predictions(out), allow_nan=False))
    print(f"wrote P5 predictions mapping for {out['customer_id'].nunique()} customers -> {p5}")


if __name__ == "__main__":
    main()
