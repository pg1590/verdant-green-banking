"""Train + validate one propensity model per product.

    python -m src.models.train                     # full: CV grid search on train, eval on val
    python -m src.models.train --fast              # smoke run with fixed params
    python -m src.models.train --products GP04     # retrain some products; reports are merged

Protocol (no test-set access here — see run_test.py):
  * two model families per product: monotone LightGBM and a logistic model (log-transformed
    inputs). Each is tuned by stratified k-fold CV on TRAIN only; the family with the higher
    train-CV score ships (decided per product, never on VAL or TEST)
  * calibration via CalibratedClassifierCV(ensemble=False): the calibrator is fitted on
    out-of-fold TRAIN predictions, then ONE model is refitted on all of TRAIN. That same
    model is what SHAP explains (reason codes) and what the calibrator maps to a propensity.
  * VAL used only to report/compare the selected model, the other family and the baselines
  * reason-code percentile thresholds fitted on each product's eligible TRAIN rows and
    persisted in the bundle (`reason_dictionary`); validation and inference reuse them
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import evaluate as ev
from .baselines import SingleSignalRule, logistic_baseline, logistic_model
from .config import (bundle_stem, feature_list, load_config, load_reason_codes, model_version, monotone_list,
                     path)
from .data import X_y, load_tables, product_frame
from .reasons import explain, global_importance


def _lgbm(cfg: dict, product_code: str, seed: int, **params) -> LGBMClassifier:
    return LGBMClassifier(monotone_constraints=monotone_list(cfg, product_code), random_state=seed,
                          deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1, **params)


def _estimator(cfg: dict, code: str, family: str, seed: int, params: dict):
    if family == "lgbm":
        return _lgbm(cfg, code, seed, **params)
    return logistic_model(seed).set_params(**params)


def _tune(cfg: dict, code: str, family: str, X, y, cv, fast: bool) -> tuple[dict, float]:
    """(best params, mean train-CV score) for one model family. --fast skips the grid but
    still scores the fixed params, so the family comparison is always made on train CV."""
    tcfg, seed = cfg["training"], cfg["seed"]
    est, grid = _estimator(cfg, code, family, seed, {}), tcfg["param_grid" if family == "lgbm" else "logreg_grid"]
    if fast:
        params = dict(tcfg["fast_params"]) if family == "lgbm" else {}
        scores = cross_val_score(est.set_params(**params), X, y, scoring=tcfg["cv_scoring"], cv=cv)
        return params, float(scores.mean())
    gs = GridSearchCV(est, grid, scoring=tcfg["cv_scoring"], cv=cv).fit(X, y)
    return dict(gs.best_params_), float(gs.best_score_)


def sign_violations(cfg: dict, code: str, model) -> list[str]:
    """Features whose fitted logistic coefficient contradicts the declared monotone direction
    in model.yaml (LightGBM enforces the direction; logistic regression does not)."""
    if not isinstance(model, Pipeline):
        return []
    coef = model[-1].coef_[0]
    return [f for f, d, w in zip(feature_list(cfg, code), monotone_list(cfg, code), coef) if d * w < 0]


def fit_product(cfg: dict, code: str, train: pd.DataFrame, fast: bool = False) -> dict:
    """Tune LightGBM and the logistic candidate on TRAIN CV, ship the one with the higher CV
    score (cv_scoring), calibrate it, and fit its bootstrap ensemble for confidence."""
    seed, tcfg = cfg["seed"], cfg["training"]
    X, y = X_y(cfg, train, code)
    minority = int(min(y.sum(), len(y) - y.sum()))
    if minority < 2:
        raise ValueError(f"{code}: need at least two examples of each class; generate more customers")
    cv = StratifiedKFold(min(tcfg["cv_folds"], minority), shuffle=True, random_state=seed)

    tuned = {fam: _tune(cfg, code, fam, X, y, cv, fast) for fam in tcfg["families"]}
    family = max(tuned, key=lambda fam: tuned[fam][1])
    best_params, cv_score = tuned[family]

    method = "isotonic" if y.sum() >= tcfg["isotonic_min_positives"] else "sigmoid"

    def calibrate(fam):
        return CalibratedClassifierCV(_estimator(cfg, code, fam, seed, tuned[fam][0]), method=method,
                                      cv=cv, ensemble=False).fit(X, y)

    calibrated = calibrate(family)
    base = calibrated.calibrated_classifiers_[0].estimator  # the scoring model; explained by SHAP
    challengers = {fam: calibrate(fam) for fam in tuned if fam != family}

    rng = np.random.default_rng(seed)
    ensemble = []
    for i in range(tcfg["ensemble_size"]):
        idx = rng.integers(0, len(X), len(X))
        ensemble.append(_estimator(cfg, code, family, seed + i + 1, best_params).fit(X.iloc[idx], y[idx]))

    rule = SingleSignalRule(cfg["products"][code]["baseline_rule_feature"]).fit(X, y)
    logreg = logistic_baseline(seed).fit(X, y)

    return {
        "product_code": code, "model_version": model_version(cfg, code, family), "family": family,
        "features": feature_list(cfg, code), "monotone": monotone_list(cfg, code),
        "sign_violations": sign_violations(cfg, code, base),
        "reason_dictionary": load_reason_codes(cfg, X), "best_params": best_params,
        "cv_score": cv_score, "cv_scores": {fam: round(sc, 4) for fam, (_, sc) in tuned.items()},
        "calibration": method,
        "base": base, "calibrated": calibrated, "ensemble": ensemble, "challengers": challengers,
        "baseline_rule": rule, "baseline_logreg": logreg,
        "n_train": int(len(y)), "n_train_pos": int(y.sum()),
    }


def evaluate_product(cfg: dict, bundle: dict, val: pd.DataFrame, fig_dir) -> dict:
    code = bundle["product_code"]
    X, y = X_y(cfg, val, code)
    preds = {
        "B1_rule": bundle["baseline_rule"].predict_proba(X)[:, 1],
        "B2_logreg": bundle["baseline_logreg"].predict_proba(X)[:, 1],
        **{f"{fam}_calibrated": m.predict_proba(X)[:, 1] for fam, m in bundle["challengers"].items()},
        "selected_raw": bundle["base"].predict_proba(X)[:, 1],
        "selected_calibrated": bundle["calibrated"].predict_proba(X)[:, 1],
    }
    result = {"selected": bundle["family"], "cv_scores": bundle["cv_scores"],
              "sign_violations": bundle["sign_violations"], "models": {}}
    for name, p in preds.items():
        m = ev.metrics(y, p)
        for metric in ("roc_auc", "pr_auc"):
            m[f"{metric}_ci95"] = ev.bootstrap_ci(y, p, metric, n_boot=200, seed=cfg["seed"])
        result["models"][name] = m

    p_final = preds["selected_calibrated"]
    lift = ev.lift_table(y, p_final)
    result["lift"] = lift.to_dict(orient="records")
    ev.reliability_plot(y, {k: preds[k] for k in ("B2_logreg", "selected_raw", "selected_calibrated")},
                        fig_dir / f"{code}_reliability.png", f"{code} calibration (val)")
    ev.lift_plot(lift, fig_dir / f"{code}_lift.png", f"{code} lift by decile (val)")
    result["shap_importance"] = global_importance(bundle["base"], X).to_dict(orient="records")

    codes, _ = explain(bundle["base"], X, code, bundle["reason_dictionary"], cfg["reasons"]["max_codes"])
    result["reason_code_coverage"] = float(np.mean([len(c) > 0 for c in codes]))
    result["reason_code_counts"] = pd.Series([c for cs in codes for c in cs]).value_counts().to_dict()

    slices = []
    vdf = val.assign(_y=y, _p=p_final)
    for col in cfg["columns"]["fairness_slices"]:
        if col in vdf.columns:
            slices.append(ev.fairness_slices(vdf, col, "_y", "_p", seed=cfg["seed"]).assign(product_code=code))
    result["_fairness"] = pd.concat(slices) if slices else pd.DataFrame()
    return result


def input_hashes(cfg: dict) -> dict[str, str]:
    """sha1 (12 hex chars) of every input that determines a model: data tables and configs."""
    files = {key: path(cfg, key) for key in ("features", "labels", "splits")}
    files.update({name: cfg["_root"] / "config" / name
                  for name in ("model.yaml", "reason_codes.yaml", "settings.yaml", "products.yaml")})
    return {k: hashlib.sha1(p.read_bytes()).hexdigest()[:12] for k, p in files.items()}


def _merge_json(dest, update: dict) -> dict:
    """Keep other products' entries when only some products were retrained (--products)."""
    old = json.loads(dest.read_text()) if dest.exists() else {}
    return {**old, **update}


def reason_thresholds(bundle: dict) -> dict:
    """{code: fitted threshold} for the product's reason codes (frozen on its TRAIN cohort)."""
    return {c: d["threshold"] for c, d in bundle["reason_dictionary"].items()
            if bundle["product_code"] in d["products"]}


def _manifest(cfg: dict, bundle: dict, val_metrics: dict, hashes: dict) -> dict:
    keep = ("product_code", "model_version", "family", "features", "monotone", "sign_violations",
            "best_params", "cv_score", "cv_scores", "calibration", "n_train", "n_train_pos")
    return {**{k: bundle[k] for k in keep}, "reason_thresholds": reason_thresholds(bundle), "seed": cfg["seed"],
            "train_data_sha1": hashes["features"], "input_sha1": hashes,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "val_metrics": val_metrics["models"]["selected_calibrated"]}


def write_tables_md(results: dict, out_path) -> None:
    """Auto-generated metric tables, pasted/linked into reports/ml_eval.md."""
    lines = ["# Validation metrics (auto-generated by src/models/train.py)\n"]
    for code, r in results.items():
        lines += [f"\n## {code}\n",
                  "| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | ECE | Precision@10% | Lift@10% |",
                  "|---|---|---|---|---|---|---|"]
        for name, m in r["models"].items():
            lo, hi = m["roc_auc_ci95"]
            plo, phi = m["pr_auc_ci95"]
            lines.append(f"| {name} | {m['roc_auc']:.3f} ({lo:.3f}–{hi:.3f}) "
                         f"| {m['pr_auc']:.3f} ({plo:.3f}–{phi:.3f}) | {m['brier']:.4f} | {m['ece']:.4f} "
                         f"| {m['precision_at_10pct']:.3f} | {m['lift_at_10pct']:.2f} |")
        m = r["models"]["selected_calibrated"]
        cvs = ", ".join(f"{fam} {sc:.3f}" for fam, sc in r["cv_scores"].items())
        lines.append(f"\nSelected: **{r['selected']}** (train-CV average precision: {cvs}). "
                     f"n_val={m['n']}, positives={m['n_pos']}, base rate={m['base_rate']:.3f}, "
                     f"reason-code coverage={r['reason_code_coverage']:.1%}\n")
        if r["sign_violations"]:
            lines.append(f"Coefficient sign contradicts model.yaml direction: {', '.join(r['sign_violations'])}\n")
        lines.append(f"![reliability](figures/{code}_reliability.png) ![lift](figures/{code}_lift.png)\n")
        top = ", ".join(f"{d['feature']} ({d['mean_abs_shap']:.3f})" for d in r["shap_importance"][:5])
        lines.append(f"Top SHAP drivers: {top}\n")
    out_path.write_text("\n".join(lines))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fixed params, no grid search")
    ap.add_argument("--products", nargs="*", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    features, labels, splits = load_tables(cfg)
    hashes = input_hashes(cfg)
    models_dir, reports_dir = path(cfg, "models_dir"), path(cfg, "reports_dir")
    fig_dir = reports_dir / "figures"
    for d in (models_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    split_col = cfg["columns"]["split"]
    results, fairness, thresholds = {}, [], {}
    for code in args.products or list(cfg["products"]):
        df = product_frame(cfg, code, features, labels, splits)
        train, val = df[df[split_col] == "train"], df[df[split_col] == "val"]
        print(f"[{code}] train={len(train)} (pos={int(train[cfg['columns']['label']].sum())}) val={len(val)}")
        bundle = fit_product(cfg, code, train, fast=args.fast)
        thresholds[code] = reason_thresholds(bundle)
        r = evaluate_product(cfg, bundle, val, fig_dir)
        fairness.append(r.pop("_fairness"))
        results[code] = r
        stem = bundle_stem(cfg, code)
        for old in models_dir.glob(f"{stem}.*"):
            old.unlink()
        joblib.dump(bundle, models_dir / f"{stem}.pkl")
        (models_dir / f"{stem}.manifest.json").write_text(
            json.dumps(_manifest(cfg, bundle, r, hashes), indent=2, default=str))
        m = r["models"]
        print(f"[{code}] selected {bundle['family']} (train CV {bundle['cv_scores']}); val ROC-AUC "
              f"rule={m['B1_rule']['roc_auc']:.3f} logreg={m['B2_logreg']['roc_auc']:.3f} "
              f"selected={m['selected_calibrated']['roc_auc']:.3f} (calibration={bundle['calibration']})")

    metrics_path, thr_path = reports_dir / "metrics_val.json", reports_dir / "reason_code_thresholds.json"
    all_results = _merge_json(metrics_path, json.loads(json.dumps(results, default=str)))
    metrics_path.write_text(json.dumps(all_results, indent=2, default=str))
    thr_path.write_text(json.dumps(_merge_json(thr_path, json.loads(json.dumps(thresholds, default=str))),
                                   indent=2, default=str))
    fair_path, fair = reports_dir / "fairness_val.csv", pd.concat(fairness)
    if fair_path.exists() and args.products:
        old = pd.read_csv(fair_path)
        fair = pd.concat([old[~old["product_code"].isin(results)], fair], ignore_index=True)
    fair.sort_values(["product_code", "slice"], kind="stable").to_csv(fair_path, index=False)
    write_tables_md({k: all_results[k] for k in sorted(all_results)}, reports_dir / "ml_eval_tables.md")
    return results


if __name__ == "__main__":
    main()
