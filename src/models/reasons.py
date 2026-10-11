"""Reason codes: top positive SHAP contributors, mapped to the fixed dictionary, gated by evidence.

SHAP (log-odds) is computed on the SAME model that scores: native TreeSHAP for LightGBM,
exact linear SHAP for the logistic model. Calibration is a monotone map fitted on top of
it (train.py, ensemble=False), so a positive driver raises the calibrated propensity too.

At most one code per feature: the first code in config/reason_codes.yaml whose evidence
gate passes (so list the more specific code first), at most `max_codes` per row.
"""
from __future__ import annotations

import math
import operator
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}


def passes_gate(value, op: str, threshold) -> bool:
    if threshold is None or value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(OPS[op](value, threshold))


def codes_by_feature(dictionary: dict, product_code: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for code, d in dictionary.items():
        if product_code in d["products"]:
            out.setdefault(d["feature"], []).append(code)
    return out


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Per-feature contributions to the log-odds, shape (n_rows, n_features)."""
    if isinstance(model, Pipeline):
        # Logistic model: exact (interventional) linear SHAP, coef * (z - mean z). The scaler is
        # fitted on TRAIN, so the training mean of z is 0 and the background is the TRAIN mean.
        return np.asarray(model[:-1].transform(X), dtype=float) * model[-1].coef_[0]
    if hasattr(model, "booster_"):
        # LightGBM computes exact TreeSHAP contributions; the final column is the bias.
        return np.asarray(model.booster_.predict(X, pred_contrib=True))[:, :-1]
    import shap  # optional backend for non-LightGBM models
    with warnings.catch_warnings():  # shap warns that LGBM output may be a list; handled below
        warnings.simplefilter("ignore", UserWarning)
        sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):  # older shap: [class0, class1]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv


def explain(model, X: pd.DataFrame, product_code: str, dictionary: dict,
            max_codes: int = 3) -> tuple[list[list[str]], list[dict]]:
    """Return (reason_codes, evidence) per row of X."""
    sv = shap_values(model, X)
    feat_codes = codes_by_feature(dictionary, product_code)
    cols = list(X.columns)
    values = X.to_numpy(dtype=float)
    all_codes, all_evidence = [], []
    for i in range(len(X)):
        codes, evidence = [], {}
        for j in np.argsort(-sv[i], kind="stable"):
            if sv[i, j] <= 0 or len(codes) >= max_codes:
                break
            for code in feat_codes.get(cols[j], []):
                d = dictionary[code]
                if passes_gate(values[i, j], d["op"], d["threshold"]):
                    codes.append(code)
                    v = values[i, j]
                    evidence[d["evidence_key"]] = int(v) if float(v).is_integer() else round(float(v), 2)
                    break
        all_codes.append(codes)
        all_evidence.append(evidence)
    return all_codes, all_evidence


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    sv = shap_values(model, X)
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
