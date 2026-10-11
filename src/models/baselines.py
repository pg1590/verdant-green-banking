"""Baselines the shipped model must beat (brief: "compare against a simple baseline"), and the
logistic candidate model that competes with LightGBM in train.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class SingleSignalRule:
    """B1: 'feature > train p75' -> predict the train take-up rate of that group.

    Mirrors how a business would target without ML (e.g. 'fuel spend in top quartile').
    Missing values fall into the low group."""

    def __init__(self, feature: str, quantile: float = 0.75):
        self.feature, self.quantile = feature, quantile

    def fit(self, X: pd.DataFrame, y) -> "SingleSignalRule":
        y = np.asarray(y)
        x = X[self.feature]
        self.threshold_ = float(x.quantile(self.quantile))
        hi = (x > self.threshold_).to_numpy()
        self.p_hi_ = float(y[hi].mean()) if hi.any() else float(y.mean())
        self.p_lo_ = float(y[~hi].mean()) if (~hi).any() else float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        hi = (X[self.feature] > self.threshold_).to_numpy()
        p = np.where(hi, self.p_hi_, self.p_lo_)
        return np.column_stack([1 - p, p])


def logistic_baseline(seed: int):
    """B2: median-impute + standardise + L2 logistic regression (no class reweighting,
    so outputs stay interpretable as probabilities)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=seed))


def signed_log1p(X):
    """Compress heavy-tailed spend / kWh / km values; 0/1 flags and ordinals keep their order."""
    return np.sign(X) * np.log1p(np.abs(X))


def logistic_model(seed: int, C: float = 1.0):
    """Candidate model (not a baseline): B2 plus a log transform of skewed inputs. Selected per
    product when it beats LightGBM in train CV (train.py); explained by exact linear SHAP."""
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(signed_log1p),
                         StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
