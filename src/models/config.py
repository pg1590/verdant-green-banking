"""Config loading for the propensity models. All column names come from config/model.yaml.

Seed, bootstrap size and excluded features are shared settings (config/settings.yaml);
they are read from there, never repeated in model.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.common import config as common
from src.schemas import tables as T

ROOT = common.ROOT
MODEL_CONFIG = ROOT / "config" / "model.yaml"
# Never model inputs, whatever model.yaml says (D1 hidden traits, D9 age, FR2 grading column).
FORBIDDEN_INPUTS = {*T.NON_MODEL_COLUMNS, *T.HIDDEN_COLUMNS, "true_category", "took_up"}


def load_config(path: Path | str = MODEL_CONFIG) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    s = common.settings()
    cfg["_root"] = ROOT
    cfg["seed"] = s["seed"]
    cfg["training"]["ensemble_size"] = s["model"]["bootstrap_models"]
    cfg["excluded_features"] = list(s["model"]["excluded_features"])
    check_feature_policy(cfg)
    return cfg


def check_feature_policy(cfg: dict) -> None:
    """Fail loudly if a forbidden column (or an encoding of one) is configured as a model input."""
    forbidden = FORBIDDEN_INPUTS | set(cfg["excluded_features"])
    for code, product in cfg["products"].items():
        for feat in product["features"]:
            bad = ({feat} | set(sources(cfg, feat))) & forbidden
            assert not bad, f"{code}: {sorted(bad)} must never be a model input (settings.model / tables.py)"
            assert feat not in cfg["slices"], f"{code}: {feat} is a fairness slice, not a model input"


def sources(cfg: dict, feature: str) -> list[str]:
    """Feature-store columns a model feature is computed from (itself if not derived)."""
    spec = cfg["derived"].get(feature)
    if spec is None:
        return [feature]
    return [spec["source"]] if "source" in spec else [spec["numerator"], spec["denominator"]]


def path(cfg: dict, key: str) -> Path:
    """Resolve a `paths:` entry relative to the repo root."""
    return cfg["_root"] / cfg["paths"][key]


def load_reason_codes(cfg: dict, train=None) -> dict:
    """Compile config/reason_codes.yaml into P3's gate dictionary:
    {code: {feature, products, op, threshold, evidence_key, label}}, in file order.

    `feature` is the MODEL column (model_feature). Percentile thresholds are fitted on
    `train` (each product's training cohort) and persisted in the model bundle; without
    `train` they are None and the code can never fire."""
    with open(cfg["_root"] / cfg["reasons"]["dictionary"]) as f:
        raw = yaml.safe_load(f)
    if "reason_codes" not in raw:
        return raw
    out = {}
    for code, spec in raw["reason_codes"].items():
        feature = spec.get("model_feature", spec["feature"])
        products = [p for p in spec["products"] if feature in feature_list(cfg, p)]
        if not products:
            continue
        threshold = spec["threshold"]
        value, op = threshold["value"], ">="
        if threshold["type"] == "percentile":
            values = train[feature].dropna() if train is not None and feature in train else []
            value = float(values.quantile(value / 100)) if len(values) else None
        elif threshold["type"] == "flag":
            if feature == "epc_rating_ord":
                value = min("ABCDEFG".index(v) + 1 for v in value)
            elif feature == "income_band_ord":
                value = min(int(v[1:]) for v in value)
            elif feature in cfg["derived"]:  # 0/1 encoding of the listed categories
                op, value = "==", 1
            else:
                op, value = "==", int(value) if isinstance(value, bool) else value
        out[code] = {"feature": feature, "products": products, "op": op, "threshold": value,
                     "evidence_key": spec.get("evidence_key", spec["feature"]),
                     "label": spec.get("customer_text", code)}
    return out


def model_version(cfg: dict, product_code: str, family: str) -> str:
    """e.g. gp01_logreg_v3: product, selected model family, config version."""
    return f"{product_code.lower()}_{family}_{cfg['version']}"


def bundle_stem(cfg: dict, product_code: str) -> str:
    """File name (without extension) of a product's model bundle and manifest in models_dir;
    independent of the selected family so inference can find it."""
    return f"{product_code.lower()}_propensity_{cfg['version']}"


def feature_list(cfg: dict, product_code: str) -> list[str]:
    return list(cfg["products"][product_code]["features"])


def monotone_list(cfg: dict, product_code: str) -> list[int]:
    return [int(v) for v in cfg["products"][product_code]["features"].values()]
