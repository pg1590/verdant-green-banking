"""The only place that reads P1's tables (paths in config/model.yaml)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.records import to_records
from src.recommend.eligibility import check_eligibility, load_catalogue

from .config import feature_list, path


def load_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = encode(cfg, pd.read_parquet(path(cfg, "features")))
    labels = pd.read_parquet(path(cfg, "labels"))
    splits = pd.read_parquet(path(cfg, "splits"))
    check_splits(cfg, splits)
    check_same_dataset(cfg, features, labels, splits)
    return features, labels, splits


def encode(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add P3's numeric encodings (`derived:`) and fairness slices (`slices:`) from model.yaml
    without changing P1's persisted feature store. Idempotent. A missing source value stays
    NaN so LightGBM learns from missingness; a missing source column is skipped (reported
    later by missing_features)."""
    out = df.copy()
    for name, spec in cfg["derived"].items():
        if "numerator" in spec:
            if {spec["numerator"], spec["denominator"]} <= set(out.columns):
                num = out[spec["numerator"]].to_numpy(dtype=float, na_value=np.nan)
                den = out[spec["denominator"]].to_numpy(dtype=float, na_value=np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[name] = np.where(den > 0, num / den, np.nan)
            continue
        if spec["source"] not in out.columns:
            continue
        src = out[spec["source"]].astype(object)
        if "map" in spec:
            out[name] = src.map(spec["map"]).astype(float)
        else:
            out[name] = np.where(src.isna(), np.nan, src.isin(spec["in"]).astype(float))
    for name, spec in cfg["slices"].items():
        if spec["source"] in out.columns:
            out[name] = pd.cut(out[spec["source"]], spec["bins"], labels=spec["labels"]).astype(str)
    return out


def encoded_columns(cfg: dict) -> list[str]:
    """Columns encode() adds; drop them before handing customer rows to P5."""
    return [*cfg["derived"], *cfg["slices"]]


def check_splits(cfg: dict, splits: pd.DataFrame) -> None:
    """Customer-level split integrity: each customer in exactly one split."""
    c = cfg["columns"]
    assert not splits[c["id"]].duplicated().any(), "customer appears in more than one split row"
    assert set(splits[c["split"]].unique()) <= {"train", "val", "test"}, "unexpected split names"


def check_same_dataset(cfg: dict, features: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame,
                       min_overlap: float = 0.99) -> None:
    """Features, labels and splits must describe the same customers (a stale or wrong path
    would otherwise inner-join silently down to a handful of rows). P1's fixed judge-scenario
    customers (config/scenarios.yaml) have no split or labels by design, so they are left out;
    otherwise any run below ~1,290 customers fails the 99% overlap."""
    from src.data_gen.generate import scenarios
    c = cfg["columns"]
    assert not features[c["id"]].duplicated().any(), "feature table has duplicate customer ids"
    f_ids = set(features[c["id"]].astype(str)) - set(scenarios())
    s_ids = set(splits[c["id"]].astype(str))
    assert len(f_ids & s_ids) >= min_overlap * len(f_ids), (
        f"only {len(f_ids & s_ids)}/{len(f_ids)} feature customers have a split: features and splits "
        f"come from different datasets? ({path(cfg, 'features')} vs {path(cfg, 'splits')})")
    for code in cfg["products"]:
        lab = labels.loc[labels[c["product"]] == code]
        assert len(lab), f"labels have no rows for {code}"
        assert not lab[c["id"]].duplicated().any(), f"{code}: duplicate labels per customer"
        l_ids = set(lab[c["id"]].astype(str))
        assert len(f_ids & l_ids) >= min_overlap * len(f_ids), (
            f"{code}: only {len(f_ids & l_ids)}/{len(f_ids)} feature customers have a label "
            f"({path(cfg, 'features')} vs {path(cfg, 'labels')})")
        assert set(lab[c["label"]].unique()) <= {0, 1}, f"{code}: took_up must be 0/1"


def missing_features(cfg: dict, df: pd.DataFrame, product_code: str) -> list[str]:
    return [f for f in feature_list(cfg, product_code) if f not in df.columns]


def numeric(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Model matrix: bool / nullable bool / Int64 -> float, missing -> NaN."""
    return pd.DataFrame({f: df[f].to_numpy(dtype=float, na_value=np.nan) for f in features}, index=df.index)


def customer_snapshots(cfg: dict, df: pd.DataFrame) -> list[dict]:
    """Rows of `df` as customer snapshots carrying every field P5's catalogue rules read,
    exported through to_records() as P5 requires."""
    fields = {cfg["columns"]["id"], cfg["columns"]["conflict_flags"], cfg["columns"]["data_flags"]}
    for product in load_catalogue()["products"].values():
        for rule in product["rules"]:
            fields |= {rule["field"], rule.get("when", {}).get("field"), rule.get("unless_present")}
    cols = [f for f in fields if f and f in df.columns]
    return to_records(df[cols])


def eligible_mask(cfg: dict, df: pd.DataFrame, product_code: str) -> pd.Series:
    """Model population = P5's catalogue eligibility with purpose='training' (ignores
    marketing consent). Customers outside it are still scored, but are excluded from
    training and from reported metrics."""
    catalogue = load_catalogue()
    flags = [check_eligibility(s, product_code, catalogue, purpose="training").eligible_to_show
             for s in customer_snapshots(cfg, df)]
    return pd.Series(flags, index=df.index, dtype=bool)


def product_frame(cfg: dict, product_code: str, features: pd.DataFrame,
                  labels: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Features + this product's label + split, restricted to the model population."""
    c = cfg["columns"]
    features = encode(cfg, features)
    lab = labels.loc[labels[c["product"]] == product_code, [c["id"], c["label"]]]
    df = features.merge(lab, on=c["id"], how="inner").merge(splits[[c["id"], c["split"]]], on=c["id"])
    missing = missing_features(cfg, df, product_code)
    assert not missing, f"{product_code}: feature columns missing from feature table: {missing}"
    return df[eligible_mask(cfg, df, product_code)].reset_index(drop=True)


def X_y(cfg: dict, df: pd.DataFrame, product_code: str):
    return numeric(df, feature_list(cfg, product_code)), df[cfg["columns"]["label"]].astype(int).to_numpy()
