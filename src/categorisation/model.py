"""FR2 main model: TF-IDF (char n-grams) on merchant + description, plus MCC, channel
and amount -> LogisticRegression. Rules that recognise a transaction decide; the model
only fills rule fallbacks (see categorise()). Owner: P2.

    python -m src.categorisation.model            (stage 2 of scripts/run_pipeline.py)

Output: data/processed/transactions_categorised.parquet = the input transactions
(minus true_category) plus
    category             one of CATEGORIES (the FR2 answer)
    category_confidence  1.0 when a strong rule decided, else the model's probability of that category
    category_source      "rule:<RULE>" (see src/categorisation/rules.py) or "model"
    is_ev_charging       Fuel row that is EV charging (P1: VEHICLE_FUEL_CONFLICT uses road fuel only)
    is_dual_fuel         Utilities row that is a dual-fuel energy bill (P1: D6 apportioning)

Leakage: the model trains only on transactions of TRAIN-split customers, so the
spend features of validation/test customers come from a categoriser that never saw
their labels. true_category is a training/evaluation label only, never a feature.
Evaluation (merchant-brand split, D4) lives in src/categorisation/evaluate.py.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.categorisation import rules
from src.common import config
from src.schemas.tables import CATEGORIES

MODEL_VERSION = "fr2_tfidf_lr_v2"
# The model replaces a rule fallback (ambiguous MCC / default Other) only above this probability.
MODEL_MIN_CONFIDENCE = 0.9
MODEL_PATH = config.MODELS_DIR / "categoriser.joblib"
OUTPUT_PATH = config.PROCESSED_DIR / "transactions_categorised.parquet"
OUTPUT_COLUMNS = ("category", "category_confidence", "category_source", "is_ev_charging", "is_dual_fuel")
_KEY_COLS = ["text", "mcc_tok", "channel", "log_amount"]
# Everything the rules or the model read from a transaction (amount via _amount_bin).
_RAW_COLS = ["merchant_name", "description", "mcc", "channel", "amount"]


@dataclass
class Categoriser:
    pipeline: Pipeline
    version: str = MODEL_VERSION
    trained_on: dict = field(default_factory=dict)


def model_inputs(txns: pd.DataFrame) -> pd.DataFrame:
    """The model's view of a transaction. Never includes true_category."""
    noise = rules.ruleset().noise
    merchant = txns["merchant_name"].map(lambda s: rules.clean_text(s, noise))
    description = txns["description"].map(rules.clean_text)
    mcc = txns["mcc"] if "mcc" in txns else pd.Series(pd.NA, index=txns.index)
    return pd.DataFrame({
        "text": (merchant + " | " + description).to_numpy(),
        "mcc_tok": ["MCC_NONE" if pd.isna(v) else f"MCC_{int(v)}" for v in mcc],
        "channel": txns["channel"].astype("string").fillna("unknown").to_numpy(),
        "log_amount": _amount_bin(txns["amount"]),
    }, index=txns.index)


def _amount_bin(amount: pd.Series) -> np.ndarray:
    # rounded so identical merchant strings with similar amounts collapse together
    return np.round(np.log1p(amount.abs().astype(float).fillna(0).to_numpy()), 1)


def _distinct_inputs(txns: pd.DataFrame, *extra_keys) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per distinct raw input (plus `extra_keys`), the position in txns each was taken
    from, and each transaction's row number in it.

    Rows that share a raw input get the same rules result and model probabilities, so the
    expensive work (text cleaning, rules, predict_proba) runs on the distinct rows only and
    is copied back with `result[codes]`, never a merge.
    """
    cols = [c for c in _RAW_COLS if c in txns]
    keys = [txns[c] for c in cols if c != "amount"]
    first, codes = rules.group_codes(*keys, _amount_bin(txns["amount"]), *extra_keys)
    distinct = txns.iloc[first, [txns.columns.get_loc(c) for c in cols]].reset_index(drop=True)
    return distinct, first, codes


def build_pipeline(seed: int) -> Pipeline:
    features = ColumnTransformer([
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                  lowercase=False), "text"),
        ("codes", OneHotEncoder(handle_unknown="ignore"), ["mcc_tok", "channel"]),
        ("amount", StandardScaler(), ["log_amount"]),
    ])
    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def _dedupe(x: pd.DataFrame, y: pd.Series | None = None,
            counts: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse identical model inputs (and labels) into one weighted row.

    `counts` gives how many transactions each row of `x` already stands for (default 1).
    """
    df = x[_KEY_COLS].copy()
    cols = list(_KEY_COLS)
    if y is not None:
        df["_y"] = y.to_numpy()
        cols.append("_y")
    df["_w"] = np.ones(len(df), dtype=np.int64) if counts is None else counts
    grouped = df.groupby(cols, dropna=False, sort=True)["_w"].sum().reset_index()
    return grouped, grouped.pop("_w").to_numpy()


def train(txns: pd.DataFrame, seed: int | None = None) -> Categoriser:
    """Fit on labelled transactions (needs true_category)."""
    seed = config.settings()["seed"] if seed is None else seed
    labels = txns["true_category"]
    bad = set(labels.unique()) - set(CATEGORIES)
    if bad:
        raise ValueError(f"true_category has values outside CATEGORIES: {sorted(bad)}")
    distinct, first, codes = _distinct_inputs(txns, labels)
    rows, weights = _dedupe(model_inputs(distinct), labels.iloc[first], np.bincount(codes))
    pipe = build_pipeline(seed)
    pipe.fit(rows[_KEY_COLS], rows["_y"], clf__sample_weight=weights)
    return Categoriser(pipe, trained_on={"transactions": int(len(txns)), "distinct_inputs": int(len(rows)),
                                         "classes": sorted(labels.unique())})


def _model_outputs(model: Categoriser, x: pd.DataFrame) -> pd.DataFrame:
    """pred, proba and p_<category> for each row of model inputs `x`."""
    proba = model.pipeline.predict_proba(x[_KEY_COLS])
    classes = list(model.pipeline.classes_)
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    out["pred"] = np.asarray(classes)[proba.argmax(axis=1)]
    out["proba"] = proba.max(axis=1)
    for c in CATEGORIES:  # a class absent from training has probability 0
        if f"p_{c}" not in out:
            out[f"p_{c}"] = 0.0
    return out[["pred", "proba"] + [f"p_{c}" for c in CATEGORIES]]


def model_only(model: Categoriser, txns: pd.DataFrame) -> pd.DataFrame:
    """Model prediction, its probability and every class probability (p_<category>), ignoring the rules."""
    distinct, _, codes = _distinct_inputs(txns)
    out = _model_outputs(model, model_inputs(distinct)).take(codes)
    out.index = txns.index
    return out


def categorise(model: Categoriser, txns: pd.DataFrame,
               min_model_confidence: float = MODEL_MIN_CONFIDENCE) -> pd.DataFrame:
    """Final FR2 output.

    Every rule that recognises the transaction decides: the strong rules (steps 1-4 in
    config/categories.yaml) and the merchant/description keywords (step 5). The model
    only replaces a rule fallback (ambiguous MCC or default "Other", steps 6-7), and only
    when it is at least `min_model_confidence` sure. Letting the model override keyword
    rules made the categoriser worse than the rules alone on unseen merchants and
    missing MCCs (docs/categorisation.md).

    Decided once per distinct raw input, then copied to every transaction by position.
    """
    distinct, _, codes = _distinct_inputs(txns)
    r = rules.explain_rules(distinct)
    m = _model_outputs(model, model_inputs(distinct))
    recognised = (r["strong"] | (r["rule"] == rules.KEYWORD)).to_numpy()
    use_model = ~recognised & (m["proba"].to_numpy() >= min_model_confidence)
    category = np.where(use_model, m["pred"], r["category"])
    # 1.0 when a strong rule decided, otherwise the model's probability of the chosen category
    probs = m[[f"p_{c}" for c in CATEGORIES]].to_numpy(dtype=float)
    p_chosen = probs[np.arange(len(category)), pd.Index(CATEGORIES).get_indexer(category)]
    decided = {
        "category": category,
        "category_confidence": np.where(r["strong"].to_numpy(), 1.0, p_chosen).astype(float),
        "category_source": np.where(use_model, "model", "rule:" + r["rule"].astype(str)),
        "is_ev_charging": r["is_ev_charging"].to_numpy() & (category == "Fuel"),
        "is_dual_fuel": r["is_dual_fuel"].to_numpy() & (category == "Utilities"),
    }
    out = pd.DataFrame({c: np.asarray(v)[codes] for c, v in decided.items()}, index=txns.index)
    assert set(out["category"].unique()) <= set(CATEGORIES)
    return out


def predict(model: Categoriser, txns: pd.DataFrame) -> pd.Series:
    """One category per transaction (rules first, model for rule fallbacks)."""
    return categorise(model, txns)["category"]


def save(model: Categoriser, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path: Path = MODEL_PATH) -> Categoriser:
    return joblib.load(path)


def default_data_dir() -> Path:
    """P1's generated data if present, otherwise the committed mock sample."""
    return config.RAW_DIR if (config.RAW_DIR / "transactions.parquet").exists() else config.SAMPLE_DIR


def find_splits(data_dir: Path) -> Path | None:
    """P1's customer splits: next to the transactions (mock sample) or in data/processed (P3's path)."""
    for path in (data_dir / "splits.parquet", config.PROCESSED_DIR / "splits.parquet"):
        if path.exists():
            return path
    return None


def split_customers(data_dir: Path, split: str) -> set[str]:
    """Customer ids in one split. Fails loudly: training on every customer would leak test labels."""
    path = find_splits(data_dir)
    if path is None:
        raise FileNotFoundError(
            f"no splits.parquet in {data_dir} or {config.PROCESSED_DIR}; the categoriser must train on "
            "TRAIN-split customers only (pass --allow-no-splits to train on everyone, e.g. for a demo)")
    splits = pd.read_parquet(path)
    return set(splits.loc[splits["split"] == split, "customer_id"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="FR2: categorise transactions")
    p.add_argument("--data-dir", type=Path, default=None, help="folder with transactions/splits parquet")
    p.add_argument("--out", type=Path, default=OUTPUT_PATH)
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--allow-no-splits", action="store_true", help="train on all customers if no splits file")
    args = p.parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    if data_dir == config.SAMPLE_DIR:
        print(f"    note: using the MOCK sample in {data_dir}; do not report these metrics")

    txns = pd.read_parquet(data_dir / "transactions.parquet")
    if args.allow_no_splits and find_splits(data_dir) is None:
        fit_on = txns
    else:
        fit_on = txns[txns["customer_id"].isin(split_customers(data_dir, "train"))]
    if fit_on.empty:
        raise ValueError("no transactions belong to train-split customers")
    model = train(fit_on)
    n_fit = len(fit_on)
    del fit_on
    save(model, args.model_path)

    result = categorise(model, txns)
    # change txns in place: drop() or concat would copy every transaction once more
    out = txns
    if "true_category" in out:
        del out["true_category"]
    for c in OUTPUT_COLUMNS:
        out[c] = result[c].to_numpy()
    del result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    by_model = (out["category_source"] == "model").mean()
    print(f"    categorised {len(out):,} transactions ({1 - by_model:.0%} by rules, {by_model:.0%} by the model) "
          f"-> {args.out}")
    print(f"    model trained on {n_fit:,} train-customer transactions -> {args.model_path}")


if __name__ == "__main__":
    main()
