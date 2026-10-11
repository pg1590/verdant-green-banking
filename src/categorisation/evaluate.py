"""FR2 evaluation -> reports/categorisation_eval.md and .json (part of report G). Owner: P2.

    python -m src.categorisation.evaluate                  (validation customers)
    python -m src.categorisation.evaluate --split test     (score the test set ONCE, at the end)

Setup (D4):
- Fit on TRAIN-split customers only, after removing a held-out set of merchant brands
  (about 25% of the brands in each category, always leaving at least one).
- Score the chosen split's customers, reporting brands SEEN in training separately from
  brands NEVER seen. The seen-brand numbers are what a transaction-level split would show;
  the unseen-brand numbers are the honest ones.
- Three systems: the rules baseline, the model alone, and the final categoriser
  (strong rules override the model).
- Plus an independent, hand-labelled set (tests/categorisation/fixtures/) written without
  reference to P1's generator, to check the categoriser is not just learning its templates.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.categorisation import model as fr2
from src.categorisation import rules
from src.common import config
from src.schemas.tables import CATEGORIES

REPORT_MD = config.REPORTS_DIR / "categorisation_eval.md"
REPORT_JSON = config.REPORTS_DIR / "categorisation_eval.json"
HAND_LABELLED = config.ROOT / "tests" / "categorisation" / "fixtures" / "hand_labelled_transactions.csv"
SYSTEMS = ("rules", "model", "final")


def brand(merchant: str) -> str:
    """Merchant brand = first word of the cleaned merchant name ("SHELL 4421 LEEDS" -> "SHELL")."""
    cleaned = rules.clean_text(merchant, rules.ruleset().noise)
    return cleaned.split()[0] if cleaned else "UNKNOWN"


def holdout_brands(txns: pd.DataFrame, frac: float, seed: int) -> set[str]:
    """Pick ~frac of the brands in each category to hide from training (keep >= 1 per category)."""
    b = txns["merchant_name"].map(brand)
    majority = txns.assign(_b=b).groupby("_b")["true_category"].agg(lambda s: s.mode().iloc[0])
    rng = np.random.default_rng(seed)
    held: set[str] = set()
    for cat in CATEGORIES:
        brands = sorted(majority[majority == cat].index)
        n = min(int(round(frac * len(brands))), len(brands) - 1)
        if n > 0:
            held.update(rng.choice(brands, size=n, replace=False).tolist())
    return held


def scores(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [c for c in CATEGORIES if (y_true == c).any()]
    per_class = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
        if len(y_true) else float("nan"),
        "per_class_f1": {c: float(f) for c, f in zip(present, per_class)},
        "support": {c: int((y_true == c).sum()) for c in present},
    }


def predictions(model: fr2.Categoriser, txns: pd.DataFrame) -> pd.DataFrame:
    final = fr2.categorise(model, txns)
    return pd.DataFrame({
        "rules": rules.categorise_rules(txns),
        "model": fr2.model_only(model, txns)["pred"],
        "final": final["category"],
        "source": final["category_source"],
    }, index=txns.index)


def boundary_slices(txns: pd.DataFrame) -> dict[str, pd.Series]:
    """Awkward cases the judges will look at, defined from the data, not from predictions."""
    rs = rules.ruleset()
    merchant = txns["merchant_name"].map(lambda s: rules.clean_text(s, rs.noise))
    fuel_kw = dict((c, p) for c, p, _ in rs.keyword_lists)["Fuel"]
    is_fuel_brand = merchant.map(lambda m: bool(fuel_kw and fuel_kw.search(m)))
    is_supplier = merchant.map(lambda m: bool(rs.suppliers and rs.suppliers.search(m)))
    mcc = txns["mcc"]
    ev = (mcc == 5552).fillna(False) | merchant.map(lambda m: bool(rs.ev_kw and rs.ev_kw.search(m)))
    y = txns["true_category"]
    return {
        "missing MCC": mcc.isna(),
        "EV charging": ev & (y == "Fuel"),
        "fuel-station shop purchases": is_fuel_brand & (y == "Other"),
        "energy-supplier bills": is_supplier & y.isin(["Electricity", "Gas", "Utilities"]),
        "dual-fuel bills": is_supplier & (y == "Utilities"),
    }


def evaluate(txns: pd.DataFrame, train_ids: set[str] | None, eval_ids: set[str] | None,
             seed: int, holdout_frac: float = 0.25) -> dict:
    train_pool = txns if train_ids is None else txns[txns["customer_id"].isin(train_ids)]
    eval_set = txns if eval_ids is None else txns[txns["customer_id"].isin(eval_ids)]
    held = holdout_brands(train_pool, holdout_frac, seed)
    fit_on = train_pool[~train_pool["merchant_name"].map(brand).isin(held)]
    model = fr2.train(fit_on, seed=seed)

    preds = predictions(model, eval_set)
    y = eval_set["true_category"]
    unseen = eval_set["merchant_name"].map(brand).isin(held)
    groups = {"all": pd.Series(True, index=eval_set.index), "seen brands": ~unseen, "unseen brands": unseen}
    result = {
        "model_version": fr2.MODEL_VERSION,
        "train_transactions": int(len(fit_on)),
        "eval_transactions": int(len(eval_set)),
        "heldout_brands": sorted(held),
        "metrics": {g: {s: scores(y[m], preds.loc[m, s]) for s in SYSTEMS}
                    for g, m in groups.items() if m.any()},
        "slices": {},
        "confusion": {},
        "final_source_mix": preds["source"].value_counts(normalize=True).round(4).to_dict(),
    }
    for name, mask in boundary_slices(eval_set).items():
        if mask.any():
            result["slices"][name] = {s: scores(y[mask], preds.loc[mask, s])["accuracy"] for s in SYSTEMS}
            result["slices"][name]["n"] = int(mask.sum())
    for g in ("all", "unseen brands"):
        m = groups[g]
        if m.any():
            cm = confusion_matrix(y[m], preds.loc[m, "final"], labels=list(CATEGORIES))
            result["confusion"][g] = cm.tolist()

    if HAND_LABELLED.exists():
        hand = load_hand_labelled()
        hp = predictions(model, hand)
        result["hand_labelled"] = {s: scores(hand["true_category"], hp[s]) for s in SYSTEMS}
        cols = ["txn_id", "merchant_name", "description", "mcc", "true_category", "final", "source", "note"]
        wrong = hand.assign(final=hp["final"], source=hp["source"])
        wrong = wrong.loc[wrong["final"] != wrong["true_category"], cols]
        result["hand_labelled_errors"] = [
            {k: None if pd.isna(v) else (int(v) if k == "mcc" else v) for k, v in row.items()}
            for row in wrong.to_dict(orient="records")
        ]
    return result


def load_hand_labelled(path: Path = HAND_LABELLED) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"mcc": "Int64"}, keep_default_na=False, na_values={"mcc": [""]})
    df["date"] = pd.to_datetime(df["date"])
    return df


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def to_markdown(r: dict, data_dir: Path, split: str) -> str:
    mock = data_dir.resolve() == config.SAMPLE_DIR.resolve()
    lines = [
        "# FR2 categorisation evaluation",
        "",
        f"Generated {date.today().isoformat()} by `python -m src.categorisation.evaluate --split {split}` "
        f"on `{data_dir.relative_to(config.ROOT) if data_dir.is_relative_to(config.ROOT) else data_dir}`. "
        f"Model `{r['model_version']}`.",
        "",
    ]
    if mock:
        lines += ["> **MOCK DATA.** These numbers come from the schema-correct mock sample "
                  "(`src/data_gen/mock.py`), which has only a few merchant templates. "
                  "They show the pipeline works; do not report them as results.", ""]
    lines += [
        f"Trained on {r['train_transactions']:,} transactions of train-split customers, excluding "
        f"{len(r['heldout_brands'])} held-out brands ({', '.join(r['heldout_brands']) or 'none'}). "
        f"Scored on {r['eval_transactions']:,} transactions of `{split}` customers.",
        "",
        "## Headline: macro-F1 / accuracy",
        "",
        "| Transactions | n | Rules baseline | Model only | Final (rules + model) |",
        "|---|---|---|---|---|",
    ]
    for g, by_sys in r["metrics"].items():
        cells = [f"{_pct(by_sys[s]['macro_f1'])} / {_pct(by_sys[s]['accuracy'])}" for s in SYSTEMS]
        lines.append(f"| {g} | {by_sys['final']['n']:,} | " + " | ".join(cells) + " |")
    lines += ["", "Seen brands approximate a transaction-level split; unseen brands are the honest test (D4).", ""]

    lines += ["## Per-class F1 (all transactions)", "", "| Category | Support | Rules | Model | Final |",
              "|---|---|---|---|---|"]
    allm = r["metrics"]["all"]
    for c in CATEGORIES:
        if c in allm["final"]["per_class_f1"]:
            lines.append(f"| {c} | {allm['final']['support'][c]:,} | "
                         + " | ".join(_pct(allm[s]["per_class_f1"][c]) for s in SYSTEMS) + " |")

    lines += ["", "## Boundary cases (accuracy)", "", "| Slice | n | Rules | Model | Final |", "|---|---|---|---|---|"]
    for name, v in r["slices"].items():
        lines.append(f"| {name} | {v['n']:,} | " + " | ".join(_pct(v[s]) for s in SYSTEMS) + " |")

    for g, cm in r["confusion"].items():
        lines += ["", f"## Confusion matrix, final categoriser ({g})", "",
                  "Rows are the true category, columns the predicted one.", "",
                  "| true \\ pred | " + " | ".join(CATEGORIES) + " |",
                  "|---" * (len(CATEGORIES) + 1) + "|"]
        for c, row in zip(CATEGORIES, cm):
            lines.append(f"| **{c}** | " + " | ".join(str(v) if v else "·" for v in row) + " |")

    mix = ", ".join(f"{k}: {_pct(v)}" for k, v in r["final_source_mix"].items())
    lines += ["", f"Who decided the final category: {mix}.", ""]

    if "hand_labelled" in r:
        h = r["hand_labelled"]
        lines += ["## Independent hand-labelled set", "",
                  f"{h['final']['n']} transactions written by hand, independently of P1's generator, "
                  "covering the boundary cases (`tests/categorisation/fixtures/hand_labelled_transactions.csv`).", "",
                  "| | Rules | Model | Final |", "|---|---|---|---|",
                  "| macro-F1 | " + " | ".join(_pct(h[s]["macro_f1"]) for s in SYSTEMS) + " |",
                  "| accuracy | " + " | ".join(_pct(h[s]["accuracy"]) for s in SYSTEMS) + " |", "",
                  "Errors of the final categoriser:", "",
                  "| txn | merchant | description | MCC | true | predicted | decided by | note |",
                  "|---|---|---|---|---|---|---|---|"]
        for e in r["hand_labelled_errors"]:
            mcc = "" if e["mcc"] is None else e["mcc"]
            lines.append(f"| {e['txn_id']} | {e['merchant_name']} | {e['description']} | {mcc} | "
                         f"{e['true_category']} | {e['final']} | {e['source']} | {e['note'] or ''} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="FR2 evaluation")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--holdout-frac", type=float, default=0.25)
    args = p.parse_args(argv)
    data_dir = args.data_dir or fr2.default_data_dir()
    txns = pd.read_parquet(data_dir / "transactions.parquet")
    train_ids = fr2.split_customers(data_dir, "train")
    eval_ids = fr2.split_customers(data_dir, args.split)
    r = evaluate(txns, train_ids, eval_ids, seed=config.settings()["seed"], holdout_frac=args.holdout_frac)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(to_markdown(r, data_dir, args.split))
    REPORT_JSON.write_text(json.dumps(r, indent=2, default=str))
    f = r["metrics"]["all"]["final"]
    print(f"FR2 {args.split}: final macro-F1 {_pct(f['macro_f1'])}, accuracy {_pct(f['accuracy'])} "
          f"on {f['n']:,} txns -> {REPORT_MD.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
