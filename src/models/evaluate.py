"""Model-agnostic evaluation: everything here is a function of (y_true, y_prob)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _safe(fn, y, p):
    return float(fn(y, p)) if len(np.unique(y)) == 2 else float("nan")


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def precision_at_k(y, p, k: float = 0.10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    n = max(1, int(round(len(p) * k)))
    top = np.argsort(-p, kind="stable")[:n]
    return float(y[top].mean())


def metrics(y, p, k: float = 0.10) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    base = float(y.mean()) if len(y) else float("nan")
    p_at_k = precision_at_k(y, p, k)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": base,
        "roc_auc": _safe(roc_auc_score, y, p),
        "pr_auc": _safe(average_precision_score, y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        f"precision_at_{int(k * 100)}pct": p_at_k,
        f"lift_at_{int(k * 100)}pct": p_at_k / base if base > 0 else float("nan"),
    }


def bootstrap_ci(y, p, metric: str = "roc_auc", n_boot: int = 500, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI over customers for one metric from `metrics()`."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = metrics(y[idx], p[idx])[metric]
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def lift_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Decile 1 = highest scores. cum_capture = share of all positives captured up to that decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), n_bins, labels=range(1, n_bins + 1))
    base = df["y"].mean()
    t = df.groupby("decile", observed=True).agg(n=("y", "size"), positives=("y", "sum"), mean_score=("p", "mean"))
    t["take_up_rate"] = t["positives"] / t["n"]
    t["lift"] = t["take_up_rate"] / base if base > 0 else np.nan
    t["cum_capture"] = t["positives"].cumsum() / max(1, df["y"].sum())
    return t.reset_index()


def fairness_slices(df: pd.DataFrame, group_col: str, y_col: str, p_col: str,
                    min_class: int = 10, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Per-group size, mean propensity, actual take-up and AUC (for the RAI note).

    AUC (and its bootstrap 95% CI) is reported only when the group has at least `min_class`
    positives AND negatives; below that it is NaN and `auc_reliable` is False, so small
    groups (e.g. ~25 GP02 val customers per region) never produce quotable noise.
    calibration_ratio = mean propensity / actual take-up (1 = calibrated in this group)."""
    rows = []
    for g, d in df.groupby(group_col, observed=True):
        y, p = d[y_col].to_numpy(), d[p_col].to_numpy()
        n_pos = int(y.sum())
        reliable = n_pos >= min_class and len(y) - n_pos >= min_class
        lo, hi = bootstrap_ci(y, p, "roc_auc", n_boot=n_boot, seed=seed) if reliable else (np.nan, np.nan)
        rows.append({
            "slice": group_col, "group": g, "n": len(d), "n_pos": n_pos,
            "mean_propensity": float(p.mean()), "take_up_rate": float(y.mean()),
            "calibration_ratio": float(p.mean() / y.mean()) if n_pos else np.nan,
            "roc_auc": _safe(roc_auc_score, y, p) if reliable else np.nan,
            "roc_auc_ci95_lo": lo, "roc_auc_ci95_hi": hi, "auc_reliable": reliable,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["propensity_ratio_vs_max"] = out["mean_propensity"] / out["mean_propensity"].max()
    return out


def reliability_plot(y, probs: dict[str, np.ndarray], out_path, title: str, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    y = np.asarray(y)
    for name, p in probs.items():
        p = np.asarray(p)
        bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = bins == b
            if m.sum() >= 5:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", label=name)
    ax.set(xlabel="Predicted propensity", ylabel="Observed take-up rate", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def lift_plot(table: pd.DataFrame, out_path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["decile"].astype(int), table["lift"], color="#1f7a50")
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set(xlabel="Score decile (1 = highest)", ylabel="Lift vs base rate", title=title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
