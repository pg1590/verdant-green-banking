"""Confidence = how much to trust a propensity, separate from how high it is.

    confidence = completeness x agreement x data_quality
    agreement    = 1 - min(1, std(logit of bootstrap ensemble predictions) / std_norm)
    data_quality = (1 - conflict_penalty) if a product-relevant conflict flag is set
                   x  the product's data_flag_penalties for each P1 data flag present

- completeness (P1 `data_completeness`): missing data lowers confidence.
- agreement: if models trained on bootstrap resamples disagree, the estimate is unstable. Spread
  is measured in log-odds: in probability units it grows with the score itself (0.001 at p~0.01,
  ~0.18 at p>0.4), which would give the highest-propensity customers the lowest
  confidence.
- conflict_flags: contradictory data (e.g. EV on record but road-fuel spend) is flagged, never guessed.
- data_flags: RECENT_CIRCUMSTANCE_CHANGE lowers GP01 confidence (D5); energy apportioned from a
  dual-fuel bill lowers GP02/GP03 confidence (D6). P5 applies no second penalty.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit

from src.common.records import to_python


def parse_data_flags(value) -> set[str]:
    value = to_python(value)
    return {f.strip() for f in value.split(",") if f.strip()} if isinstance(value, str) else set()


def quality_multiplier(conflict_flags, data_flags, conflict_codes, data_flag_penalties: dict,
                       conflict_penalty: float = 0.5) -> np.ndarray:
    """Per-customer multiplier in [0, 1] from conflict_flags (lists) and data_flags (strings)."""
    relevant = set(conflict_codes)
    out = []
    for conflicts, flags in zip(conflict_flags, data_flags):
        m = 1.0 - conflict_penalty if relevant & set(to_python(conflicts) or []) else 1.0
        for flag in parse_data_flags(flags):
            m *= data_flag_penalties.get(flag, 1.0)
        out.append(m)
    return np.asarray(out, dtype=float)


def compute_confidence(completeness, ensemble_preds: np.ndarray, quality=None,
                       std_norm: float = 2.0) -> np.ndarray:
    """ensemble_preds: shape (n_models, n_customers); quality: multiplier per customer (default 1)."""
    completeness = np.nan_to_num(np.asarray(completeness, dtype=float), nan=0.0).clip(0, 1)
    std = logit(np.clip(np.asarray(ensemble_preds, dtype=float), 1e-4, 1 - 1e-4)).std(axis=0)
    agreement = 1.0 - np.minimum(1.0, std / std_norm)
    quality = np.ones_like(completeness) if quality is None else np.asarray(quality, dtype=float)
    return (completeness * agreement * quality).clip(0, 1)
