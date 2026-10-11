"""Customer-level train/val/test split (D4). Every customer is in exactly one split,
so no customer's transactions or labels leak across splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import config


def split_customers(customer_ids, seed: int) -> pd.DataFrame:
    frac = config.settings()["splits"]
    ids = np.array(sorted(customer_ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train = int(round(frac["train"] * len(ids)))
    n_val = int(round(frac["val"] * len(ids)))
    split = np.empty(len(ids), dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_val]] = "val"
    split[order[n_train + n_val:]] = "test"
    return pd.DataFrame({"customer_id": ids, "split": split})
