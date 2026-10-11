"""Convert DataFrames into the plain-Python records P5's eligibility and
integration bundle expect (docs/integration.md).

P5's engine needs dates as ISO strings and its bundle loader rejects NaN, and its
rules compare exact Python types (a numpy.bool_ True is not `True`). Raw
DataFrame.to_dict("records") gives Timestamps, NaN and, depending on the pandas
version, numpy scalars. Always export customer snapshots through to_records().
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd


def to_python(value):
    """numpy/pandas scalar -> Python primitive; NaN/NA/NaT -> None; dates -> ISO string."""
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_python(v) for v in value.tolist()]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def to_records(df: pd.DataFrame) -> list[dict]:
    return [{k: to_python(v) for k, v in row.items()} for row in df.to_dict("records")]
