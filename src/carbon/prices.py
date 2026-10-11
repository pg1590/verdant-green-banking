"""Historical energy conversion inputs shared by P1 and P2."""
from functools import lru_cache

import pandas as pd

from src.common import config


@lru_cache
def window_energy_prices(start=None, cutoff=None) -> dict[str, float]:
    """Day-weighted window rates and standing charges; no post-cutoff tariffs."""
    calendar = config.settings()['calendar']
    start = pd.Timestamp(start or calendar['feature_window_start'])
    cutoff = pd.Timestamp(cutoff or calendar['feature_cutoff_date'])
    quarters = pd.read_csv(config.CONFIG_DIR / 'energy_prices.csv', parse_dates=['period_start', 'period_end'])
    quarters = quarters[(quarters.period_end >= start) & (quarters.period_start <= cutoff)].copy().sort_values('period_start')
    quarters['period_start'] = quarters.period_start.clip(lower=start)
    quarters['period_end'] = quarters.period_end.clip(upper=cutoff)
    expected = start
    for row in quarters.itertuples():
        if row.period_start != expected or row.period_end < row.period_start:
            raise ValueError('Energy tariff periods must cover the feature window without gaps or overlaps')
        expected = row.period_end + pd.Timedelta(1, unit='D')
    if expected != cutoff + pd.Timedelta(1, unit='D'):
        raise ValueError('Energy tariff periods must cover the entire feature window')
    days = (quarters.period_end - quarters.period_start).dt.days + 1
    result = {}
    for fuel in ('elec', 'gas'):
        result[f'{fuel}_unit'] = float((quarters[f'{fuel}_unit_p_per_kwh'] * days).sum() / days.sum() / 100)
        result[f'{fuel}_standing_year'] = float((quarters[f'{fuel}_standing_p_per_day'] * days).sum() / days.sum() * 365 / 100)
    return result
