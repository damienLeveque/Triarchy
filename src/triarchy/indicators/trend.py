"""Trend indicators: EMA and slope."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with strict warmup (no values until `period` bars)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def ema_slope(series: pd.Series, lookback: int) -> pd.Series:
    """Percent change of an EMA over `lookback` bars — used as a directional filter."""
    return series.pct_change(periods=lookback)
