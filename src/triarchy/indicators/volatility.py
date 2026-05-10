"""Volatility indicators: ATR and Bollinger bands."""

from __future__ import annotations

import pandas as pd


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Average True Range using Wilder smoothing (EWM with alpha = 1/period).

    Expects columns: high, low, close.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(
    series: pd.Series, period: int, std_mult: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger bands. Returns (mid, upper, lower, width_pct).

    `width_pct` is normalized by price: (upper - lower) / close.
    """
    mid = series.rolling(period, min_periods=period).mean()
    sd = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    width = (upper - lower) / series
    return mid, upper, lower, width
