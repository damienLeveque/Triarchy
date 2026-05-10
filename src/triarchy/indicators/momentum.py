"""Momentum indicators: ADX (with directional movement components)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def adx(df: pd.DataFrame, atr_series: pd.Series, period: int) -> pd.Series:
    """
    Average Directional Index. Returns a single ADX series.

    Expects columns: high, low. `atr_series` is the precomputed ATR over the same period.
    """
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    alpha = 1 / period
    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_series
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_series
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=alpha, adjust=False).mean()
