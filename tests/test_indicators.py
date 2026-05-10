"""Unit tests for technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triarchy.indicators import adx, atr, bollinger, ema, ema_slope


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """200 bars of synthetic OHLCV with a clear uptrend then sideways."""
    rng = np.random.default_rng(42)
    n = 200
    trend = np.concatenate([np.linspace(100, 150, 100), np.full(100, 150.0)])
    noise = rng.normal(0, 0.5, n)
    close = trend + noise
    high = close + np.abs(rng.normal(0, 0.8, n))
    low = close - np.abs(rng.normal(0, 0.8, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


class TestEMA:
    def test_ema_warmup(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """EMA should return NaN for the first `period - 1` bars."""
        result = ema(synthetic_ohlcv["close"], period=20)
        assert result.iloc[:19].isna().all()
        assert not result.iloc[19:].isna().any()

    def test_ema_tracks_trend(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """During the uptrend phase, EMA should be rising."""
        result = ema(synthetic_ohlcv["close"], period=20)
        # Compare two points well inside the uptrend
        assert result.iloc[80] > result.iloc[40]

    def test_ema_slope_positive_in_uptrend(self, synthetic_ohlcv: pd.DataFrame) -> None:
        ema_series = ema(synthetic_ohlcv["close"], period=20)
        slope = ema_slope(ema_series, lookback=5)
        assert slope.iloc[80] > 0


class TestATR:
    def test_atr_is_non_negative(self, synthetic_ohlcv: pd.DataFrame) -> None:
        result = atr(synthetic_ohlcv, period=14).dropna()
        assert (result >= 0).all()

    def test_atr_warmup(self, synthetic_ohlcv: pd.DataFrame) -> None:
        result = atr(synthetic_ohlcv, period=14)
        assert result.iloc[:13].isna().all()


class TestBollinger:
    def test_bands_ordering(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """Lower < mid < upper at every defined point."""
        mid, upper, lower, _ = bollinger(synthetic_ohlcv["close"], period=20, std_mult=2.0)
        df = pd.concat([lower, mid, upper], axis=1).dropna()
        df.columns = ["lo", "mi", "up"]
        assert (df["lo"] <= df["mi"]).all()
        assert (df["mi"] <= df["up"]).all()

    def test_width_is_positive(self, synthetic_ohlcv: pd.DataFrame) -> None:
        _, _, _, width = bollinger(synthetic_ohlcv["close"], period=20, std_mult=2.0)
        assert (width.dropna() >= 0).all()


class TestADX:
    def test_adx_in_valid_range(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """ADX is bounded in [0, 100]."""
        atr_s = atr(synthetic_ohlcv, period=14)
        result = adx(synthetic_ohlcv, atr_s, period=14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()
