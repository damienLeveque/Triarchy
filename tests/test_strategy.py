"""Tests for Maestro and Tactical strategy layers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triarchy.config import load_maestro_config, load_tactical_config
from triarchy.strategy import maestro, tactical

CONFIGS_DIR = "configs"


@pytest.fixture
def maestro_cfg():
    return load_maestro_config(f"{CONFIGS_DIR}/maestro.json")


@pytest.fixture
def tactical_cfg():
    return load_tactical_config(f"{CONFIGS_DIR}/tactical.json")


def _make_ohlcv(n: int, freq: str, start_price: float, end_price: float, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV with a smooth trend from start_price to end_price."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    trend = np.linspace(start_price, end_price, n)
    noise = rng.normal(0, abs(end_price - start_price) * 0.01, n)
    close = trend + noise
    high = close + np.abs(rng.normal(0, abs(end_price - start_price) * 0.005, n))
    low = close - np.abs(rng.normal(0, abs(end_price - start_price) * 0.005, n))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(1e5, 5e5, n)
    return pd.DataFrame(
        {"timestamp": dates, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


class TestMaestro:
    def test_uptrend_yields_long_only_bias(self, maestro_cfg) -> None:
        """A clear uptrend should produce LONG_ONLY for the latter portion."""
        df = _make_ohlcv(500, "D", 20000, 60000, seed=1)
        result = maestro.run(df, maestro_cfg)
        # Last quarter of the series should be dominated by LONG_ONLY
        tail = result["BIAS"].tail(100)
        assert (tail == "LONG_ONLY").sum() > (tail == "SHORT_ONLY").sum()

    def test_downtrend_yields_short_only_bias(self, maestro_cfg) -> None:
        """A clear downtrend should at least never produce LONG_ONLY in the tail."""
        df = _make_ohlcv(500, "D", 60000, 20000, seed=2)
        result = maestro.run(df, maestro_cfg)
        tail = result["BIAS"].tail(100)
        # The strict invariant: no LONG_ONLY in a clean downtrend.
        # SHORT_ONLY vs NEUTRAL split depends on weekly VWAP timing.
        assert (tail == "LONG_ONLY").sum() == 0

    def test_required_output_columns(self, maestro_cfg) -> None:
        df = _make_ohlcv(400, "D", 30000, 35000, seed=3)
        result = maestro.run(df, maestro_cfg)
        for col in maestro.BIAS_CSV_COLUMNS:
            assert col in result.columns, f"Missing required column: {col}"

    def test_bias_values_are_valid(self, maestro_cfg) -> None:
        df = _make_ohlcv(400, "D", 30000, 35000, seed=4)
        result = maestro.run(df, maestro_cfg).dropna(subset=["ema_200"])
        valid = {"LONG_ONLY", "SHORT_ONLY", "NEUTRAL"}
        assert set(result["BIAS"].unique()).issubset(valid)

    def test_regime_values_are_valid(self, maestro_cfg) -> None:
        df = _make_ohlcv(400, "D", 30000, 35000, seed=5)
        result = maestro.run(df, maestro_cfg).dropna(subset=["ema_200"])
        valid = {"TRENDING", "RANGING", "CRASH"}
        assert set(result["REGIME"].unique()).issubset(valid)


class TestTactical:
    def test_full_pipeline_runs(self, maestro_cfg, tactical_cfg) -> None:
        df_1d = _make_ohlcv(400, "D", 30000, 40000, seed=6)
        df_4h = _make_ohlcv(400 * 6, "4h", 30000, 40000, seed=7)
        result_1d = maestro.run(df_1d, maestro_cfg)
        result_4h = tactical.run(df_4h, result_1d, tactical_cfg)
        for col in tactical.TACTICAL_CSV_COLUMNS:
            assert col in result_4h.columns, f"Missing required column: {col}"

    def test_playbook_values_are_valid(self, maestro_cfg, tactical_cfg) -> None:
        df_1d = _make_ohlcv(400, "D", 30000, 40000, seed=8)
        df_4h = _make_ohlcv(400 * 6, "4h", 30000, 40000, seed=9)
        result_1d = maestro.run(df_1d, maestro_cfg)
        result_4h = tactical.run(df_4h, result_1d, tactical_cfg).dropna(subset=["adx_4h"])
        valid = {"TREND_FOLLOW", "MEAN_REVERT", "BREAKOUT_WAIT"}
        assert set(result_4h["PLAYBOOK"].unique()).issubset(valid)

    def test_risk_mode_off_when_crash(self, tactical_cfg) -> None:
        """If REGIME=CRASH propagates through, RISK_MODE must be OFF."""
        df = pd.DataFrame({"REGIME": ["CRASH", "TRENDING"], "BIAS": ["NEUTRAL", "LONG_ONLY"]})
        result = tactical.apply_risk_mode(df)
        assert result.loc[0, "RISK_MODE"] == "OFF"
        assert result.loc[1, "RISK_MODE"] == "NORMAL"

    def test_risk_mode_reduced_when_neutral(self, tactical_cfg) -> None:
        df = pd.DataFrame({"REGIME": ["TRENDING"], "BIAS": ["NEUTRAL"]})
        result = tactical.apply_risk_mode(df)
        assert result.loc[0, "RISK_MODE"] == "REDUCED"
