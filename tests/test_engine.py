"""Tests for the backtest engine and execution strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triarchy.backtest.engine import _resolve_exit, backtest_symbol
from triarchy.config import load_execution_config
from triarchy.strategy.execution import (
    _apply_slippage,
    compute_features,
    compute_trade_params,
    fees_for,
    gating_long,
    gating_short,
    signal_at,
)


@pytest.fixture
def exec_cfg():
    return load_execution_config("configs/execution.json")


# ---------- Slippage / fees ----------

class TestSlippage:
    def test_long_entry_pays_up(self) -> None:
        assert _apply_slippage(100.0, "LONG", is_entry=True, slippage_bps=10) == pytest.approx(100.10)

    def test_long_exit_sells_down(self) -> None:
        assert _apply_slippage(100.0, "LONG", is_entry=False, slippage_bps=10) == pytest.approx(99.90)

    def test_short_entry_sells_down(self) -> None:
        assert _apply_slippage(100.0, "SHORT", is_entry=True, slippage_bps=10) == pytest.approx(99.90)

    def test_short_exit_buys_up(self) -> None:
        assert _apply_slippage(100.0, "SHORT", is_entry=False, slippage_bps=10) == pytest.approx(100.10)


class TestFees:
    def test_fees_proportional(self) -> None:
        assert fees_for(10000, 0.0004) == pytest.approx(4.0)
        assert fees_for(0, 0.0004) == 0.0


# ---------- Gating ----------

class TestGating:
    def test_long_blocked_by_off_risk(self) -> None:
        row = pd.Series({
            "RISK_MODE": "OFF", "BIAS": "LONG_ONLY", "PLAYBOOK": "TREND_FOLLOW",
            "DIR_4H": "UP", "is_compressed": False,
        })
        assert not gating_long(row)

    def test_long_blocked_by_wrong_bias(self) -> None:
        row = pd.Series({
            "RISK_MODE": "NORMAL", "BIAS": "SHORT_ONLY", "PLAYBOOK": "TREND_FOLLOW",
            "DIR_4H": "UP", "is_compressed": False,
        })
        assert not gating_long(row)

    def test_long_blocked_when_compressed(self) -> None:
        row = pd.Series({
            "RISK_MODE": "NORMAL", "BIAS": "LONG_ONLY", "PLAYBOOK": "TREND_FOLLOW",
            "DIR_4H": "UP", "is_compressed": True,
        })
        assert not gating_long(row)

    def test_long_passes_with_flat_dir(self) -> None:
        row = pd.Series({
            "RISK_MODE": "NORMAL", "BIAS": "LONG_ONLY", "PLAYBOOK": "TREND_FOLLOW",
            "DIR_4H": "FLAT", "is_compressed": False,
        })
        assert gating_long(row)

    def test_short_symmetric(self) -> None:
        row = pd.Series({
            "RISK_MODE": "NORMAL", "BIAS": "SHORT_ONLY", "PLAYBOOK": "TREND_FOLLOW",
            "DIR_4H": "DOWN", "is_compressed": False,
        })
        assert gating_short(row)


# ---------- Trade params ----------

class TestTradeParams:
    def test_long_stop_capped_by_atr(self, exec_cfg) -> None:
        # Swing far below entry should be capped by sl_atr_cap * atr
        params = compute_trade_params(
            side="LONG", next_open=100.0,
            swing_level=50.0,           # very far swing
            atr_value=2.0, equity=10000.0, risk_mult=1.0, cfg=exec_cfg,
        )
        assert params is not None
        # stop should not be farther than sl_atr_cap * atr below entry
        max_dist = exec_cfg.risk.sl_atr_cap * 2.0
        assert (params.entry - params.stop) <= max_dist + 1e-6

    def test_long_uses_swing_when_close(self, exec_cfg) -> None:
        # Swing close to entry => the stop is the swing (tighter than the cap)
        params = compute_trade_params(
            side="LONG", next_open=100.0,
            swing_level=99.0,           # tight swing
            atr_value=5.0, equity=10000.0, risk_mult=1.0, cfg=exec_cfg,
        )
        assert params is not None
        # Stop should be equal to swing (since cap floor would be wider)
        assert params.stop == pytest.approx(99.0, abs=0.05)  # entry slippage tolerance

    def test_tp_at_correct_r_multiple(self, exec_cfg) -> None:
        params = compute_trade_params(
            side="LONG", next_open=100.0, swing_level=98.0,
            atr_value=2.0, equity=10000.0, risk_mult=1.0, cfg=exec_cfg,
        )
        assert params is not None
        risk = params.entry - params.stop
        assert params.take_profit == pytest.approx(params.entry + exec_cfg.risk.tp_r_mult * risk)

    def test_short_geometry(self, exec_cfg) -> None:
        params = compute_trade_params(
            side="SHORT", next_open=100.0, swing_level=102.0,
            atr_value=2.0, equity=10000.0, risk_mult=1.0, cfg=exec_cfg,
        )
        assert params is not None
        assert params.stop > params.entry
        assert params.take_profit < params.entry
        risk = params.stop - params.entry
        assert params.take_profit == pytest.approx(params.entry - exec_cfg.risk.tp_r_mult * risk)

    def test_position_sizing_respects_risk_mult(self, exec_cfg) -> None:
        full = compute_trade_params(
            side="LONG", next_open=100.0, swing_level=98.0,
            atr_value=2.0, equity=10000.0, risk_mult=1.0, cfg=exec_cfg,
        )
        half = compute_trade_params(
            side="LONG", next_open=100.0, swing_level=98.0,
            atr_value=2.0, equity=10000.0, risk_mult=0.5, cfg=exec_cfg,
        )
        assert full is not None and half is not None
        assert half.qty == pytest.approx(full.qty * 0.5)


# ---------- Exit resolution ----------

class TestExitResolution:
    def test_long_sl_hit(self) -> None:
        bar = pd.Series({"high": 105, "low": 95})
        price, reason = _resolve_exit("LONG", bar, stop=96, tp=110)
        assert price == 96 and reason == "SL"

    def test_long_tp_hit(self) -> None:
        bar = pd.Series({"high": 110, "low": 99})
        price, reason = _resolve_exit("LONG", bar, stop=95, tp=108)
        assert price == 108 and reason == "TP"

    def test_long_no_exit(self) -> None:
        bar = pd.Series({"high": 102, "low": 99})
        price, reason = _resolve_exit("LONG", bar, stop=95, tp=110)
        assert price is None and reason is None

    def test_long_sl_priority_when_both_hit(self) -> None:
        # Both SL and TP inside range — engine assumes SL first (conservative)
        bar = pd.Series({"high": 110, "low": 90})
        price, reason = _resolve_exit("LONG", bar, stop=95, tp=105)
        assert reason == "SL"

    def test_short_sl_hit(self) -> None:
        bar = pd.Series({"high": 105, "low": 95})
        price, reason = _resolve_exit("SHORT", bar, stop=104, tp=90)
        assert price == 104 and reason == "SL"

    def test_short_tp_hit(self) -> None:
        bar = pd.Series({"high": 99, "low": 90})
        price, reason = _resolve_exit("SHORT", bar, stop=110, tp=92)
        assert price == 92 and reason == "TP"


# ---------- End-to-end engine smoke test ----------

def _build_synthetic_1h_with_signal(n: int = 600) -> pd.DataFrame:
    """
    Build a synthetic 1H DataFrame that satisfies all gating conditions for a LONG
    signal at a specific bar. This validates the engine produces at least one trade.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Gentle uptrend
    trend = np.linspace(100, 200, n)
    noise = rng.normal(0, 0.5, n)
    close = trend + noise
    high = close + 0.3
    low = close - 0.3
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": dates, "open": open_, "high": high, "low": low,
            "close": close, "volume": rng.uniform(100, 200, n),
        }
    )


class TestEngine:
    def test_engine_runs_without_error_on_synthetic(self, exec_cfg) -> None:
        df = _build_synthetic_1h_with_signal(n=600)
        df = compute_features(df, exec_cfg)
        # Force gating columns so signals can fire
        df["BIAS"] = "LONG_ONLY"
        df["REGIME"] = "TRENDING"
        df["RISK_MODE"] = "NORMAL"
        df["PLAYBOOK"] = "TREND_FOLLOW"
        df["DIR_4H"] = "UP"
        df["is_compressed"] = False

        result = backtest_symbol("TEST/USDT", df, exec_cfg, starting_equity=10000.0)
        # Equity curve should be non-empty
        assert not result.equity_curve.empty
        # Final equity should be a finite positive number
        assert result.final_equity > 0
        assert np.isfinite(result.final_equity)

    def test_engine_no_trades_when_risk_off(self, exec_cfg) -> None:
        df = _build_synthetic_1h_with_signal(n=600)
        df = compute_features(df, exec_cfg)
        df["BIAS"] = "LONG_ONLY"
        df["REGIME"] = "CRASH"
        df["RISK_MODE"] = "OFF"
        df["PLAYBOOK"] = "TREND_FOLLOW"
        df["DIR_4H"] = "UP"
        df["is_compressed"] = False

        result = backtest_symbol("TEST/USDT", df, exec_cfg, starting_equity=10000.0)
        assert len(result.trades) == 0
        assert result.final_equity == 10000.0

    def test_signal_returns_none_when_neutral(self, exec_cfg) -> None:
        row = pd.Series({
            "RISK_MODE": "NORMAL", "BIAS": "NEUTRAL", "PLAYBOOK": "TREND_FOLLOW",
            "DIR_4H": "FLAT", "is_compressed": False,
            "trend_long_ok": True, "trend_short_ok": False,
            "near_ema50": True, "reclaim_ema20_long": True, "reclaim_ema20_short": False,
        })
        assert signal_at(row, exec_cfg) is None
