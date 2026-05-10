"""
Maestro — 1D Strategic Layer.

Determines BIAS (LONG_ONLY / SHORT_ONLY / NEUTRAL) and REGIME (TRENDING /
RANGING / CRASH) from daily candles. The output cascades down to the Tactical
and Execution layers.

This module contains pure functions only. Data I/O happens in the CLI / pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from triarchy.config import MaestroConfig
from triarchy.indicators import adx, atr, ema, ema_slope


def compute_indicators(df: pd.DataFrame, cfg: MaestroConfig) -> pd.DataFrame:
    """
    Compute all daily indicators required for BIAS/REGIME classification.

    Adds columns: ema_200, ema_slope, atr, atr_pct, adx, dist_ema.
    """
    out = df.copy()
    p = cfg.indicators

    out["ema_200"] = ema(out["close"], p.ema_period)
    out["ema_slope"] = ema_slope(out["ema_200"], p.ema_slope_lookback)

    out["atr"] = atr(out, p.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]

    out["adx"] = adx(out, out["atr"], p.adx_period)

    out["dist_ema"] = (out["close"] - out["ema_200"]) / out["ema_200"]

    return out


def compute_weekly_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a weekly VWAP column. Resets every Monday (UTC).

    Used as a tactical filter in the BIAS rules: a long bias requires
    price > weekly VWAP (above value), and vice versa for shorts.
    """
    out = df.copy()

    weekday = out["timestamp"].dt.weekday
    new_week = weekday == 0
    week_group = new_week.cumsum()

    pv = out["close"] * out["volume"]
    cum_pv = pv.groupby(week_group).cumsum()
    cum_vol = out["volume"].groupby(week_group).cumsum()

    out["vwap_weekly"] = cum_pv / cum_vol
    return out


def classify(df: pd.DataFrame, cfg: MaestroConfig) -> pd.DataFrame:
    """
    Apply the BIAS and REGIME classification rules.

    Expects `compute_indicators` and `compute_weekly_vwap` to have already run.
    """
    out = df.copy()
    p = cfg.indicators

    # Strategic direction (EMA200 slope)
    cond_slope_up = out["ema_slope"] > p.ema_slope_min
    cond_slope_down = out["ema_slope"] < -p.ema_slope_min

    # Tactical filter (price vs weekly VWAP)
    cond_above_vwap = out["close"] > out["vwap_weekly"]
    cond_below_vwap = out["close"] < out["vwap_weekly"]

    # Regime
    is_crash = out["dist_ema"] <= -p.crash_deviation
    is_trend = out["adx"] >= p.adx_threshold_trend

    out["BIAS"] = np.select(
        [
            cond_slope_up & cond_above_vwap & ~is_crash,
            cond_slope_down & cond_below_vwap & ~is_crash,
        ],
        ["LONG_ONLY", "SHORT_ONLY"],
        default="NEUTRAL",
    )

    out["REGIME"] = np.select(
        [is_crash, is_trend],
        ["CRASH", "TRENDING"],
        default="RANGING",
    )

    return out


def run(df: pd.DataFrame, cfg: MaestroConfig) -> pd.DataFrame:
    """Full Maestro pipeline: indicators -> weekly VWAP -> classification."""
    df = compute_indicators(df, cfg)
    df = compute_weekly_vwap(df)
    df = classify(df, cfg)
    return df


# Columns persisted to the BIAS cache CSV (consumed by Tactical and Execution).
BIAS_CSV_COLUMNS = [
    "timestamp", "open", "high", "low", "close",
    "ema_200", "vwap_weekly", "adx", "dist_ema",
    "REGIME", "BIAS",
]
