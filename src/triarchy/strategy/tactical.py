"""
Tactical — 4H Playbook Layer.

Classifies the 4H context into a PLAYBOOK (TREND_FOLLOW / MEAN_REVERT /
BREAKOUT_WAIT) and a directional bias (UP / DOWN / FLAT). Also derives
RISK_MODE based on the cascaded Maestro BIAS/REGIME.

Pure logic; no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from triarchy.config import TacticalConfig
from triarchy.indicators import adx, atr, bollinger, ema


def compute_indicators(df: pd.DataFrame, cfg: TacticalConfig) -> pd.DataFrame:
    """
    Compute the 4H indicator stack: EMAs, slope, ATR, ADX, Bollinger width.
    """
    out = df.copy()

    out["ema_20"] = ema(out["close"], cfg.ema.fast)
    out["ema_50"] = ema(out["close"], cfg.ema.mid)
    out["ema_200_4h"] = ema(out["close"], cfg.ema.slow)

    out["ema50_slope"] = out["ema_50"].pct_change(periods=cfg.slope.lookback)

    out["atr_4h"] = atr(out, cfg.atr.period)
    out["atr_pct_4h"] = out["atr_4h"] / out["close"]

    out["adx_4h"] = adx(out, out["atr_4h"], cfg.adx.period)

    bb_mid, bb_up, bb_low, bb_width = bollinger(
        out["close"], cfg.bollinger.period, cfg.bollinger.std
    )
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_up
    out["bb_lower"] = bb_low
    out["bb_width"] = bb_width

    out["is_compressed"] = out["bb_width"] <= cfg.bollinger.width_low_pct
    return out


def add_range_levels(df: pd.DataFrame, cfg: TacticalConfig) -> pd.DataFrame:
    """
    Define a rolling range from the last N 4H bars and tag whether price is at
    the edge. Used by mean-reversion playbooks.
    """
    out = df.copy()
    n = cfg.range.lookback

    out["RANGE_HIGH"] = out["high"].rolling(n, min_periods=n).max()
    out["RANGE_LOW"] = out["low"].rolling(n, min_periods=n).min()
    out["RANGE_MID"] = (out["RANGE_HIGH"] + out["RANGE_LOW"]) / 2.0

    width = out["RANGE_HIGH"] - out["RANGE_LOW"]
    out["RANGE_WIDTH_PCT"] = width / out["close"]
    out["RANGE_WIDTH_ATR"] = width / out["atr_4h"]

    out["RANGE_VALID"] = (
        out["RANGE_WIDTH_ATR"].between(cfg.range.min_atr, cfg.range.max_atr)
        & (out["RANGE_WIDTH_PCT"] <= cfg.range.max_pct)
    )

    tol = cfg.range.edge_atr_tol * out["atr_4h"]
    out["AT_RANGE_LOW"] = out["close"] <= (out["RANGE_LOW"] + tol)
    out["AT_RANGE_HIGH"] = out["close"] >= (out["RANGE_HIGH"] - tol)

    return out


def classify_playbook(df: pd.DataFrame, cfg: TacticalConfig) -> pd.DataFrame:
    """
    Decide PLAYBOOK and DIR_4H.

    PLAYBOOK priority:
      1. is_compressed       => BREAKOUT_WAIT
      2. ADX >= trend_on     => TREND_FOLLOW
      3. ADX <= range_on AND valid range => MEAN_REVERT
      4. otherwise           => BREAKOUT_WAIT (safe default)
    """
    out = df.copy()

    dir_up = (out["ema50_slope"] > cfg.slope.min) & (out["close"] > out["ema_50"])
    dir_down = (out["ema50_slope"] < -cfg.slope.min) & (out["close"] < out["ema_50"])
    out["DIR_4H"] = np.select([dir_up, dir_down], ["UP", "DOWN"], default="FLAT")

    trend_on = out["adx_4h"] >= cfg.adx.trend_on
    range_on = (out["adx_4h"] <= cfg.adx.range_on) & (out["RANGE_VALID"])

    out["PLAYBOOK"] = np.select(
        [out["is_compressed"], trend_on, range_on],
        ["BREAKOUT_WAIT", "TREND_FOLLOW", "MEAN_REVERT"],
        default="BREAKOUT_WAIT",
    )
    return out


def merge_daily_bias(df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the daily BIAS / REGIME to each 4H bar by date (UTC).

    `df_1d` must have columns: timestamp, BIAS, REGIME.
    """
    df_i = df_4h.copy()
    df_d = df_1d.copy()

    df_i["date"] = df_i["timestamp"].dt.floor("D")
    df_d["date"] = df_d["timestamp"].dt.floor("D")

    merged = df_i.merge(df_d[["date", "BIAS", "REGIME"]], on="date", how="left")
    merged[["BIAS", "REGIME"]] = merged[["BIAS", "REGIME"]].ffill()
    merged.drop(columns=["date"], inplace=True)
    return merged


def apply_risk_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cascade risk gating from Maestro:
      REGIME=CRASH    -> RISK_MODE=OFF
      BIAS=NEUTRAL    -> RISK_MODE=REDUCED
      else            -> RISK_MODE=NORMAL
    """
    out = df.copy()
    out["RISK_MODE"] = np.select(
        [out["REGIME"] == "CRASH", out["BIAS"] == "NEUTRAL"],
        ["OFF", "REDUCED"],
        default="NORMAL",
    )
    return out


def run(df_4h: pd.DataFrame, df_1d: pd.DataFrame, cfg: TacticalConfig) -> pd.DataFrame:
    """Full Tactical pipeline: indicators -> ranges -> playbook -> merge daily -> risk."""
    df = compute_indicators(df_4h, cfg)
    df = add_range_levels(df, cfg)
    df = merge_daily_bias(df, df_1d)
    df = classify_playbook(df, cfg)
    df = apply_risk_mode(df)
    return df


# Columns persisted to the TACTICAL cache CSV (consumed by Execution).
TACTICAL_CSV_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "BIAS", "REGIME", "RISK_MODE",
    "ema_20", "ema_50", "ema_200_4h", "ema50_slope",
    "adx_4h", "atr_4h", "atr_pct_4h",
    "bb_mid", "bb_upper", "bb_lower", "bb_width", "is_compressed",
    "DIR_4H", "PLAYBOOK",
    "RANGE_HIGH", "RANGE_LOW", "RANGE_MID",
    "RANGE_WIDTH_PCT", "RANGE_WIDTH_ATR", "RANGE_VALID",
    "AT_RANGE_LOW", "AT_RANGE_HIGH",
]
