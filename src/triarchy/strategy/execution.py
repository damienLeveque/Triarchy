"""
Execution — 1H Trade Signal Layer.

Generates entry signals (LONG / SHORT / NONE) and computes per-bar trade
parameters (entry price reference, stop, take-profit, risk distance) given
the cascaded Maestro BIAS / Tactical PLAYBOOK / RISK_MODE.

This module contains pure logic only. The simulation engine in
`triarchy.backtest.engine` consumes these signals and walks through bars to
produce a trade log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from triarchy.config import ExecutionConfig
from triarchy.indicators import atr, ema

# ---------- Feature engineering ----------

def compute_features(df: pd.DataFrame, cfg: ExecutionConfig) -> pd.DataFrame:
    """
    Add 1H indicators and structural columns used for entry decisions.

    Adds: ema20, ema50, ema200, atr,
          trend_long_ok, trend_short_ok, near_ema50,
          reclaim_ema20_long, reclaim_ema20_short,
          swing_low, swing_high.
    """
    out = df.copy()
    p = cfg.indicators

    out["ema20"] = ema(out["close"], p.ema_fast)
    out["ema50"] = ema(out["close"], p.ema_mid)
    out["ema200"] = ema(out["close"], p.ema_slow)
    out["atr"] = atr(out, p.atr_period)

    # Trend structure: aligned EMAs and price on the right side of EMA200
    out["trend_long_ok"] = (out["close"] > out["ema200"]) & (out["ema50"] > out["ema200"])
    out["trend_short_ok"] = (out["close"] < out["ema200"]) & (out["ema50"] < out["ema200"])

    # "Near EMA50" = within N * ATR — a soft pullback proxy
    out["near_ema50"] = (
        (out["close"] - out["ema50"]).abs() <= (cfg.entry.near_ema50_atr_mult * out["atr"])
    )

    # EMA20 reclaim: close was at/under EMA20 last bar and above this bar (long-side),
    # symmetric for shorts. This filters in continuation entries after a pullback.
    out["reclaim_ema20_long"] = (
        (out["close"] > out["ema20"]) & (out["close"].shift(1) <= out["ema20"].shift(1))
    )
    out["reclaim_ema20_short"] = (
        (out["close"] < out["ema20"]) & (out["close"].shift(1) >= out["ema20"].shift(1))
    )

    # Recent swing low/high — used as natural stop levels
    n = cfg.entry.pullback_lookback
    out["swing_low"] = out["low"].rolling(n, min_periods=n).min()
    out["swing_high"] = out["high"].rolling(n, min_periods=n).max()

    return out


# ---------- Gating (cascaded from upper layers) ----------

def gating_long(row: pd.Series) -> bool:
    """All upper-layer conditions required for a LONG entry to be eligible."""
    if row.get("RISK_MODE", "NORMAL") == "OFF":
        return False
    if row.get("BIAS") != "LONG_ONLY":
        return False
    if row.get("PLAYBOOK", "") != "TREND_FOLLOW":
        return False
    if row.get("DIR_4H", "") not in ("UP", "FLAT"):
        return False
    if bool(row.get("is_compressed", False)):
        return False
    return True


def gating_short(row: pd.Series) -> bool:
    """All upper-layer conditions required for a SHORT entry to be eligible."""
    if row.get("RISK_MODE", "NORMAL") == "OFF":
        return False
    if row.get("BIAS") != "SHORT_ONLY":
        return False
    if row.get("PLAYBOOK", "") != "TREND_FOLLOW":
        return False
    if row.get("DIR_4H", "") not in ("DOWN", "FLAT"):
        return False
    if bool(row.get("is_compressed", False)):
        return False
    return True


# ---------- Signal generation ----------

@dataclass(frozen=True)
class Signal:
    """A potential entry signal at a given bar."""
    side: str  # "LONG" or "SHORT"
    risk_mult: float


def signal_at(row: pd.Series, cfg: ExecutionConfig) -> Signal | None:
    """
    Return a Signal for this bar, or None if no entry is triggered.

    Entry rules:
      LONG:  BIAS=LONG_ONLY + PLAYBOOK=TREND_FOLLOW + DIR_4H in (UP, FLAT)
             + close > ema200 + ema50 > ema200
             + near_ema50 (pullback) + reclaim_ema20 (if confirm_reclaim)

      SHORT: symmetric
    """
    risk_mode = row.get("RISK_MODE", "NORMAL")
    if risk_mode == "OFF":
        return None
    risk_mult = 1.0 if risk_mode == "NORMAL" else 0.5

    if (
        gating_long(row)
        and bool(row.get("trend_long_ok", False))
        and bool(row.get("near_ema50", False))
        and (not cfg.entry.confirm_reclaim or bool(row.get("reclaim_ema20_long", False)))
    ):
        return Signal(side="LONG", risk_mult=risk_mult)

    if (
        gating_short(row)
        and bool(row.get("trend_short_ok", False))
        and bool(row.get("near_ema50", False))
        and (not cfg.entry.confirm_reclaim or bool(row.get("reclaim_ema20_short", False)))
    ):
        return Signal(side="SHORT", risk_mult=risk_mult)

    return None


# ---------- Trade-parameter computation ----------

@dataclass(frozen=True)
class TradeParams:
    """Concrete entry/stop/TP prices and risk distance for a candidate trade."""
    side: str
    entry: float            # exec price after slippage
    stop: float
    take_profit: float
    risk_per_unit: float    # |entry - stop|
    qty: float


def _apply_slippage(price: float, side: str, is_entry: bool, slippage_bps: float) -> float:
    """Worse-of-fill slippage model in bps."""
    bps = slippage_bps / 10_000.0
    if is_entry:
        return price * (1 + bps) if side == "LONG" else price * (1 - bps)
    return price * (1 - bps) if side == "LONG" else price * (1 + bps)


def fees_for(notional: float, fee_rate: float) -> float:
    """Per-side fee."""
    return notional * fee_rate


def compute_trade_params(
    side: str,
    next_open: float,
    swing_level: float,
    atr_value: float,
    equity: float,
    risk_mult: float,
    cfg: ExecutionConfig,
) -> TradeParams | None:
    """
    Compute concrete entry/stop/TP and position size.

    Returns None if the resulting risk distance is non-positive (can happen
    when slippage pushes entry past the swing level).

    `swing_level` is swing_low for LONG, swing_high for SHORT.
    """
    entry_exec = _apply_slippage(next_open, side, is_entry=True, slippage_bps=cfg.costs.slippage_bps)
    atr_cap_dist = cfg.risk.sl_atr_cap * atr_value

    if side == "LONG":
        # If swing_level is NaN (warmup), fall back to ATR-based stop
        sl_swing = swing_level if not np.isnan(swing_level) else entry_exec - 1.2 * atr_value
        # Final stop is the *closer* of (swing, ATR-cap-floor) — never wider than the cap
        stop_px = max(sl_swing, entry_exec - atr_cap_dist)
        risk_per_unit = entry_exec - stop_px
        if risk_per_unit <= 0:
            return None
        tp_px = entry_exec + cfg.risk.tp_r_mult * risk_per_unit
    else:
        sl_swing = swing_level if not np.isnan(swing_level) else entry_exec + 1.2 * atr_value
        stop_px = min(sl_swing, entry_exec + atr_cap_dist)
        risk_per_unit = stop_px - entry_exec
        if risk_per_unit <= 0:
            return None
        tp_px = entry_exec - cfg.risk.tp_r_mult * risk_per_unit

    risk_dollars = equity * cfg.risk.per_trade_pct * risk_mult
    qty = (risk_dollars / risk_per_unit) * cfg.risk.leverage

    return TradeParams(
        side=side,
        entry=entry_exec,
        stop=stop_px,
        take_profit=tp_px,
        risk_per_unit=risk_per_unit,
        qty=qty,
    )


def attach(df_1h: pd.DataFrame, df_4h_tactical: pd.DataFrame, df_1d_bias: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the cascaded context (4H tactical + 1D bias) onto each 1H bar.

    Uses asof-merge for the 4H layer (latest 4H context at-or-before each 1H bar)
    and date-floor merge for the 1D bias.
    """
    df = df_1h.sort_values("timestamp").copy()
    df_4 = df_4h_tactical.sort_values("timestamp").copy()
    df_d = df_1d_bias.copy()

    # 4H asof merge — keep the columns the executor needs
    keep_4h = [
        c for c in [
            "timestamp", "BIAS", "REGIME", "RISK_MODE", "DIR_4H",
            "PLAYBOOK", "adx_4h", "is_compressed",
        ]
        if c in df_4.columns
    ]
    df = pd.merge_asof(
        df, df_4[keep_4h], on="timestamp", direction="backward",
        tolerance=pd.Timedelta("4h"),
    )

    # 1D bias by date (re-merge in case 4H asof produced gaps near boundaries)
    df["date"] = df["timestamp"].dt.floor("D")
    df_d = df_d.assign(date=df_d["timestamp"].dt.floor("D"))
    df = df.merge(
        df_d[["date", "BIAS", "REGIME"]].drop_duplicates("date"),
        on="date", how="left", suffixes=("", "_1d"),
    )
    # Prefer 4H values where present, fall back to daily
    for col in ("BIAS", "REGIME"):
        if f"{col}_1d" in df.columns:
            df[col] = df[col].fillna(df[f"{col}_1d"])
            df.drop(columns=[f"{col}_1d"], inplace=True)
    df[["BIAS", "REGIME"]] = df[["BIAS", "REGIME"]].ffill()
    df.drop(columns=["date"], inplace=True)
    return df
