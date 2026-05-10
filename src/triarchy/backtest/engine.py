"""
Bar-by-bar backtest engine.

Consumes 1H bars enriched with cascaded context (BIAS, PLAYBOOK, RISK_MODE,
DIR_4H, is_compressed, atr_4h) and execution features (ema*, swing levels,
near_ema50, reclaim flags, atr).

Produces a trade log and an equity curve. Models:
  - Slippage in bps on entry and exit
  - Per-side fees on entry and exit notional
  - Intrabar SL/TP fills (SL checked first, conservative)
  - Time stop after `max_hold_bars`

Design choice: the engine itself contains *no strategy logic*. All entry
decisions come from `triarchy.strategy.execution.signal_at` and trade
parameters come from `compute_trade_params`. This keeps the engine
strategy-agnostic and reusable for the future agentic overlay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from triarchy.config import ExecutionConfig
from triarchy.strategy.execution import (
    _apply_slippage,
    compute_trade_params,
    fees_for,
    signal_at,
)


@dataclass
class Trade:
    """A completed trade record."""
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    stop: float
    tp: float
    qty: float
    pnl: float
    r_mult: float
    reason: str          # "TP" | "SL" | "TIME"
    bias: str
    regime: str
    playbook: str


@dataclass
class BacktestResult:
    """Output of a single-symbol backtest."""
    symbol: str
    trades: list[Trade]
    equity_curve: pd.Series  # indexed by timestamp
    final_equity: float

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "symbol", "side", "entry_time", "exit_time",
                    "entry", "exit", "stop", "tp", "qty",
                    "pnl", "r_mult", "reason", "bias", "regime", "playbook",
                ]
            )
        return pd.DataFrame([asdict(t) for t in self.trades])


def _resolve_exit(
    side: str, bar: pd.Series, stop: float, tp: float
) -> tuple[float | None, str | None]:
    """
    Check if SL or TP would have been hit during this bar.

    Conservative tie-break: if both SL and TP are inside the bar's range,
    assume SL fills first (worst-case).
    """
    hi, lo = float(bar["high"]), float(bar["low"])
    if side == "LONG":
        if lo <= stop:
            return stop, "SL"
        if hi >= tp:
            return tp, "TP"
    else:
        if hi >= stop:
            return stop, "SL"
        if lo <= tp:
            return tp, "TP"
    return None, None


def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    cfg: ExecutionConfig,
    starting_equity: float,
) -> BacktestResult:
    """
    Run the backtest on one symbol's enriched 1H DataFrame.

    `df` must already have execution features (compute_features) and cascaded
    context columns (attach). Rows before the slow-EMA warmup are skipped.
    """
    trades: list[Trade] = []
    equity = starting_equity

    # Equity curve — sampled at every bar from warmup onward.
    eq_index: list[pd.Timestamp] = []
    eq_values: list[float] = []

    in_pos = False
    side = ""
    entry = stop = tp = qty = 0.0
    entry_idx: int | None = None
    entry_time: pd.Timestamp | None = None
    bias_at_entry = regime_at_entry = playbook_at_entry = ""

    warmup = cfg.indicators.ema_slow + 10  # safety margin
    n = len(df)

    for i in range(warmup, n - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        # ---------------- Manage open position ----------------
        if in_pos:
            exit_price, reason = _resolve_exit(side, nxt, stop, tp)

            # Time stop
            if exit_price is None and entry_idx is not None:
                if (i + 1 - entry_idx) >= cfg.risk.max_hold_bars:
                    exit_price = float(nxt["close"])
                    reason = "TIME"

            if exit_price is not None and reason is not None:
                exit_exec = _apply_slippage(
                    exit_price, side, is_entry=False, slippage_bps=cfg.costs.slippage_bps
                )
                # PnL accounting
                if side == "LONG":
                    raw = qty * (exit_exec - entry)
                else:
                    raw = qty * (entry - exit_exec)
                cost = fees_for(qty * entry, cfg.costs.fee_rate) + fees_for(
                    qty * exit_exec, cfg.costs.fee_rate
                )
                pnl = raw - cost
                equity += pnl

                risk_per_unit = abs(entry - stop)
                r_mult = pnl / (qty * risk_per_unit) if (qty * risk_per_unit) > 0 else 0.0

                trades.append(
                    Trade(
                        symbol=symbol,
                        side=side,
                        entry_time=entry_time,  # type: ignore[arg-type]
                        exit_time=nxt["timestamp"],
                        entry=entry,
                        exit=exit_exec,
                        stop=stop,
                        tp=tp,
                        qty=qty,
                        pnl=pnl,
                        r_mult=r_mult,
                        reason=reason,
                        bias=bias_at_entry,
                        regime=regime_at_entry,
                        playbook=playbook_at_entry,
                    )
                )

                in_pos = False
                side = ""
                entry = stop = tp = qty = 0.0
                entry_idx = None
                entry_time = None

            eq_index.append(row["timestamp"])
            eq_values.append(equity)
            continue

        # ---------------- Look for entry ----------------
        sig = signal_at(row, cfg)
        if sig is not None:
            swing_level = (
                float(row["swing_low"]) if sig.side == "LONG" else float(row["swing_high"])
            )
            atr_value = float(row["atr"])
            params = compute_trade_params(
                side=sig.side,
                next_open=float(nxt["open"]),
                swing_level=swing_level,
                atr_value=atr_value,
                equity=equity,
                risk_mult=sig.risk_mult,
                cfg=cfg,
            )
            if params is not None:
                in_pos = True
                side = params.side
                entry = params.entry
                stop = params.stop
                tp = params.take_profit
                qty = params.qty
                entry_idx = i + 1
                entry_time = nxt["timestamp"]
                bias_at_entry = str(row.get("BIAS", ""))
                regime_at_entry = str(row.get("REGIME", ""))
                playbook_at_entry = str(row.get("PLAYBOOK", ""))

        eq_index.append(row["timestamp"])
        eq_values.append(equity)

    # If still in position at end-of-data, mark to last close (no synthetic close-trade).
    equity_curve = pd.Series(eq_values, index=pd.DatetimeIndex(eq_index, name="timestamp"))

    return BacktestResult(
        symbol=symbol,
        trades=trades,
        equity_curve=equity_curve,
        final_equity=equity,
    )


def merge_equity_curves(curves: dict[str, pd.Series], starting_equity: float) -> pd.Series:
    """
    Combine per-symbol equity curves into a portfolio curve.

    The trading model is sequential per symbol (each backtest_symbol passes its
    final equity into the next). For multi-symbol portfolio analysis we instead
    sum per-bar PnL increments anchored at `starting_equity`.
    """
    if not curves:
        return pd.Series(dtype=float)

    aligned = []
    for sym, curve in curves.items():
        if curve.empty:
            continue
        # Convert per-symbol equity to per-symbol PnL increments
        pnl_inc = curve.diff().fillna(curve.iloc[0] - starting_equity)
        aligned.append(pnl_inc.rename(sym))

    if not aligned:
        return pd.Series(dtype=float)

    combined = pd.concat(aligned, axis=1).sort_index().fillna(0.0)
    portfolio = starting_equity + combined.sum(axis=1).cumsum()
    portfolio.name = "portfolio_equity"
    return portfolio
