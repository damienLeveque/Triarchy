"""
Performance metrics for backtest evaluation.

All metrics are computed from a trade log and/or an equity curve. They follow
standard definitions used in systematic trading literature (Bailey & López de
Prado, "The Sharpe Ratio Efficient Frontier"; Chan, "Quantitative Trading").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Bars-per-year for annualization. Crypto trades 24/7.
BARS_PER_YEAR_DAILY = 365
BARS_PER_YEAR_4H = 365 * 6
BARS_PER_YEAR_1H = 365 * 24


@dataclass
class PerformanceReport:
    """Aggregate performance metrics for a backtest run."""

    n_trades: int
    win_rate: float
    avg_r: float
    expectancy_r: float
    profit_factor: float

    total_return_pct: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float

    max_drawdown_pct: float
    max_drawdown_duration_days: int

    starting_equity: float
    ending_equity: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _annualization_factor(periods_per_year: int) -> float:
    return float(np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = BARS_PER_YEAR_DAILY, rf: float = 0.0
) -> float:
    """
    Annualized Sharpe ratio.

    `returns` is a series of *periodic* (e.g. daily) returns, not cumulative.
    `rf` is the per-period risk-free rate (default 0).
    """
    excess = returns - rf
    std = excess.std(ddof=1)
    # Treat numerically-zero std as zero (constant returns)
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * _annualization_factor(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int = BARS_PER_YEAR_DAILY, rf: float = 0.0
) -> float:
    """
    Annualized Sortino ratio. Penalizes only downside volatility.
    """
    excess = returns - rf
    downside = excess[excess < 0]
    dd_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if not np.isfinite(dd_std) or dd_std < 1e-12:
        return 0.0
    return float(excess.mean() / dd_std * _annualization_factor(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """
    Maximum drawdown of an equity curve.

    Returns (max_dd_pct, duration_in_bars).
    `max_dd_pct` is negative (e.g. -0.23 means a 23% drawdown).
    """
    if len(equity_curve) == 0:
        return 0.0, 0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    # Duration of the longest drawdown
    is_dd = drawdown < 0
    if not is_dd.any():
        return 0.0, 0

    # Compute longest consecutive run of drawdown bars
    groups = (is_dd != is_dd.shift()).cumsum()
    durations = is_dd.groupby(groups).sum()
    max_duration = int(durations.max())

    return max_dd, max_duration


def calmar_ratio(cagr_value: float, max_dd_pct: float) -> float:
    """CAGR divided by absolute max drawdown."""
    if max_dd_pct == 0:
        return 0.0
    return float(cagr_value / abs(max_dd_pct))


def cagr(equity_curve: pd.Series, periods_per_year: int = BARS_PER_YEAR_DAILY) -> float:
    """Compound Annual Growth Rate from an equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    start, end = float(equity_curve.iloc[0]), float(equity_curve.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    n_periods = len(equity_curve) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1)


def profit_factor(trade_pnls: pd.Series) -> float:
    """Sum of winners divided by absolute sum of losers."""
    wins = trade_pnls[trade_pnls > 0].sum()
    losses = trade_pnls[trade_pnls < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / abs(losses))


def expectancy_r(r_multiples: pd.Series) -> float:
    """Average R per trade — expected payoff in risk units."""
    if len(r_multiples) == 0:
        return 0.0
    return float(r_multiples.mean())


def build_report(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    starting_equity: float,
    periods_per_year: int = BARS_PER_YEAR_DAILY,
) -> PerformanceReport:
    """
    Compose a full performance report from trades and an equity curve.

    `trades_df` requires columns: pnl, r_mult.
    `equity_curve` must be indexed by timestamp at uniform `periods_per_year` frequency.
    """
    n = len(trades_df)
    if n == 0:
        return PerformanceReport(
            n_trades=0,
            win_rate=0.0,
            avg_r=0.0,
            expectancy_r=0.0,
            profit_factor=0.0,
            total_return_pct=0.0,
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            starting_equity=starting_equity,
            ending_equity=starting_equity,
        )

    pnls = trades_df["pnl"]
    rs = trades_df["r_mult"]
    win_rate = float((pnls > 0).mean())
    avg_r = float(rs.mean())
    pf = profit_factor(pnls)
    expectancy = expectancy_r(rs)

    ending_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else starting_equity
    total_return_pct = (ending_equity - starting_equity) / starting_equity

    returns = equity_curve.pct_change().dropna()
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino = sortino_ratio(returns, periods_per_year=periods_per_year)
    cagr_val = cagr(equity_curve, periods_per_year=periods_per_year)

    mdd, mdd_dur = max_drawdown(equity_curve)
    cal = calmar_ratio(cagr_val, mdd)

    return PerformanceReport(
        n_trades=n,
        win_rate=win_rate,
        avg_r=avg_r,
        expectancy_r=expectancy,
        profit_factor=pf,
        total_return_pct=total_return_pct,
        cagr=cagr_val,
        sharpe=sr,
        sortino=sortino,
        calmar=cal,
        max_drawdown_pct=mdd,
        max_drawdown_duration_days=mdd_dur,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
    )
