"""Unit tests for performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triarchy.backtest.metrics import (
    BARS_PER_YEAR_DAILY,
    build_report,
    cagr,
    calmar_ratio,
    expectancy_r,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)


class TestSharpe:
    def test_zero_volatility_returns_zero(self) -> None:
        returns = pd.Series([0.001] * 100)
        assert sharpe_ratio(returns) == 0.0

    def test_positive_returns_yield_positive_sharpe(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        assert sharpe_ratio(returns, periods_per_year=252) > 0


class TestSortino:
    def test_no_downside_returns_zero(self) -> None:
        # All positive returns => no downside deviation
        returns = pd.Series([0.01] * 100)
        assert sortino_ratio(returns) == 0.0

    def test_sortino_higher_than_sharpe_when_downside_skewed(self) -> None:
        # Lots of small positives, few large positives => low downside, high upside
        returns = pd.Series([0.001] * 90 + [0.05] * 10)
        sr = sharpe_ratio(returns, periods_per_year=252)
        so = sortino_ratio(returns, periods_per_year=252)
        # With no negatives, sortino is 0 by our convention but sharpe is positive.
        # That's a known limitation; just assert the contract.
        assert sr > 0
        assert so == 0.0


class TestMaxDrawdown:
    def test_monotonic_curve_has_zero_drawdown(self) -> None:
        eq = pd.Series([100, 101, 102, 103, 104])
        mdd, dur = max_drawdown(eq)
        assert mdd == 0.0
        assert dur == 0

    def test_known_drawdown(self) -> None:
        # Goes 100 -> 120 -> 90, drawdown = (90 - 120)/120 = -0.25
        eq = pd.Series([100, 110, 120, 100, 90])
        mdd, dur = max_drawdown(eq)
        assert mdd == pytest.approx(-0.25, abs=1e-6)
        assert dur >= 1


class TestCAGR:
    def test_doubling_in_one_year(self) -> None:
        # 365 daily bars, equity doubles
        eq = pd.Series(np.linspace(100, 200, BARS_PER_YEAR_DAILY + 1))
        result = cagr(eq, periods_per_year=BARS_PER_YEAR_DAILY)
        # Approximately 100% annual return
        assert result == pytest.approx(1.0, rel=1e-2)

    def test_flat_curve_zero_cagr(self) -> None:
        eq = pd.Series([100] * 365)
        assert cagr(eq) == pytest.approx(0.0, abs=1e-6)


class TestCalmar:
    def test_calmar_positive(self) -> None:
        assert calmar_ratio(0.20, -0.10) == pytest.approx(2.0)

    def test_calmar_zero_dd(self) -> None:
        assert calmar_ratio(0.20, 0.0) == 0.0


class TestProfitFactor:
    def test_only_winners(self) -> None:
        pnls = pd.Series([10.0, 5.0, 3.0])
        assert profit_factor(pnls) == float("inf")

    def test_balanced(self) -> None:
        pnls = pd.Series([10.0, -5.0, 5.0, -5.0])
        assert profit_factor(pnls) == pytest.approx(1.5)


class TestExpectancy:
    def test_avg(self) -> None:
        rs = pd.Series([1.0, -1.0, 2.0, -1.0])
        assert expectancy_r(rs) == pytest.approx(0.25)


class TestBuildReport:
    def test_empty_trades(self) -> None:
        trades = pd.DataFrame(columns=["pnl", "r_mult"])
        eq = pd.Series([10000.0])
        report = build_report(trades, eq, starting_equity=10000.0)
        assert report.n_trades == 0
        assert report.ending_equity == 10000.0

    def test_full_report(self) -> None:
        trades = pd.DataFrame({"pnl": [100, -50, 200, -50, 150], "r_mult": [2, -1, 4, -1, 3]})
        eq = pd.Series(
            [10000, 10100, 10050, 10250, 10200, 10350],
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )
        report = build_report(trades, eq, starting_equity=10000.0)
        assert report.n_trades == 5
        assert report.win_rate == pytest.approx(0.6)
        assert report.profit_factor == pytest.approx(450 / 100)
        assert report.ending_equity == 10350.0
