"""Visualization utilities for backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_equity_curve(equity: pd.Series, out_path: Path, title: str = "Equity curve") -> None:
    """Plot the equity curve over time."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(equity.index, equity.values, linewidth=1.4, color="#1f77b4")
    ax.fill_between(equity.index, equity.values, equity.iloc[0], alpha=0.08, color="#1f77b4")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_drawdown(equity: pd.Series, out_path: Path) -> None:
    """Plot the drawdown curve (always <= 0)."""
    _setup_style()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.35)
    ax.plot(drawdown.index, drawdown.values, linewidth=0.9, color="#8b0000")
    ax.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_monthly_returns_heatmap(equity: pd.Series, out_path: Path) -> None:
    """Heatmap of monthly returns (rows = year, cols = month)."""
    _setup_style()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return
    table = (
        monthly.to_frame("ret")
        .assign(year=lambda d: d.index.year, month=lambda d: d.index.month)
        .pivot_table(index="year", columns="month", values="ret")
    )
    fig, ax = plt.subplots(figsize=(11, 0.55 * max(len(table), 2) + 1.5))
    sns.heatmap(
        table,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Return (%)"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("Monthly returns (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_r_distribution(trades_df: pd.DataFrame, out_path: Path) -> None:
    """Histogram of R-multiples per trade."""
    _setup_style()
    if trades_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    rs = trades_df["r_mult"]
    ax.hist(rs, bins=40, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(rs.mean(), color="#d62728", linewidth=1.5, label=f"Mean R = {rs.mean():.2f}")
    ax.set_title("R-multiple distribution per trade", fontsize=13, fontweight="bold")
    ax.set_xlabel("R")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_performance_by_regime(trades_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of avg R by REGIME (TRENDING / RANGING / CRASH)."""
    _setup_style()
    if trades_df.empty or "regime" not in trades_df.columns:
        return
    grouped = trades_df.groupby("regime")["r_mult"].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = {"TRENDING": "#2ca02c", "RANGING": "#ff7f0e", "CRASH": "#d62728"}
    colors = [palette.get(r, "#7f7f7f") for r in grouped["regime"]]
    ax.bar(grouped["regime"], grouped["mean"], color=colors, edgecolor="white")
    for i, (mean, count) in enumerate(zip(grouped["mean"], grouped["count"], strict=False)):
        ax.text(i, mean, f"n={count}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Average R-multiple by regime", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg R")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_all(
    equity: pd.Series,
    trades_df: pd.DataFrame,
    out_dir: Path,
    year: int | str = "all",
) -> None:
    """Render the full plot suite into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_equity_curve(equity, out_dir / "equity_curve.png", title=f"Equity curve — {year}")
    plot_drawdown(equity, out_dir / "drawdown.png")
    plot_monthly_returns_heatmap(equity, out_dir / "monthly_returns.png")
    plot_r_distribution(trades_df, out_dir / "r_distribution.png")
    plot_performance_by_regime(trades_df, out_dir / "by_regime.png")
