"""
End-to-end pipeline orchestrator.

Runs each layer in sequence and persists intermediate results to disk as CSV
caches. This makes every step independently re-runnable and debuggable: you
can inspect any cache file by hand to understand exactly what each layer
produced.

Cache layout:

    DATA_CACHE/{year}/{tf}/{SYMBOL}_{tf}_{year}.csv      <- raw OHLCV (data layer)
    STRATEGY_DATA/{year}/{SYMBOL}_BIAS.csv               <- Maestro 1D output
    STRATEGY_DATA/{year}/{SYMBOL}_TACTICAL_4H.csv        <- Tactical 4H output
    STRATEGY_DATA/{year}/{SYMBOL}_EXECUTION_1H.csv       <- Execution 1H features + context
    BACKTEST_RESULTS/{year}/trades.csv                   <- per-trade log
    BACKTEST_RESULTS/{year}/summary.csv                  <- aggregate metrics
    BACKTEST_RESULTS/{year}/equity_curve.csv             <- portfolio equity curve
    BACKTEST_RESULTS/{year}/*.png                        <- plots
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from triarchy.backtest.engine import backtest_symbol, merge_equity_curves
from triarchy.backtest.metrics import BARS_PER_YEAR_1H, build_report
from triarchy.backtest.plots import render_all
from triarchy.config import (
    ExecutionConfig,
    MaestroConfig,
    TacticalConfig,
)
from triarchy.data import fetcher
from triarchy.strategy import execution as exec_strategy
from triarchy.strategy import maestro as maestro_strategy
from triarchy.strategy import tactical as tactical_strategy

# ---------- Path helpers ----------

DEFAULT_STRATEGY_ROOT = Path("STRATEGY_DATA")
DEFAULT_RESULTS_ROOT = Path("BACKTEST_RESULTS")


def _safe_sym(symbol: str) -> str:
    return symbol.replace("/", "")


def bias_path(symbol: str, year: int, root: Path = DEFAULT_STRATEGY_ROOT) -> Path:
    return root / str(year) / f"{_safe_sym(symbol)}_BIAS.csv"


def tactical_path(symbol: str, year: int, root: Path = DEFAULT_STRATEGY_ROOT) -> Path:
    return root / str(year) / f"{_safe_sym(symbol)}_TACTICAL_4H.csv"


def execution_path(symbol: str, year: int, root: Path = DEFAULT_STRATEGY_ROOT) -> Path:
    return root / str(year) / f"{_safe_sym(symbol)}_EXECUTION_1H.csv"


def results_dir(year: int, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return root / str(year)


# ---------- Per-layer runners ----------

def run_maestro(
    symbol: str,
    year: int,
    cfg: MaestroConfig,
    cache_root: Path = fetcher.DEFAULT_CACHE_ROOT,
    strategy_root: Path = DEFAULT_STRATEGY_ROOT,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run Maestro 1D for a symbol/year. Reads the OHLCV cache, writes the BIAS CSV.
    """
    out_path = bias_path(symbol, year, root=strategy_root)
    if out_path.exists() and not force:
        df = pd.read_csv(out_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    raw = fetcher.load_cached_only(symbol, "1d", year, cache_root=cache_root)
    if raw is None or raw.empty:
        raise FileNotFoundError(
            f"No 1d cache for {symbol} year={year}. Run `triarchy fetch` first."
        )

    result = maestro_strategy.run(raw, cfg)
    target = result[result["timestamp"].dt.year == year].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    target[maestro_strategy.BIAS_CSV_COLUMNS].to_csv(out_path, index=False)
    return target


def run_tactical(
    symbol: str,
    year: int,
    cfg_tactical: TacticalConfig,
    cfg_maestro: MaestroConfig,
    cache_root: Path = fetcher.DEFAULT_CACHE_ROOT,
    strategy_root: Path = DEFAULT_STRATEGY_ROOT,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run Tactical 4H for a symbol/year. Reads the 4h OHLCV cache and the BIAS
    CSV produced by `run_maestro`.
    """
    out_path = tactical_path(symbol, year, root=strategy_root)
    if out_path.exists() and not force:
        df = pd.read_csv(out_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    raw_4h = fetcher.load_cached_only(symbol, "4h", year, cache_root=cache_root)
    if raw_4h is None or raw_4h.empty:
        raise FileNotFoundError(
            f"No 4h cache for {symbol} year={year}. Run `triarchy fetch` first."
        )

    bias_csv = bias_path(symbol, year, root=strategy_root)
    if not bias_csv.exists():
        # Auto-generate the bias CSV if missing
        run_maestro(symbol, year, cfg_maestro, cache_root, strategy_root, force=False)

    df_1d = pd.read_csv(bias_csv)
    df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"], utc=True)

    result = tactical_strategy.run(raw_4h, df_1d, cfg_tactical)
    target = result[result["timestamp"].dt.year == year].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    target[tactical_strategy.TACTICAL_CSV_COLUMNS].to_csv(out_path, index=False)
    return target


def run_execution_features(
    symbol: str,
    year: int,
    cfg_exec: ExecutionConfig,
    cfg_maestro: MaestroConfig,
    cfg_tactical: TacticalConfig,
    cache_root: Path = fetcher.DEFAULT_CACHE_ROOT,
    strategy_root: Path = DEFAULT_STRATEGY_ROOT,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build the enriched 1H DataFrame: execution features + cascaded context.

    This is the input to the backtest engine.
    """
    out_path = execution_path(symbol, year, root=strategy_root)
    if out_path.exists() and not force:
        df = pd.read_csv(out_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    raw_1h = fetcher.load_cached_only(symbol, "1h", year, cache_root=cache_root)
    if raw_1h is None or raw_1h.empty:
        raise FileNotFoundError(
            f"No 1h cache for {symbol} year={year}. Run `triarchy fetch` first."
        )

    bias_csv = bias_path(symbol, year, root=strategy_root)
    tactical_csv = tactical_path(symbol, year, root=strategy_root)
    if not bias_csv.exists():
        run_maestro(symbol, year, cfg_maestro, cache_root, strategy_root, force=False)
    if not tactical_csv.exists():
        run_tactical(symbol, year, cfg_tactical, cfg_maestro, cache_root, strategy_root, force=False)

    df_1d = pd.read_csv(bias_csv)
    df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"], utc=True)
    df_4h = pd.read_csv(tactical_csv)
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], utc=True)

    df_1h_feat = exec_strategy.compute_features(raw_1h, cfg_exec)
    df_attached = exec_strategy.attach(df_1h_feat, df_4h, df_1d)
    target = df_attached[df_attached["timestamp"].dt.year == year].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(out_path, index=False)
    return target


# ---------- Orchestration ----------

def run_full_year(
    year: int,
    cfg_maestro: MaestroConfig,
    cfg_tactical: TacticalConfig,
    cfg_exec: ExecutionConfig,
    cache_root: Path = fetcher.DEFAULT_CACHE_ROOT,
    strategy_root: Path = DEFAULT_STRATEGY_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    symbols: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full pipeline for one year and persist all intermediates + results.

    Returns a dict with the report and paths to the output files.
    """
    syms = symbols or cfg_maestro.assets

    per_symbol_curves: dict[str, pd.Series] = {}
    per_symbol_trades: list[pd.DataFrame] = []
    final_equities: dict[str, float] = {}

    for sym in syms:
        if verbose:
            print(f"  [{sym}]", end=" ", flush=True)
        try:
            run_maestro(sym, year, cfg_maestro, cache_root, strategy_root, force=force)
            run_tactical(sym, year, cfg_tactical, cfg_maestro, cache_root, strategy_root, force=force)
            df_exec = run_execution_features(
                sym, year, cfg_exec, cfg_maestro, cfg_tactical,
                cache_root, strategy_root, force=force,
            )
        except FileNotFoundError as e:
            if verbose:
                print(f"SKIP ({e})")
            continue

        result = backtest_symbol(sym, df_exec, cfg_exec, cfg_exec.starting_equity)
        per_symbol_curves[sym] = result.equity_curve
        final_equities[sym] = result.final_equity
        if result.trades:
            per_symbol_trades.append(result.trades_df())
        if verbose:
            print(f"trades={len(result.trades)} final=${result.final_equity:,.0f}")

    out_dir = results_dir(year, root=results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist trades
    trades_df = (
        pd.concat(per_symbol_trades, ignore_index=True)
        if per_symbol_trades
        else pd.DataFrame()
    )
    trades_df.to_csv(out_dir / "trades.csv", index=False)

    # Portfolio equity
    portfolio = merge_equity_curves(per_symbol_curves, cfg_exec.starting_equity)
    if not portfolio.empty:
        portfolio.to_frame("equity").to_csv(out_dir / "equity_curve.csv")

    # Performance report
    report = build_report(
        trades_df=trades_df if not trades_df.empty else pd.DataFrame(columns=["pnl", "r_mult"]),
        equity_curve=portfolio if not portfolio.empty else pd.Series([cfg_exec.starting_equity]),
        starting_equity=cfg_exec.starting_equity,
        periods_per_year=BARS_PER_YEAR_1H,
    )
    pd.DataFrame([report.to_dict()]).to_csv(out_dir / "summary.csv", index=False)

    # Plots
    if not portfolio.empty and not trades_df.empty:
        render_all(portfolio, trades_df, out_dir, year=year)

    if verbose:
        print(f"\n  Saved to {out_dir}")
        print(f"  Sharpe={report.sharpe:.2f}  Sortino={report.sortino:.2f}")
        print(f"  CAGR={report.cagr*100:.1f}%  MaxDD={report.max_drawdown_pct*100:.1f}%")
        print(f"  Trades={report.n_trades}  WinRate={report.win_rate*100:.1f}%")

    return {
        "year": year,
        "report": report,
        "out_dir": out_dir,
        "n_symbols": len(per_symbol_curves),
    }
