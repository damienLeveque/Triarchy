"""
Triarchy command-line interface.

Usage:
    triarchy fetch --year 2024 --timeframes 1d 4h 1h
    triarchy maestro --year 2024
    triarchy tactical --year 2024
    triarchy execution --year 2024
    triarchy backtest --year 2024
    triarchy run --year 2024            # end-to-end: maestro + tactical + execution + backtest
    triarchy run --years 2022 2023 2024 2025   # walk-forward across multiple years
"""

from __future__ import annotations

from pathlib import Path

import click

from triarchy.config import (
    load_execution_config,
    load_maestro_config,
    load_tactical_config,
)
from triarchy.data import fetcher
from triarchy.pipeline import (
    DEFAULT_RESULTS_ROOT,
    run_execution_features,
    run_full_year,
    run_maestro,
    run_tactical,
)

CONFIG_DIR_DEFAULT = Path("configs")


def _opt_config_dir():
    return click.option(
        "--config-dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=CONFIG_DIR_DEFAULT,
        show_default=True,
        help="Directory containing maestro.json, tactical.json, execution.json.",
    )


def _load_all_configs(config_dir: Path):
    return (
        load_maestro_config(config_dir / "maestro.json"),
        load_tactical_config(config_dir / "tactical.json"),
        load_execution_config(config_dir / "execution.json"),
    )


@click.group()
def cli() -> None:
    """Triarchy — three-layer hierarchical systematic trading."""


# ---------- fetch ----------

@cli.command()
@click.option("--year", type=int, required=True, help="Target calendar year.")
@click.option(
    "--timeframes",
    multiple=True,
    type=click.Choice(["1d", "4h", "1h", "15m"]),
    default=("1d", "4h", "1h"),
    show_default=True,
    help="Timeframes to fetch (one --timeframes flag per timeframe).",
)
@click.option("--force", is_flag=True, help="Refetch even if cached.")
@_opt_config_dir()
def fetch(year: int, timeframes: tuple[str, ...], force: bool, config_dir: Path) -> None:
    """Fetch OHLCV data for all configured assets."""
    cfg_maestro, cfg_tactical, cfg_exec = _load_all_configs(config_dir)

    warmup_for = {
        "1d": cfg_maestro.warmup_days,
        "4h": cfg_tactical.warmup_days,
        "1h": cfg_exec.warmup_days,
        "15m": 45,
    }

    for tf in timeframes:
        click.echo(f"Fetching {tf} for {year}...")
        fetcher.fetch_many(
            symbols=cfg_maestro.assets,
            timeframe=tf,
            year=year,
            warmup_days=warmup_for[tf],
            force=force,
        )


# ---------- per-layer ----------

@cli.command()
@click.option("--year", type=int, required=True)
@click.option("--symbol", default=None, help="Single symbol; default: all configured.")
@click.option("--force", is_flag=True, help="Recompute even if cached CSV exists.")
@_opt_config_dir()
def maestro(year: int, symbol: str | None, force: bool, config_dir: Path) -> None:
    """Run Maestro 1D layer (BIAS + REGIME)."""
    cfg_maestro, _, _ = _load_all_configs(config_dir)
    syms = [symbol] if symbol else cfg_maestro.assets
    for sym in syms:
        try:
            run_maestro(sym, year, cfg_maestro, force=force)
            click.echo(f"  {sym}: OK")
        except FileNotFoundError as e:
            click.echo(f"  {sym}: SKIP ({e})", err=True)


@cli.command()
@click.option("--year", type=int, required=True)
@click.option("--symbol", default=None)
@click.option("--force", is_flag=True)
@_opt_config_dir()
def tactical(year: int, symbol: str | None, force: bool, config_dir: Path) -> None:
    """Run Tactical 4H layer (PLAYBOOK + DIR_4H + RISK_MODE)."""
    cfg_maestro, cfg_tactical, _ = _load_all_configs(config_dir)
    syms = [symbol] if symbol else cfg_maestro.assets
    for sym in syms:
        try:
            run_tactical(sym, year, cfg_tactical, cfg_maestro, force=force)
            click.echo(f"  {sym}: OK")
        except FileNotFoundError as e:
            click.echo(f"  {sym}: SKIP ({e})", err=True)


@cli.command()
@click.option("--year", type=int, required=True)
@click.option("--symbol", default=None)
@click.option("--force", is_flag=True)
@_opt_config_dir()
def execution(year: int, symbol: str | None, force: bool, config_dir: Path) -> None:
    """Build the 1H execution feature set (entry features + cascaded context)."""
    cfg_maestro, cfg_tactical, cfg_exec = _load_all_configs(config_dir)
    syms = [symbol] if symbol else cfg_maestro.assets
    for sym in syms:
        try:
            run_execution_features(
                sym, year, cfg_exec, cfg_maestro, cfg_tactical, force=force
            )
            click.echo(f"  {sym}: OK")
        except FileNotFoundError as e:
            click.echo(f"  {sym}: SKIP ({e})", err=True)


# ---------- backtest / pipeline ----------

@cli.command()
@click.option("--year", type=int, default=None)
@click.option(
    "--years",
    multiple=True,
    type=int,
    help="Multiple years for walk-forward. Each year runs independently.",
)
@click.option("--force", is_flag=True, help="Recompute all intermediates.")
@_opt_config_dir()
def run(year: int | None, years: tuple[int, ...], force: bool, config_dir: Path) -> None:
    """Run the end-to-end pipeline (maestro -> tactical -> execution -> backtest)."""
    cfg_maestro, cfg_tactical, cfg_exec = _load_all_configs(config_dir)

    target_years = list(years) if years else ([year] if year else [])
    if not target_years:
        raise click.UsageError("Specify --year or --years.")

    for y in target_years:
        click.echo(f"\n=== Year {y} ===")
        run_full_year(
            year=y,
            cfg_maestro=cfg_maestro,
            cfg_tactical=cfg_tactical,
            cfg_exec=cfg_exec,
            force=force,
        )


@cli.command()
@click.option("--year", type=int, required=True)
def show_results(year: int) -> None:
    """Print the saved summary for a year."""
    import pandas as pd
    path = DEFAULT_RESULTS_ROOT / str(year) / "summary.csv"
    if not path.exists():
        click.echo(f"No summary at {path}. Run `triarchy run --year {year}` first.", err=True)
        return
    df = pd.read_csv(path)
    click.echo(df.T.to_string(header=False))


if __name__ == "__main__":
    cli()
