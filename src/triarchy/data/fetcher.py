"""
OHLCV data fetcher with disk caching.

Consolidates the multiple fetch scripts into a single parameterized module.
Cache layout:
    DATA_CACHE/{year}/{timeframe}/{SYMBOL}_{timeframe}_{year}.csv

Cache files store data with a configurable warmup buffer prepended so
indicators (EMA200, ATR14, etc.) have warmed up by the time the target year
starts.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ccxt
import pandas as pd

DEFAULT_CACHE_ROOT = Path("DATA_CACHE")


def _safe_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT' -> 'BTCUSDT' for filesystem-safe filenames."""
    return symbol.replace("/", "")


def cache_path(
    symbol: str, timeframe: str, year: int, root: Path = DEFAULT_CACHE_ROOT
) -> Path:
    return root / str(year) / timeframe / f"{_safe_symbol(symbol)}_{timeframe}_{year}.csv"


def _load_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _is_cache_complete(df: pd.DataFrame, end_dt: datetime, timeframe: str) -> bool:
    """Cache is considered complete if it reaches within ~one bar of end_dt."""
    if df.empty:
        return False
    tolerance = {
        "1d": pd.Timedelta(days=1),
        "4h": pd.Timedelta(hours=4),
        "1h": pd.Timedelta(hours=1),
        "15m": pd.Timedelta(minutes=15),
    }.get(timeframe, pd.Timedelta(hours=1))
    return df["timestamp"].max() >= pd.Timestamp(end_dt) - tolerance


def _fetch_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV in chunks, respecting exchange rate limits."""
    rows: list[list] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if batch[-1][0] >= until_ms:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_year(
    symbol: str,
    timeframe: str,
    year: int,
    warmup_days: int,
    exchange: ccxt.Exchange | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV for one symbol/timeframe/year (with warmup buffer prepended).

    Returns the full DataFrame from cache if present and complete; otherwise
    fetches from the exchange and writes the cache file.
    """
    out_path = cache_path(symbol, timeframe, year, root=cache_root)
    end_dt = datetime(year + 1, 1, 1, tzinfo=UTC)

    if not force:
        cached = _load_cached(out_path)
        if cached is not None and _is_cache_complete(cached, end_dt, timeframe):
            return cached

    if exchange is None:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

    start_dt = datetime(year, 1, 1, tzinfo=UTC) - timedelta(days=warmup_days)
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    df = _fetch_paginated(exchange, symbol, timeframe, since_ms, until_ms)
    if df.empty:
        return df

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def fetch_many(
    symbols: list[str],
    timeframe: str,
    year: int,
    warmup_days: int,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple symbols. Returns {symbol: df}."""
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if verbose:
            print(f"  Fetching {sym} {timeframe} {year}...", end=" ", flush=True)
        df = fetch_year(
            sym, timeframe, year, warmup_days,
            exchange=exchange, cache_root=cache_root, force=force,
        )
        if verbose:
            print(f"{len(df)} rows" if not df.empty else "EMPTY")
        out[sym] = df
    return out


def load_cached_only(
    symbol: str, timeframe: str, year: int, cache_root: Path = DEFAULT_CACHE_ROOT
) -> pd.DataFrame | None:
    """Read from cache without fetching. Returns None if missing."""
    return _load_cached(cache_path(symbol, timeframe, year, root=cache_root))
