"""
Typed configuration loader.

JSON is the source of truth (in `configs/`). Each layer (Maestro / Tactical /
Execution) has a typed dataclass mirror so the rest of the codebase can rely on
attribute access (`cfg.indicators.ema_period`) instead of dict lookups.

Keys starting with `_` in the JSON are treated as comments and ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

# ---------- Maestro (1D) ----------

@dataclass(frozen=True)
class MaestroIndicators:
    ema_period: int
    ema_slope_lookback: int
    ema_slope_min: float
    adx_period: int
    adx_threshold_trend: float
    atr_period: int
    crash_deviation: float


@dataclass(frozen=True)
class MaestroConfig:
    assets: list[str]
    warmup_days: int
    indicators: MaestroIndicators


# ---------- Tactical (4H) ----------

@dataclass(frozen=True)
class TacticalEMA:
    fast: int
    mid: int
    slow: int


@dataclass(frozen=True)
class TacticalADX:
    period: int
    trend_on: float
    range_on: float


@dataclass(frozen=True)
class TacticalATR:
    period: int


@dataclass(frozen=True)
class TacticalBollinger:
    period: int
    std: float
    width_low_pct: float


@dataclass(frozen=True)
class TacticalSlope:
    lookback: int
    min: float


@dataclass(frozen=True)
class TacticalRange:
    lookback: int
    edge_atr_tol: float
    min_atr: float
    max_atr: float
    max_pct: float


@dataclass(frozen=True)
class TacticalConfig:
    warmup_days: int
    ema: TacticalEMA
    adx: TacticalADX
    atr: TacticalATR
    bollinger: TacticalBollinger
    slope: TacticalSlope
    range: TacticalRange


# ---------- Execution (1H) ----------

@dataclass(frozen=True)
class ExecutionIndicators:
    ema_fast: int
    ema_mid: int
    ema_slow: int
    atr_period: int


@dataclass(frozen=True)
class ExecutionEntry:
    pullback_lookback: int
    confirm_reclaim: bool
    near_ema50_atr_mult: float


@dataclass(frozen=True)
class ExecutionRisk:
    per_trade_pct: float
    leverage: float
    sl_atr_cap: float
    tp_r_mult: float
    max_hold_bars: int


@dataclass(frozen=True)
class ExecutionCosts:
    fee_rate: float
    slippage_bps: float


@dataclass(frozen=True)
class ExecutionConfig:
    warmup_days: int
    indicators: ExecutionIndicators
    entry: ExecutionEntry
    risk: ExecutionRisk
    costs: ExecutionCosts
    starting_equity: float


# ---------- Loader ----------

T = TypeVar("T")


def _strip_comments(d: Any) -> Any:
    """Recursively drop keys starting with underscore (treated as comments)."""
    if isinstance(d, dict):
        return {k: _strip_comments(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [_strip_comments(x) for x in d]
    return d


def _from_dict(cls: type[T], data: dict) -> T:
    """
    Instantiate a (possibly nested) dataclass from a dict.

    Raises a clear error if a required key is missing or unexpected.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]

    field_map = {f.name: f for f in fields(cls)}
    extras = set(data.keys()) - set(field_map.keys())
    if extras:
        raise ValueError(f"Unexpected keys for {cls.__name__}: {sorted(extras)}")

    kwargs = {}
    for name, f in field_map.items():
        if name not in data:
            if f.default is not field(default=None).default or f.default_factory is not field(default_factory=lambda: None).default_factory:  # type: ignore[misc]
                continue
            raise ValueError(f"Missing key '{name}' for {cls.__name__}")
        value = data[name]
        ftype = f.type
        # Resolve forward references / string annotations to the class
        if isinstance(ftype, str):
            ftype = globals().get(ftype, ftype)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]


def _load_json(path: Path | str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return _strip_comments(raw)


def load_maestro_config(path: Path | str) -> MaestroConfig:
    return _from_dict(MaestroConfig, _load_json(path))


def load_tactical_config(path: Path | str) -> TacticalConfig:
    return _from_dict(TacticalConfig, _load_json(path))


def load_execution_config(path: Path | str) -> ExecutionConfig:
    return _from_dict(ExecutionConfig, _load_json(path))
