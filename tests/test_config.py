"""Tests for config loading (typed dataclasses from JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triarchy.config import (
    load_execution_config,
    load_maestro_config,
    load_tactical_config,
)


class TestMaestroConfig:
    def test_loads_real_config(self) -> None:
        cfg = load_maestro_config("configs/maestro.json")
        assert len(cfg.assets) > 0
        assert cfg.warmup_days >= 200
        assert cfg.indicators.ema_period == 200

    def test_strips_comment_keys(self, tmp_path: Path) -> None:
        data = {
            "_comment": "this should be ignored",
            "assets": ["BTC/USDT"],
            "warmup_days": 365,
            "indicators": {
                "_doc": "ignored too",
                "ema_period": 200,
                "ema_slope_lookback": 5,
                "ema_slope_min": 0.0005,
                "adx_period": 14,
                "adx_threshold_trend": 25,
                "atr_period": 14,
                "crash_deviation": 0.20,
            },
            "_bias_rules": {"foo": "bar"},
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data))
        cfg = load_maestro_config(path)
        assert cfg.indicators.ema_period == 200

    def test_unexpected_key_raises(self, tmp_path: Path) -> None:
        data = {
            "assets": ["BTC/USDT"],
            "warmup_days": 365,
            "indicators": {
                "ema_period": 200,
                "ema_slope_lookback": 5,
                "ema_slope_min": 0.0005,
                "adx_period": 14,
                "adx_threshold_trend": 25,
                "atr_period": 14,
                "crash_deviation": 0.20,
            },
            "this_should_not_exist": True,
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Unexpected keys"):
            load_maestro_config(path)


class TestTacticalConfig:
    def test_loads_real_config(self) -> None:
        cfg = load_tactical_config("configs/tactical.json")
        assert cfg.ema.fast == 20
        assert cfg.ema.slow == 200
        assert cfg.adx.trend_on > cfg.adx.range_on


class TestExecutionConfig:
    def test_loads_real_config(self) -> None:
        cfg = load_execution_config("configs/execution.json")
        assert cfg.starting_equity > 0
        assert 0 < cfg.risk.per_trade_pct < 0.05
        assert cfg.risk.tp_r_mult > 0
