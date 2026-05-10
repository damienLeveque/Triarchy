# Walk-forward analysis methodology

## The problem this addresses

The most common failure mode in systematic trading is **in-sample overfitting**: you tune parameters on data you already have, and they look fantastic in the backtest, but they fall apart on new data. Every retail trading bot ever built has died on this.

Triarchy's parameters were chosen from first principles (standard EMA periods, Wilder's ATR period, conventional ADX thresholds) rather than optimized over a single dataset. To verify this discipline holds, the system is evaluated **year by year** on independent slices.

## What we do

Each year is treated as an **independent out-of-sample test**:

1. The same configuration files (`configs/maestro.json`, `configs/tactical.json`, `configs/execution.json`) are used for every year.
2. No parameters are tuned on any year's results.
3. Each year is fetched, processed, and backtested in isolation.
4. Results are reported separately so we can see whether performance is stable or fragile.

If a strategy works only in one or two regimes, it shows up immediately as inconsistency across years. If it generalises, the metrics stay in a reasonable band.

## What this is *not*

This is **not** rolling walk-forward optimization (e.g. Pardo's anchored or rolling WFO). That would mean:
- Splitting each year into in-sample / out-of-sample halves
- Re-optimizing parameters on the in-sample portion
- Testing on the out-of-sample portion
- Repeating with a sliding window

That methodology is appropriate when parameters are *meant* to adapt over time. Triarchy's design philosophy is the opposite: the parameters are intentionally **regime-agnostic** (EMA200, ATR14, ADX14 — all classic values, not data-mined). So the right test is "does the same config work across very different years?" — which is what the year-by-year independent backtest answers.

A future iteration could add anchored WFO if we introduce learnable parameters (e.g. as part of the agentic overlay).

## Years covered

| Year | Market context (broad)        | Why it matters as a test slice                |
|------|-------------------------------|-----------------------------------------------|
| 2022 | Crypto bear market / drawdown | Tests the CRASH regime detection and RISK_MODE=OFF gating |
| 2023 | Recovery / sideways → up      | Tests the transition from RANGING to TRENDING |
| 2024 | Bull market                    | Tests sustained TREND_FOLLOW playbook performance |
| 2025 | Mixed                          | Out-of-sample stress test                     |

## How to run it

```bash
# Fetch data once for each year
triarchy fetch --year 2022 --timeframes 1d --timeframes 4h --timeframes 1h
triarchy fetch --year 2023 --timeframes 1d --timeframes 4h --timeframes 1h
triarchy fetch --year 2024 --timeframes 1d --timeframes 4h --timeframes 1h
triarchy fetch --year 2025 --timeframes 1d --timeframes 4h --timeframes 1h

# Run the full walk-forward
triarchy run --years 2022 --years 2023 --years 2024 --years 2025
```

Each year produces its own `BACKTEST_RESULTS/{year}/` directory with `summary.csv`, `trades.csv`, `equity_curve.csv`, and a full plot suite. The notebook `notebooks/01_portfolio_analysis.ipynb` aggregates them into a single multi-year report.

## What "good" looks like

A robust system should display:
- **Sharpe stability**: not necessarily high, but in the same ballpark across years (e.g. 0.8–2.0 across 4 years rather than 5.0 / -1.0 / 3.0 / -2.0)
- **Drawdown control**: max drawdown roughly bounded across regimes, no catastrophic year
- **Regime alignment**: trades win disproportionately in TRENDING regimes (which is what TREND_FOLLOW is *supposed* to do) and the engine correctly disables itself in CRASH regimes
- **Number of trades scales with opportunity**: more trades in trending years, fewer in ranging years

A system that's overfit shows **dramatic year-to-year inconsistency** and a single year carrying the entire equity curve.

## Honest limitations

- The 4 years tested are not independent samples in the statistical sense — they're sequential, with autocorrelation in macro conditions.
- Crypto's history is short; we don't have 30 years of regime variation like equities.
- Transaction costs are modeled but slippage is conservative-but-fixed; in stressed conditions real slippage can be much worse.
- The portfolio aggregation assumes equal capital allocation across symbols and ignores correlation.

These are the kinds of limitations that should always be stated up-front in a quant project. The agentic overlay (next phase) will add an evaluation framework comparing rules-only vs rules+agentic across the same year-by-year slices.
