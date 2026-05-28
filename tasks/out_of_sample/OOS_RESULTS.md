# Out-of-Sample Results: Task 1 vs Task 2

## Executive Summary

This report documents the final out-of-sample comparison between the Task 1
baseline strategy and the Task 2 enhanced strategy.

The out-of-sample period is:

```text
2020-01-01 to 2026-05-18
```

The results come from `OOS_testing.ipynb`. Training data is used only as
historical warm-up for rolling indicators and macro state. The reported
performance uses only testing-period returns.

No OOS parameter tuning, SAR grid search, SL multiplier grid search, or
transaction-cost optimization is performed in the OOS notebook.

## Test Discipline

The OOS notebook follows a purged holdout workflow:

- Training-period close prices are used only to warm up indicators such as SMA,
  EMA, RSI, ATR, Bollinger Bands, SAR state inputs, and macro rolling signals.
- OOS close prices are reconstructed from `data/testing/returns.csv`, anchored
  to the final training close.
- Pre-OOS Layer 2 entries are purged before positions are built.
- Both strategies start flat at the OOS boundary.
- Performance tables and charts are sliced strictly to OOS dates.
- A 310-bar embargo sensitivity is reported as a robustness check.

The 310-bar embargo is based on the longest effective rolling warm-up in the
enhanced strategy:

```text
SMA200 + max Layer 1 slope lookback 110 = 310 bars
```

## Strategy Descriptions

### Task 1: Baseline Trend-Following Strategy

Task 1 is the baseline commodity trend-following strategy. It uses a two-layer
signal process:

- Layer 1 identifies persistent trend regimes:
  - Long when SMA50 is above SMA200 and the SMA200 slope is positive.
  - Short when SMA50 is below SMA200 and the SMA200 slope is negative.
- Layer 2 times entries using EMA and RSI:
  - Long entries require a price cross above EMA14 and RSI confirmation.
  - Short entries require a price cross below EMA14 and RSI confirmation.
- The slot manager holds at most one position per commodity.
- Exits come from:
  - Fixed ATR stop.
  - Bollinger Band profit exit.
  - Regime mismatch.
- Portfolio sizing uses ATR stop distance, a per-position max weight, and a
  gross leverage cap.

Task 1 is useful because it is simple, interpretable, and provides a clean
baseline for judging whether Task 2's extra complexity adds value.

### Task 2: Enhanced Trend-Following Strategy

Task 2 keeps the Task 1 strategy identity but adds several enhancements:

- Multi-lookback Layer 1 conviction instead of a single slope lookback.
- Macro conviction modulation using FX, equity, credit, and FX volatility data.
- Stricter short RSI filtering, including an execution-bar short RSI cap.
- Close-price Parabolic SAR-style trailing exits with ATR-based initial
  placement.
- Conviction-weighted sizing.
- Transaction-cost-aware net returns.

Task 2 is not a different strategy family. It is still a commodity
trend-following strategy, but it tries to improve robustness by sizing stronger
regimes more heavily, reducing low-quality shorts, and replacing the fixed stop
with a trailing SAR-style exit.

## Enhancement Comparison

| Component | Task 1 | Task 2 | Why Task 2 Changed It |
|---|---|---|---|
| Layer 1 regime | Binary SMA50/SMA200 trend plus one SMA200 slope lookback. | Binary regime plus continuous multi-lookback `regime_strength`. | Measures conviction across several horizons instead of relying on one slope window. |
| Conviction | No continuous conviction score. | Conviction ranges from weak to strong based on lookback agreement. | Allows position size to reflect signal strength. |
| Macro input | None. | FX, equity, credit, and FX-vol inputs can scale conviction. | Macro affects sizing when conditions support or oppose the commodity trend. |
| Macro direction | Not applicable. | Macro does not flip trade direction. | Preserves commodity trend as the source of long vs short decisions. |
| Entry timing | EMA14 crossover plus RSI confirmation. | Same structure, with stricter short RSI logic. | Keeps the Task 1 timing style while reducing poor short entries. |
| Short RSI cap | No execution-bar oversold cap. | Short can execute only if execution-bar RSI is at least `50`. | Prevents entering shorts after the move is already too stretched. |
| Stop logic | Fixed ATR stop based on entry price. | Close-price SAR-style trailing stop with ATR-based initial placement. | Lets the stop trail favorable trends rather than staying fixed. |
| SAR implementation | Not used. | SAR-style exit only; not full textbook Wilder high/low PSAR. | Provides a trend-following trailing exit while matching the available close-price data. |
| Position sizing | ATR-based risk sizing. | ATR-based sizing scaled by `abs(regime_strength)`. | Reduces size in weak regimes and increases size in strong regimes. |
| Transaction costs | Gross returns only in the baseline path. | Net returns after one-way turnover costs, default `2` bps. | Makes the enhanced result more realistic. |
| Trade stacking | One active position per commodity. | One active position per commodity. | Keeps position management comparable across tasks. |
| Leverage cap | Gross leverage cap. | Same gross leverage cap. | Keeps portfolio-level risk control consistent. |

## OOS Performance

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Task 1 OOS | 5.52% | 15.95% | 0.346 | 0.451 | 0.166 | -33.32% | 0.502 |
| Task 2 OOS Net | 5.75% | 12.74% | 0.451 | 0.568 | 0.216 | -26.62% | 0.501 |
| Task 2 OOS Gross | 6.06% | 12.74% | 0.476 | 0.597 | 0.233 | -25.98% | 0.501 |
| EW Factor OOS | 11.25% | 16.20% | 0.695 | 0.764 | 0.266 | -42.36% | 0.559 |

## Trading Diagnostics

| Strategy | Entries | Short Entries | Avg Gross Exposure | Long Contribution Ann. | Short Contribution Ann. | Ann. Cost Drag |
|---|---:|---:|---:|---:|---:|---:|
| Task 1 OOS | 554 | 201 | 1.069 | 11.30% | -5.78% | n/a |
| Task 2 OOS Net | 596 | 138 | 0.691 | 10.07% | -4.01% | 0.32% |

Task 2 traded slightly more often than Task 1, but with fewer short entries and
lower average gross exposure. Its long book was the main return source, while
the short book detracted less than Task 1's short book.

## Embargo Sensitivity

The 310-bar embargo removes the first part of the OOS period and starts the
evaluation on:

```text
2021-03-10
```

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Calmar | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Task 1 OOS Embargoed | 8.35% | 16.13% | 0.518 | 0.665 | 0.251 | -33.32% |
| Task 2 OOS Net Embargoed | 7.11% | 13.06% | 0.544 | 0.686 | 0.267 | -26.62% |
| Task 2 OOS Gross Embargoed | 7.44% | 13.07% | 0.569 | 0.714 | 0.286 | -25.98% |
| EW Factor OOS Embargoed | 13.97% | 14.11% | 0.990 | 1.344 | 0.704 | -19.85% |

After the 310-bar embargo, Task 2 still has a higher Sharpe than Task 1:

```text
Task 2 OOS Net Sharpe: 0.544
Task 1 OOS Sharpe:    0.518
```

This means the Task 2 improvement over Task 1 is directionally stable after the
warm-up-sensitive part of the OOS period is removed.

## Benchmark Interpretation

The EW Factor is an equal-weight passive commodity benchmark:

```python
factor_oos = test_returns.mean(axis=1)
```

It is not a strategy with the same constraints as Task 1 or Task 2. It is
always long commodity beta, has no entry rules, no exit rules, no short book, no
position-level risk sizing, and no transaction costs.

The EW Factor performed strongly because the OOS period included a powerful
commodity bull cycle. Its annual return was high, but it also experienced the
largest full-period drawdown:

```text
EW Factor OOS Max Drawdown: -42.36%
Task 2 OOS Net Max Drawdown: -26.62%
Task 1 OOS Max Drawdown: -33.32%
```

So the benchmark is useful as passive commodity beta, but it is not a
like-for-like trading strategy comparison.

## Findings

Task 2 improves risk-adjusted performance versus Task 1:

```text
Task 2 OOS Net Sharpe: 0.451
Task 1 OOS Sharpe:    0.346
```

Task 2 also reduces drawdown versus Task 1:

```text
Task 2 OOS Net Max Drawdown: -26.62%
Task 1 OOS Max Drawdown:    -33.32%
```

Transaction costs are visible but not dominant:

```text
Task 2 OOS Gross Sharpe: 0.476
Task 2 OOS Net Sharpe:   0.451
Annual cost drag:        0.32%
```

Task 2 does not beat the EW Factor on Sharpe, but it has a materially lower
drawdown. The EW Factor is a passive long-only commodity beta benchmark, while
Task 2 is a rule-based trend-following strategy with lower exposure, short
trades, exits, and transaction costs.

Overall, the OOS results support the main Task 2 thesis relative to Task 1:
the enhancements improve Sharpe and reduce drawdown. They do not prove that
Task 2 dominates passive long commodity beta during this particular OOS window.

## Final Interpretation

Task 1 remains the cleaner baseline. Task 2 adds complexity, but the OOS results
show that the complexity is doing useful work versus the baseline:

- Higher Sharpe.
- Better Sortino.
- Better Calmar.
- Lower max drawdown.
- Less harmful short book.
- Stable ranking after a 310-bar embargo.

The correct conclusion is not that Task 2 is the highest-return approach in the
OOS period. The correct conclusion is that Task 2 is a better risk-adjusted
trend-following strategy than Task 1 under the final frozen settings.
