# Code Summary

This project is organised as a commodity trend-following backtesting pipeline:
data preparation -> indicators -> signals -> positions -> portfolio weights ->
returns -> diagnostics -> final OOS evaluation. The code is mostly functional;
there are no core strategy classes.

## Pipeline

| Stage | Main files | What happens | Main output |
|---|---|---|---|
| Data prep | `tasks/task_0.ipynb`, `tasks/task_2/task_2_data_cleaning.ipynb` | Builds commodity returns and aligned macro datasets, split into training and testing. | CSV files under `data/raw`, `data/training`, `data/testing`, and `data/external`. |
| Indicators | `tasks/task_1/indicators.py` | Calculates `sma`, `ema`, `atr_pct`, `bollinger_bands`, `sma_slope`, `rsi`, `expanding_quantile`. | Indicator `DataFrame`s aligned to dates/assets; Bollinger returns `(mid, upper, lower)`. |
| Signals | `signals/signal_layer*.py`, `tasks/task_2/signals/macro_signal.py` | Layer 1 defines trend direction; Layer 2 defines entry timing. Task 2 adds regime conviction and macro overlays. | Signal grids: `+1`, `-1`, `0`; Task 2 also returns continuous `regime_strength`. |
| Positions | `portfolio/slot_manager*.py` | Converts signals into active trades and applies exits. Task 1 uses fixed ATR/Bollinger/regime exits; Task 2 adds execution-time RSI cap and SAR-style trailing exits. | Position `DataFrame` with `+1` long, `-1` short, `0` flat. |
| Portfolio | `portfolio/portfolio_manager*.py` | Sizes trades, caps exposure, applies execution lag, and calculates returns. Task 2 also applies conviction sizing and transaction costs. | `(weights, portfolio_returns)`; Task 2 return `Series` also stores gross returns and costs in `.attrs`. |
| Strategy wrappers | `tasks/task_1/strategy.py`, `tasks/task_2/strategy_enhanced.py` | Runs the full strategy pipeline end to end. | `build_signals()` returns signal dictionaries; `run_strategy()` returns `(positions, weights, returns)`. |
| Analytics | `tasks/task_1/analytics.py`, `tasks/task_2/diagnostics.py` | Produces performance stats, plots, rolling Sharpe, position summaries, cost sensitivity, SAR grid, and SL multiplier grid. | Stats `Series`/`DataFrame`, ranked diagnostic tables, and plots. |
| OOS testing | `OOS_testing.ipynb`, `OOS_RESULTS.md` | Runs frozen Task 1 and Task 2 strategies on the holdout period only. | OOS performance tables, diagnostics, charts, and written findings. |

## Key Files And Functions

| File | Main functions | Role and returns |
|---|---|---|
| `tasks/task_1/indicators.py` | `sma`, `ema`, `atr_pct`, `bollinger_bands`, `sma_slope`, `rsi`, `expanding_quantile` | Shared indicator calculations returning aligned numeric `DataFrame`s. |
| `tasks/task_1/signals/signal_layer1.py` | `layer1_signal` | Baseline SMA trend regime; returns `+1/-1/0` regime grid. |
| `tasks/task_1/signals/signal_layer2.py` | `layer2_signal` | EMA/RSI entry timing; returns `+1/-1/0` entry-trigger grid. |
| `tasks/task_1/portfolio/slot_manager.py` | `build_positions` | Opens one position per commodity and applies Task 1 exits; returns position grid. |
| `tasks/task_1/portfolio/portfolio_manager.py` | `build_portfolio` | ATR-sized weights and returns; returns `(weights, portfolio_returns)`. |
| `tasks/task_1/strategy.py` | `build_signals`, `run_strategy` | Baseline strategy wrapper; returns signal dict or `(positions, weights, returns)`. |
| `tasks/task_1/analytics.py` | `performance_stats`, `position_summary`, `plot_signals`, `plot_comparison`, `rolling_sharpe`, `plot_rolling_sharpe`, `compare_strategies` | Reporting helpers; return tables, rolling Sharpe data, or plots. |
| `tasks/task_2/signals/macro_signal.py` | `dollar_signal`, `equity_signal`, `credit_signal`, `vol_percentile`, `macro_regime_score`, `blend_regime_strength`, `apply_vol_cap` | Macro conviction overlays; return macro `Series` or adjusted strength grids. |
| `tasks/task_2/signals/signal_layer1_enhanced.py` | `regime_strength`, `layer1_signal` | Multi-lookback Task 2 conviction and binary trend direction; returns strength grid and regime grid. |
| `tasks/task_2/signals/signal_layer2_enhanced.py` | `layer2_signal` | Enhanced EMA/RSI entry timing; returns entry-trigger grid. |
| `tasks/task_2/portfolio/slot_manager_enhanced.py` | `build_positions` | Applies Task 2 entries, RSI short cap, SAR-style exits, and regime exits; returns position grid. |
| `tasks/task_2/portfolio/portfolio_manager_enhanced.py` | `build_portfolio` | Conviction-weighted sizing and transaction costs; returns `(weights, net_returns)` with gross/cost attrs. |
| `tasks/task_2/strategy_enhanced.py` | `build_signals`, `run_strategy` | Enhanced strategy wrapper; returns signal dict or `(positions, weights, net_returns)`. |
| `tasks/task_2/diagnostics.py` | `summarize_run`, `transaction_cost_sensitivity`, `sl_mult_grid_search`, `sar_grid_search` | Training-only diagnostics; return summary/ranked `DataFrame`s. |

## Organisation Notes

Task 1 is the baseline trend-following strategy. Task 2 keeps the same broad
structure but adds multi-lookback conviction, macro sizing overlays, stricter
short filtering, SAR-style exits, conviction-weighted sizing, and transaction
costs. Research notebooks are for training analysis; `OOS_testing.ipynb` is the
separate final holdout evaluation.
