# BUSI97075 Backtesting Project

Commodity strategy backtesting project for BUSI97075. The repository contains a full workflow from data preparation through baseline and enhanced commodity trend-following strategies, out-of-sample testing, and a Task 3 machine-learning extension.

## Project Overview

The project is organised into four task areas:

| Area | Purpose |
|---|---|
| Task 0: EDA | Explore the 23-commodity universe, calculate return summary statistics and correlations, and construct the equal-weight commodity market factor. |
| Task 1: Ideas-First Momentum / Trend Strategy | Design, justify, and evaluate a time-series momentum / trend-following strategy for each commodity, including performance versus the market factor, market-factor regression, and sector contribution. |
| Task 2: Strategy Enhancements | Improve the Task 1 strategy with a limited number of well-motivated, non-data-driven enhancements, including auxiliary macro variables, risk management, portfolio construction changes, and transaction-cost analysis. |
| Task 3: Data-Driven Strategies | Build statistical or machine-learning predictive signal models using auxiliary features, with clear target design, training/validation discipline, portfolio construction, and look-ahead controls. |
| OOS Holdout Evaluation | Run the frozen Task 1 and Task 2 strategies on the out-of-sample period as a final robustness check, without tuning on test data. |

The canonical data inputs are in `source/` and the cleaned project data is in `data/`.

## Repository Structure

```text
.
├── data/
│   ├── raw/                  # cleaned raw commodity data and asset metadata
│   ├── training/             # in-sample returns and prices
│   └── testing/              # out-of-sample returns
├── source/                   # original Excel workbooks
├── docs/                     # assignment brief and project documentation
├── tasks/
│   ├── task_0/               # data cleaning notebook
│   ├── task_1/               # baseline strategy code and notebook
│   ├── task_2/               # enhanced strategy code, diagnostics, notebook
│   ├── task_3/               # MLP/Ridge model notebooks and outputs
│   └── out_of_sample/        # OOS testing notebook and results writeup
└── README.md
```

Key documentation files:

```text
docs/code_summary.md        # one-page map of the codebase and main functions
docs/Enhanced_Stategy.md    # detailed explanation of the Task 2 enhanced strategy
```

## Data

Source workbooks:

```text
source/Commodities Data thru 18May26.xlsx
source/DataTables Updated thru 18May26.xlsx
```

Cleaned project datasets:

```text
data/raw/assets.csv
data/raw/return_indices.csv
data/raw/returns_unaligned.csv
data/training/returns.csv
data/training/close_prices.csv
data/testing/returns.csv
```

Task 1, Task 2, and OOS testing use the root `data/` folder as the canonical cleaned data source.

## Main Strategies

### Task 1: Baseline Trend Following

Task 1 implements an interpretable commodity trend-following baseline:

- Layer 1 trend regime using SMA50/SMA200 and SMA slope.
- Layer 2 entry timing using EMA and RSI.
- Position management with one active position per commodity.
- Exits using fixed ATR stop, Bollinger Band exit, and regime mismatch.
- ATR-based portfolio sizing with max weight and leverage caps.

Entry point:

```text
tasks/task_1/main.ipynb
```

Core wrapper:

```text
tasks/task_1/strategy.py
```

### Task 2: Enhanced Strategy

Task 2 keeps the same trend-following identity but adds:

- multi-lookback regime conviction,
- macro conviction sizing using FX, equity, credit, and volatility features,
- stricter short RSI filtering,
- execution-time RSI short cap,
- close-price Parabolic SAR-style trailing exits,
- conviction-weighted sizing,
- transaction costs.

Entry point:

```text
tasks/task_2/main.ipynb
```

Core wrapper:

```text
tasks/task_2/strategy_enhanced.py
```

Detailed strategy documentation:

```text
docs/Enhanced_Stategy.md
```

## Out-of-Sample Testing

The OOS evaluation compares frozen Task 1 and Task 2 settings from:

```text
2020-01-01 to 2026-05-18
```

The OOS workflow starts flat, purges pre-OOS entries, uses training data only for rolling indicator warm-up, and does not tune parameters on the test period.

Notebook:

```text
tasks/out_of_sample/OOS_testing.ipynb
```

Results writeup:

```text
tasks/out_of_sample/OOS_RESULTS.md
```

## Task 3 Outputs

Task 3 uses a single deterministic output folder:

```text
tasks/task_3/outputs/
```

Subfolders:

```text
tasks/task_3/outputs/mlp/
tasks/task_3/outputs/ridge/
tasks/task_3/outputs/portfolio/
tasks/task_3/outputs/market_factor/
```

Important files:

```text
tasks/task_3/outputs/mlp/nn_test_signals_wide.csv
tasks/task_3/outputs/ridge/ridge_test_signals_wide.csv
tasks/task_3/outputs/portfolio/portfolio_performance_stats_test.csv
```

The legacy FX prototype in `Task3_Ridge.ipynb` is disabled by default. The usable Ridge strategy is the commodity Ridge block that writes to `tasks/task_3/outputs/ridge/`.

## Suggested Run Order

From a fresh clone, run notebooks in this order:

1. `tasks/task_0/task_0.ipynb`
2. `tasks/task_1/main.ipynb`
3. `tasks/task_2/main.ipynb`
4. `tasks/out_of_sample/OOS_testing.ipynb`
5. Task 3:
   - `tasks/task_3/Task3_Ridge.ipynb`
   - `tasks/task_3/TaskThreeML.ipynb`

For Task 3, run the Ridge and MLP signal-generation blocks before the portfolio block.

## Environment

The project was developed in Python/Jupyter. The main dependencies are:

```text
numpy
pandas
matplotlib
scikit-learn
openpyxl
jupyter
```

If using Anaconda, most dependencies are already available. `openpyxl` is required for reading the Excel workbooks.

## Notes

- Notebook outputs are intentionally cleared or regenerated to avoid stale charts/tables.
- Python cache folders should not be committed.
- Task 2 research diagnostics are training-only; the final OOS notebook is the holdout evaluation.
- The passive equal-weight commodity factor is a benchmark, not a strategy with the same entry/exit and risk constraints.
