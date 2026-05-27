"""
Task 2 Training Diagnostics
===========================

This module contains research diagnostics for the enhanced strategy. It does not
define trading rules. The helpers rerun the Task 2 strategy on supplied training
prices/returns to inspect transaction-cost sensitivity, portfolio sizing
sensitivity, and Parabolic SAR parameter robustness.

Main inputs are training close prices, training asset returns, optional strategy
kwargs, and the predefined SAR grid. Main outputs are pandas DataFrames with
summary statistics.

No-lookahead and test policy: diagnostics operate only on the data passed in by
the caller. They should be used with training data during tuning; selected
parameters must be frozen before any final holdout evaluation.
"""

from itertools import product

import numpy as np
import pandas as pd

from strategy_enhanced import run_strategy


DEFAULT_SAR_GRID = {
    'sar_initial_mult': [2.0, 2.5, 3.0, 3.5, 4.0],
    'sar_af_start': [0.01, 0.02, 0.03],
    'sar_af_step': [0.01, 0.02, 0.03],
    'sar_af_max': [0.10, 0.15, 0.20, 0.25],
    'sar_grace_period': [5, 10, 15],
}

DEFAULT_SL_MULT_GRID = {
    'sl_mult': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
}


def _annualised_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    vol = r.std() * np.sqrt(252)
    return (r.mean() * 252) / vol if vol > 0 else np.nan


def _max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    cum = (1.0 + r).cumprod()
    return ((cum / cum.cummax()) - 1.0).min()


def _entry_count(positions: pd.DataFrame) -> int:
    return int(((positions != 0) & (positions.shift(1).fillna(0) == 0)).sum().sum())


def _short_entry_count(positions: pd.DataFrame) -> int:
    return int(((positions == -1) & (positions.shift(1).fillna(0) == 0)).sum().sum())


def _annual_turnover(weights: pd.DataFrame) -> float:
    daily_to = weights.fillna(0.0).diff().abs().sum(axis=1).fillna(0.0) / 2.0
    return daily_to.mean() * 252


def _side_contribution(weights: pd.DataFrame, returns: pd.DataFrame) -> tuple[float, float]:
    contrib = weights * returns
    long_ann = contrib.where(weights > 0, 0.0).sum(axis=1).mean() * 252
    short_ann = contrib.where(weights < 0, 0.0).sum(axis=1).mean() * 252
    return long_ann, short_ann


def summarize_run(
    positions: pd.DataFrame,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    asset_returns: pd.DataFrame | None = None,
) -> dict:
    """Summarise one enhanced-strategy run for training diagnostics."""
    gross = returns.attrs.get('gross_returns')
    costs = returns.attrs.get('tx_costs')

    summary = {
        'ann_return_net': returns.mean() * 252,
        'ann_vol_net': returns.std() * np.sqrt(252),
        'sharpe_net': _annualised_sharpe(returns),
        'max_drawdown': _max_drawdown(returns),
        'entries': _entry_count(positions),
        'short_entries': _short_entry_count(positions),
        'ann_turnover': _annual_turnover(weights),
        'tx_cost_bps': returns.attrs.get('tx_cost_bps', np.nan),
    }

    if gross is not None:
        summary['sharpe_gross'] = _annualised_sharpe(gross)
        summary['ann_return_gross'] = gross.mean() * 252

    if costs is not None:
        summary['ann_cost'] = costs.mean() * 252

    if asset_returns is not None:
        long_ann, short_ann = _side_contribution(
            weights,
            asset_returns.reindex(weights.index),
        )
        summary['long_contribution_ann'] = long_ann
        summary['short_contribution_ann'] = short_ann

    return summary


def transaction_cost_sensitivity(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
    strategy_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Run the enhanced strategy across training transaction-cost assumptions."""
    strategy_kwargs = dict(strategy_kwargs or {})
    rows = []

    for bps in cost_bps:
        positions, weights, port_returns = run_strategy(
            prices,
            returns=returns,
            tx_cost_bps=bps,
            **strategy_kwargs,
        )
        row = summarize_run(positions, weights, port_returns, asset_returns=returns)
        row['cost_bps'] = bps
        rows.append(row)

    return pd.DataFrame(rows).set_index('cost_bps').sort_index()


def sl_mult_grid_search(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategy_kwargs: dict | None = None,
    sl_mult_grid: dict | None = None,
    tx_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Grid search the portfolio sizing stop-distance assumption on training data.

    This is a training-only sizing diagnostic, not trading logic. It keeps the
    signal stack and SAR exit parameters fixed, then reruns the enhanced
    strategy across `sl_mult` values. Candidates are sorted by a basic
    robustness filter first, then net Sharpe.
    """
    strategy_kwargs = dict(strategy_kwargs or {})
    sl_mult_grid = dict(sl_mult_grid or DEFAULT_SL_MULT_GRID)

    baseline_positions, baseline_weights, baseline_returns = run_strategy(
        prices,
        returns=returns,
        tx_cost_bps=tx_cost_bps,
        **strategy_kwargs,
    )
    baseline = summarize_run(
        baseline_positions,
        baseline_weights,
        baseline_returns,
        asset_returns=returns,
    )
    min_entries = baseline['entries'] * 0.70
    min_drawdown = baseline['max_drawdown'] - 0.05

    keys = list(sl_mult_grid)
    rows = []

    for values in product(*(sl_mult_grid[key] for key in keys)):
        sizing_params = dict(zip(keys, values))
        kwargs = {**strategy_kwargs, **sizing_params}
        positions, weights, port_returns = run_strategy(
            prices,
            returns=returns,
            tx_cost_bps=tx_cost_bps,
            **kwargs,
        )
        row = summarize_run(positions, weights, port_returns, asset_returns=returns)
        row.update(sizing_params)
        row['passes_basic_filter'] = (
            row['entries'] >= min_entries
            and row['max_drawdown'] >= min_drawdown
        )
        row['baseline_sharpe_net'] = baseline['sharpe_net']
        row['baseline_max_drawdown'] = baseline['max_drawdown']
        row['baseline_entries'] = baseline['entries']
        row['baseline_ann_turnover'] = baseline['ann_turnover']
        row['baseline_ann_cost'] = baseline.get('ann_cost', np.nan)
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values(
        ['passes_basic_filter', 'sharpe_net', 'max_drawdown'],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def sar_grid_search(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategy_kwargs: dict | None = None,
    sar_grid: dict | None = None,
    tx_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Grid search SAR parameters on training net returns.

    This is a training-only diagnostic, not trading logic. Candidates are
    sorted by a basic robustness filter first, then net Sharpe.
    The filter rejects candidates that cut trade count below 70% of the current
    SAR baseline or worsen max drawdown by more than 5 percentage points.
    """
    strategy_kwargs = dict(strategy_kwargs or {})
    sar_grid = dict(sar_grid or DEFAULT_SAR_GRID)

    baseline_positions, baseline_weights, baseline_returns = run_strategy(
        prices,
        returns=returns,
        tx_cost_bps=tx_cost_bps,
        **strategy_kwargs,
    )
    baseline = summarize_run(
        baseline_positions,
        baseline_weights,
        baseline_returns,
        asset_returns=returns,
    )
    min_entries = baseline['entries'] * 0.70
    min_drawdown = baseline['max_drawdown'] - 0.05

    keys = list(sar_grid)
    rows = []

    for values in product(*(sar_grid[key] for key in keys)):
        sar_params = dict(zip(keys, values))
        kwargs = {**strategy_kwargs, **sar_params}
        positions, weights, port_returns = run_strategy(
            prices,
            returns=returns,
            tx_cost_bps=tx_cost_bps,
            **kwargs,
        )
        row = summarize_run(positions, weights, port_returns, asset_returns=returns)
        row.update(sar_params)
        row['passes_basic_filter'] = (
            row['entries'] >= min_entries
            and row['max_drawdown'] >= min_drawdown
        )
        row['baseline_sharpe_net'] = baseline['sharpe_net']
        row['baseline_max_drawdown'] = baseline['max_drawdown']
        row['baseline_entries'] = baseline['entries']
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values(
        ['passes_basic_filter', 'sharpe_net', 'max_drawdown'],
        ascending=[False, False, False],
    ).reset_index(drop=True)
