"""
Strategy v2 — Trend-Following Pipeline
=======================================

Pipeline
--------
prices
  │
  ▼
layer1_signal        Regime filter (persistent)
  │                  Long  : SMA50 > SMA200  AND  slope SMA200 > 0 (10d)
  │                  Short : SMA50 < SMA200  AND  slope SMA200 < 0 (10d)
  ▼
layer2_signal        Entry timing (momentary)
  │                  Long  : layer1 == +1  AND  price crosses above EMA20
  │                  Short : layer1 == -1  AND  price crosses below EMA20
  │                           AND  RSI flag breaks below 50
  ▼
layer3_signal        Exit conditions (event-driven)
  │                  Long  : close > Upper BB(200, 2σ)  OR  close < SAR
  │                  Short : close > SAR (inverted)
  ▼
build_positions      Stateful position series {-1, 0, +1} per asset
  ▼
build_weights        Vol-parity sizing locked at entry — event-driven rebalancing
  ▼
compute_portfolio_returns   Daily portfolio returns
"""

import pandas as pd

from signals.signal_layer1_v2 import layer1_signal
from signals.signal_layer2_v2 import layer2_signal
from signals.signal_layer3_v2 import layer3_signal
from portfolio.positions       import build_positions
from portfolio.weights         import build_weights, compute_portfolio_returns


def build_signals(
    prices: pd.DataFrame,
    # Layer 1
    filter_fast:    int   = 50,
    filter_slow:    int   = 200,
    slope_lookback: int   = 10,
    # Layer 2
    ema_window:      int   = 20,
    rsi_window:      int   = 14,
    rsi_level:       float = 50.0,
    rsi_flag_window: int   = 5,
    # Layer 3
    bb_window:        int   = 100,
    bb_num_std:       float = 2.0,
    atr_window:       int   = 20,
    sar_initial_mult: float = 2.5,
    sar_af_start:     float = 0.02,
    sar_af_step:      float = 0.02,
    sar_af_max:       float = 0.20,
) -> dict[str, pd.DataFrame]:
    """
    Run the full signal pipeline and return all intermediate DataFrames.

    Returns
    -------
    dict with keys:
        'layer1'    : regime filter         {-1, 0, +1, NaN}
        'layer2'    : entry timing signal   {-1, 0, +1, NaN}
        'layer3'    : exit signal           {-1, 0, +1}
        'positions' : stateful positions    {-1, 0, +1}
    """
    l1 = layer1_signal(
        prices,
        filter_fast    = filter_fast,
        filter_slow    = filter_slow,
        slope_lookback = slope_lookback,
    )

    l2 = layer2_signal(
        prices,
        layer1          = l1,
        ema_window      = ema_window,
        rsi_window      = rsi_window,
        rsi_level       = rsi_level,
        rsi_flag_window = rsi_flag_window,
    )

    # layer3 receives l2 unshifted — it handles the shift internally
    # for correct SAR initialisation at the execution price
    l3 = layer3_signal(
        prices,
        entry_signal     = l2,
        bb_window        = bb_window,
        bb_num_std       = bb_num_std,
        atr_window       = atr_window,
        sar_initial_mult = sar_initial_mult,
        sar_af_start     = sar_af_start,
        sar_af_step      = sar_af_step,
        sar_af_max       = sar_af_max,
    )

    positions = build_positions(entry_signal=l2, exit_signal=l3)

    return {
        'layer1'    : l1,
        'layer2'    : l2,
        'layer3'    : l3,
        'positions' : positions,
    }


def run_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame = None,
    vol_window: int = 30,
    exec_lag:   int = 1,
    **signal_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Full pipeline: signals → positions → weights → portfolio returns.

    Parameters
    ----------
    prices        : close prices (dates × assets)
    returns       : daily returns; computed from prices if not provided
    vol_window    : EWMA vol window for position sizing
    exec_lag      : days between signal and execution (default 1)
    **signal_kwargs : passed through to build_signals (layer params)

    Returns
    -------
    (positions, weights, portfolio_returns)
        positions         : DataFrame {-1, 0, +1} — one column per asset
        weights           : DataFrame — vol-parity weights locked at entry
        portfolio_returns : Series    — daily portfolio returns
    """
    if returns is None:
        returns = prices.pct_change()

    signals   = build_signals(prices, **signal_kwargs)
    positions = signals['positions']

    weights = build_weights(
        positions,
        returns,
        vol_window = vol_window,
        exec_lag   = exec_lag,
    )

    port_returns = compute_portfolio_returns(weights, returns)

    return positions, weights, port_returns
