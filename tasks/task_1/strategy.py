"""
Task 1 Strategy — Baseline Trend-Following Pipeline
===================================================

This module orchestrates the Task 1 baseline strategy. It accepts close-price
data, builds the two signal layers, converts signals into one open position per
commodity, and then sizes those positions into portfolio weights and returns.

Strategy overview
-----------------
Layer 1 is the persistent trend filter:
  Long  : SMA50 > SMA200 and SMA200 slope over 100 days is positive.
  Short : SMA50 < SMA200 and SMA200 slope over 100 days is negative.

Layer 2 is the entry-timing layer:
  Long  : Layer 1 is long, price crosses above EMA14, and RSI confirms strength.
  Short : Layer 1 is short, price crosses below EMA14, and RSI breaks below the
          short threshold inside the active flag window.

The slot manager opens at most one position per commodity. It exits on the
first of fixed ATR stop, Bollinger Band profit exit, or Layer 1 regime end. The
portfolio manager sizes each open position from ATR stop distance, caps each
weight, and enforces the gross leverage cap.

Main inputs
-----------
prices : daily close prices, indexed by date and with one column per commodity.
returns : optional daily returns aligned to prices; computed from prices if None.

Main outputs
------------
build_signals() returns Layer 1 regime and Layer 2 entry grids.
run_strategy() returns positions, weights, and daily portfolio returns.

No-lookahead convention
-----------------------
Layer 2 records signals at the close of t. The slot manager shifts entries by
one bar, and the portfolio manager shifts weights by exec_lag before P&L.

Pipeline
--------
prices -> Layer 1 regime -> Layer 2 entries -> positions -> weights -> returns
"""

import pandas as pd

from tasks.task_1.signals.signal_layer1      import layer1_signal
from tasks.task_1.signals.signal_layer2      import layer2_signal
from portfolio.slot_manager         import build_positions
from portfolio.portfolio_manager    import build_portfolio


def build_signals(
    prices: pd.DataFrame,
    # Layer 1
    filter_fast:    int   = 50,
    filter_slow:    int   = 200,
    slope_lookback: int   = 100,
    # Layer 2
    ema_window:      int   = 14,
    rsi_window:      int   = 14,
    rsi_long_level:  float = 50.0,
    rsi_level:       float = 60.0,
    rsi_flag_window: int   = 5,
) -> dict[str, pd.DataFrame]:
    """
    Run the Task 1 signal stack.

    Layer 1 produces the persistent trend regime. Layer 2 produces momentary
    entry events inside that regime. Execution shifting happens downstream.

    Returns
    -------
    dict with keys 'layer1' and 'layer2'.
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
        rsi_long_level  = rsi_long_level,
        rsi_level       = rsi_level,
        rsi_flag_window = rsi_flag_window,
    )

    return {'layer1': l1, 'layer2': l2}


def run_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame = None,
    # Layer 1
    filter_fast:    int   = 50,
    filter_slow:    int   = 200,
    slope_lookback: int   = 100,
    # Layer 2
    ema_window:      int   = 14,
    rsi_window:      int   = 14,
    rsi_long_level:  float = 50.0,
    rsi_level:       float = 60.0,
    rsi_flag_window: int   = 5,
    # Slot manager — exit conditions
    bb_window:  int   = 100,
    bb_num_std: float = 2.0,
    atr_window: int   = 50,
    sl_mult:    float = 3.0,
    # Portfolio manager — sizing and risk
    risk_per_trade:  float = 0.01,
    max_weight:      float = 0.20,
    leverage_cap:    float = 2.50,
    exec_lag:        int   = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Run the full baseline pipeline: signals -> positions -> weights -> returns.

    Returns
    -------
    (positions, weights, portfolio_returns)
    """
    if returns is None:
        returns = prices.pct_change()

    signals = build_signals(
        prices,
        filter_fast     = filter_fast,
        filter_slow     = filter_slow,
        slope_lookback  = slope_lookback,
        ema_window      = ema_window,
        rsi_window      = rsi_window,
        rsi_long_level  = rsi_long_level,
        rsi_level       = rsi_level,
        rsi_flag_window = rsi_flag_window,
    )

    positions = build_positions(
        prices,
        layer1     = signals['layer1'],
        layer2     = signals['layer2'],
        bb_window  = bb_window,
        bb_num_std = bb_num_std,
        atr_window = atr_window,
        sl_mult    = sl_mult,
    )

    weights, port_returns = build_portfolio(
        positions,
        prices,
        returns,
        atr_window     = atr_window,
        sl_mult        = sl_mult,
        risk_per_trade = risk_per_trade,
        max_weight     = max_weight,
        leverage_cap   = leverage_cap,
        exec_lag       = exec_lag,
    )

    return positions, weights, port_returns
