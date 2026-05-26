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
layer2_signal        Entry timing (momentary events)
  │                  Long  : layer1 == +1  AND  price crosses above EMA20
  │                  Short : layer1 == -1  AND  price crosses below EMA20
  │                           AND  RSI flag breaks below 50
  ▼
build_positions      Per-commodity position builder (slot_manager)
  │                  One position per commodity, no clustering.
  │                  Exit: SAR  OR  BB(100, 2σ)  OR  regime ends
  ▼
build_portfolio      Risk budget and weights (portfolio_manager)
  │                  ATR-based bottom-up sizing, gross leverage cap.
  ▼
portfolio_returns    Daily portfolio returns
"""

import pandas as pd

from signals.signal_layer1_v2      import layer1_signal
from signals.signal_layer2_v2      import layer2_signal
from portfolio.slot_manager         import build_positions
from portfolio.portfolio_manager    import build_portfolio


def build_signals(
    prices: pd.DataFrame,
    # Layer 1
    filter_fast:    int   = 50,
    filter_slow:    int   = 200,
    slope_lookback: int   = 10,
    # Layer 2
    ema_window:      int   = 14,
    rsi_window:      int   = 14,
    rsi_long_level:  float = 50.0,
    rsi_level:       float = 60.0,
    rsi_flag_window: int   = 5,
) -> dict[str, pd.DataFrame]:
    """
    Run Layer 1 (regime) and Layer 2 (entry timing).

    Returns
    -------
    dict with keys 'layer1' and 'layer2'
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
    slope_lookback: int   = 10,
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
    sl_mult:    float = 3.0,    # Stop Loss = entry_price × (1 ± sl_mult × ATR_pct)
    # Portfolio manager — sizing and risk
    risk_per_trade:  float = 0.01,
    max_weight:      float = 0.20,
    leverage_cap:    float = 2.50,
    exec_lag:        int   = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Full pipeline: signals → positions → weights → portfolio returns.

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
