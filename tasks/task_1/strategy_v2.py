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
build_slot_positions Concurrent slot manager (replaces Layer 3 + positions + weights)
  │                  Each EMA cross opens a NEW independent slot with its own SAR.
  │                  Multiple slots per asset run concurrently until:
  │                    · Long:  price < SAR  OR  price > Upper BB
  │                    · Short: price > SAR
  │                    · Layer 1 regime ends
  ▼
portfolio_returns    Daily portfolio returns (vol-parity, event-driven, gross = 1)
"""

import pandas as pd

from signals.signal_layer1_v2     import layer1_signal
from signals.signal_layer2_v2     import layer2_signal
from portfolio.slot_manager        import build_slot_positions


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
    Run the regime filter (Layer 1) and entry timing (Layer 2).

    Returns
    -------
    dict with keys:
        'layer1' : regime filter       {-1, 0, +1, NaN}
        'layer2' : entry timing events {-1, 0, +1, NaN}
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
    exec_lag:   int = 1,
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
    bb_window:        int   = 100,
    bb_num_std:       float = 2.0,
    atr_window:       int   = 50,
    sar_initial_mult: float = 10.0,
    sar_af_start:     float = 0.02,
    sar_af_step:      float = 0.02,
    sar_af_max:       float = 0.20,
    sar_grace_period: int   = 10,
    # Slot manager — bottom-up sizing
    risk_per_slot:       float = 0.01,   # fraction of capital risked per slot (1%)
    max_slot_size:       float = 0.20,   # hard cap per individual slot weight (20%)
    leverage_cap:        float = 2.50,   # gross leverage cap (250%)
    max_slots_per_asset: int   = 2,      # max concurrent slots per commodity
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Full pipeline: signals → concurrent slot positions → weights → portfolio returns.

    Parameters
    ----------
    prices         : close prices (dates × assets)
    returns        : daily returns; computed from prices if not provided
    exec_lag       : days between signal and execution (default 1)
    risk_per_slot  : capital fraction risked per slot — size = risk / stop_distance
    leverage_cap   : gross leverage cap; new slots blocked when reached (e.g. 2.5 = 250%)

    Returns
    -------
    (positions, weights, portfolio_returns)
        positions         : DataFrame of integer slot counts (+N long, -N short)
        weights           : bottom-up weights (not normalised), shifted by exec_lag
        portfolio_returns : Series — daily portfolio returns
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

    positions, weights, port_returns = build_slot_positions(
        prices,
        returns,
        entry_signal         = signals['layer2'],
        layer1               = signals['layer1'],
        bb_window            = bb_window,
        bb_num_std           = bb_num_std,
        atr_window           = atr_window,
        sar_initial_mult     = sar_initial_mult,
        sar_af_start         = sar_af_start,
        sar_af_step          = sar_af_step,
        sar_af_max           = sar_af_max,
        sar_grace_period     = sar_grace_period,
        risk_per_slot        = risk_per_slot,
        max_slot_size        = max_slot_size,
        leverage_cap         = leverage_cap,
        max_slots_per_asset  = max_slots_per_asset,
        exec_lag             = exec_lag,
    )

    return positions, weights, port_returns
