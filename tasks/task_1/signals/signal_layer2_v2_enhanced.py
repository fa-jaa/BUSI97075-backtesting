"""
Layer 2 — Entry Timing (Enhanced)
===================================

Identical to signal_layer2_v2 with one additional filter on the short leg:

Short RSI cap
-------------
  Short entries are blocked when RSI < rsi_short_cap (default 40).
  Rationale: if RSI is already below 40 the asset is oversold — shorting into
  an already-depressed reading increases the risk of a sharp mean-reversion.
  This is a pure level condition (not a crossover), applied on top of the
  existing RSI flag filter.

  short_ok = layer1==-1  AND  EMA cross down  AND  RSI flag active
             AND  RSI >= rsi_short_cap

Full short condition (all must hold at signal bar t):
  · Layer 1 regime == -1
  · Price crosses below EMA
  · RSI flag: RSI crossed below rsi_level within the last rsi_flag_window bars
  · RSI >= rsi_short_cap  ← enhancement

Long condition unchanged from signal_layer2_v2.

Look-ahead convention
---------------------
Signal generated at close of t. Execution at close of t+1 (backtester shifts).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from indicators import ema, rsi


def layer2_signal(
    prices:           pd.DataFrame,
    layer1:           pd.DataFrame,
    ema_window:       int   = 14,
    rsi_window:       int   = 14,
    rsi_long_level:   float = 50.0,   # RSI must cross ABOVE this to confirm long
    rsi_level:        float = 60.0,   # RSI must cross BELOW this to confirm short
    rsi_flag_window:  int   = 5,
    rsi_short_cap:    float = 40.0,   # short blocked if RSI < this (oversold cap)
) -> pd.DataFrame:
    """
    Layer 2 entry timing signal — enhanced short filter.

    Parameters
    ----------
    prices          : DataFrame of close prices (dates × assets)
    layer1          : output of layer1_signal — values in {-1, 0, +1, NaN}
    ema_window      : EMA period for price crossover
    rsi_window      : RSI lookback period
    rsi_long_level  : RSI must cross above this to confirm long entry
    rsi_level       : RSI must cross below this to confirm short entry
    rsi_flag_window : how many days the RSI flag stays active after the break
    rsi_short_cap   : short entries blocked when RSI < this level (default 40.0)

    Returns
    -------
    pd.DataFrame
        Values in {-1, 0, +1, NaN}.
        Signal at t → execution at close of t+1 (backtester responsibility).
    """
    e      = ema(prices, ema_window)
    e_prev = e.shift(1)
    p_prev = prices.shift(1)

    cross_up   = (prices > e)  & (p_prev <= e_prev)
    cross_down = (prices < e)  & (p_prev >= e_prev)

    rsi_vals = rsi(prices, rsi_window)
    rsi_prev = rsi_vals.shift(1)

    # Long RSI flag: RSI crosses ABOVE rsi_long_level → stays active rsi_flag_window days
    rsi_break_long = ((rsi_vals > rsi_long_level) & (rsi_prev <= rsi_long_level)).astype(float)
    rsi_flag_long  = rsi_break_long.rolling(rsi_flag_window, min_periods=1).max()

    # Short RSI flag: RSI crosses BELOW rsi_level → stays active rsi_flag_window days
    rsi_break_short = ((rsi_vals < rsi_level) & (rsi_prev >= rsi_level)).astype(float)
    rsi_flag_short  = rsi_break_short.rolling(rsi_flag_window, min_periods=1).max()

    long_ok  = (layer1 == 1)  & cross_up   & (rsi_flag_long  == 1)

    # Enhancement: additionally require RSI >= rsi_short_cap to avoid shorting oversold assets
    short_ok = (layer1 == -1) & cross_down & (rsi_flag_short == 1) & (rsi_vals >= rsi_short_cap)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[long_ok]  =  1.0
    signal[short_ok] = -1.0

    insufficient = e.isna() | e_prev.isna() | rsi_vals.isna() | layer1.isna()
    signal[insufficient] = float("nan")

    return signal
