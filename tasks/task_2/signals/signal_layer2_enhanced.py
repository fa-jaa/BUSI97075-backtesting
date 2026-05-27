"""
Task 2 Layer 2 — Enhanced Entry Timing
======================================

This module creates momentary Task 2 entry events from close prices and the
binary Layer 1 regime grid. It returns a DataFrame with values in {-1, 0, +1,
NaN}. The short RSI oversold cap is not applied here.

Enhancements over the Task 1 Layer 2:

  1. Short entries use a stricter RSI momentum trigger.
       Long:  RSI crosses ABOVE 50 (momentum turning bullish)
       Short: RSI crosses BELOW 60 (bearish momentum confirmation)

  2. Short RSI cap (oversold guard) moved to execution time.
       The cap (default 50) is checked in slot_manager_enhanced at the entry bar,
       not here at signal-generation time.  This avoids the 1-bar stale-RSI problem
       where the cap is satisfied at t but violated when the trade executes at t+1.

  3. RSI flag window is 5 days.

Long/short conditions (at signal bar t):
  · Layer 1 regime == ±1
  · Price crosses above/below EMA
  · RSI crossed above/below rsi_level within the last rsi_flag_window bars

No-lookahead: the signal is generated at close t using data through t. The slot
manager shifts entries to t+1 and applies the execution-bar short RSI cap.
Pipeline role: Layer 1 regime -> Layer 2 entry events -> enhanced slot manager.
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
) -> pd.DataFrame:
    """
    Build the enhanced entry-timing signal.

    Parameters
    ----------
    prices          : DataFrame of close prices (dates × assets)
    layer1          : output of layer1_signal — values in {-1, 0, +1, NaN}
    ema_window      : EMA period for price crossover
    rsi_window      : RSI lookback period
    rsi_long_level  : RSI must cross above this to confirm long entry
    rsi_level       : RSI must cross below this to confirm short entry
    rsi_flag_window : how many days the RSI flag stays active after the break

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

    # RSI cap is checked at execution time in slot_manager_enhanced, not here
    short_ok = (layer1 == -1) & cross_down & (rsi_flag_short == 1)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[long_ok]  =  1.0
    signal[short_ok] = -1.0

    insufficient = e.isna() | e_prev.isna() | rsi_vals.isna() | layer1.isna()
    signal[insufficient] = float("nan")

    return signal
