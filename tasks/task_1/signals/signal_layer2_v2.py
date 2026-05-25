"""
Layer 2 — Entry Timing

Requires Layer 1 signal as input (regime filter already applied).

Look-ahead convention
---------------------
Signal is generated at close of day t (conditions observed at t).
Execution happens at close of day t+1 — the backtester shifts by 1,
NO shift is applied here.

Long  (+1): layer1 == +1  AND  price crosses above EMA at t
                           AND  RSI flag crosses above rsi_long_level (symmetric filter)

Short (-1): layer1 == -1  AND  price crosses below EMA at t
                           AND  RSI flag crosses below rsi_level

RSI flag — symmetric for both directions
-----------------------------------------
Long  flag: RSI crosses ABOVE rsi_long_level → stays active rsi_flag_window days
Short flag: RSI crosses BELOW rsi_level      → stays active rsi_flag_window days

This filters out EMA crosses that happen without RSI momentum confirmation,
reducing false signals on both legs.

    rsi_break_long[t]  = 1  if RSI[t-1] <= rsi_long_level AND RSI[t] > rsi_long_level
    rsi_flag_long[t]   = 1  if any rsi_break_long in [t-rsi_flag_window+1 … t] == 1

    rsi_break_short[t] = 1  if RSI[t-1] >= rsi_level AND RSI[t] < rsi_level
    rsi_flag_short[t]  = 1  if any rsi_break_short in [t-rsi_flag_window+1 … t] == 1

EMA cross (long):  close[t-1] <= EMA[t-1]  AND  close[t] > EMA[t]
EMA cross (short): close[t-1] >= EMA[t-1]  AND  close[t] < EMA[t]

Neutral(0): conditions not met
NaN       : insufficient history for any indicator
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
    Layer 2 entry timing signal.

    Parameters
    ----------
    prices          : DataFrame of close prices (dates × assets)
    layer1          : output of layer1_signal — values in {-1, 0, +1, NaN}
    ema_window      : EMA period for price crossover
    rsi_window      : RSI lookback period
    rsi_long_level  : RSI must cross above this to confirm long entry (filters weak bounces)
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
    short_ok = (layer1 == -1) & cross_down & (rsi_flag_short == 1)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[long_ok]  =  1.0
    signal[short_ok] = -1.0

    insufficient = e.isna() | e_prev.isna() | rsi_vals.isna() | layer1.isna()
    signal[insufficient] = float("nan")

    return signal
