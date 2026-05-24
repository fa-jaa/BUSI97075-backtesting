"""
Layer 2 — Entry Timing

Requires Layer 1 signal as input (regime filter already applied).

Look-ahead convention
---------------------
Signal is generated at close of day t (conditions observed at t).
Execution happens at close of day t+1 — the backtester shifts by 1,
NO shift is applied here.

Long  (+1): layer1 == +1  AND  price crosses above EMA20 at t

Short (-1): layer1 == -1  AND  price crosses below EMA20 at t
                           AND  RSI flag is active at t

RSI flag (short only)
---------------------
The RSI flag fires when RSI crosses below `rsi_level` (e.g. 50).
It then stays active for `rsi_flag_window` days via a rolling max.
This allows the RSI break to occur 1–N days BEFORE the EMA cross
and still trigger the entry when the EMA cross arrives.

    rsi_break[t] = 1  if RSI[t-1] >= rsi_level AND RSI[t] < rsi_level
    rsi_flag[t]  = 1  if any rsi_break in [t-rsi_flag_window+1 … t] == 1

EMA cross (long):  close[t-1] <= EMA20[t-1]  AND  close[t] > EMA20[t]
EMA cross (short): close[t-1] >= EMA20[t-1]  AND  close[t] < EMA20[t]

Neutral(0): conditions not met
NaN       : insufficient history for any indicator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from indicators import ema, rsi


def layer2_signal(
    prices:          pd.DataFrame,
    layer1:          pd.DataFrame,
    ema_window:      int   = 20,
    rsi_window:      int   = 14,
    rsi_level:       float = 50.0,
    rsi_flag_window: int   = 5,
) -> pd.DataFrame:
    """
    Layer 2 entry timing signal.

    Parameters
    ----------
    prices          : DataFrame of close prices (dates × assets)
    layer1          : output of layer1_signal — values in {-1, 0, +1, NaN}
    ema_window      : EMA period for price crossover
    rsi_window      : RSI lookback period
    rsi_level       : RSI level to break (default 50)
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

    rsi_vals  = rsi(prices, rsi_window)
    rsi_prev  = rsi_vals.shift(1)

    # fires 1 only on the day RSI crosses below rsi_level
    rsi_break = ((rsi_vals < rsi_level) & (rsi_prev >= rsi_level)).astype(float)
    # stays 1 for rsi_flag_window days after the break
    rsi_flag  = rsi_break.rolling(rsi_flag_window, min_periods=1).max()

    long_ok  = (layer1 == 1)  & cross_up
    short_ok = (layer1 == -1) & cross_down & (rsi_flag == 1)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[long_ok]  =  1.0
    signal[short_ok] = -1.0

    insufficient = e.isna() | e_prev.isna() | rsi_vals.isna() | layer1.isna()
    signal[insufficient] = float("nan")

    return signal
