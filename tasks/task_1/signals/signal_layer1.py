"""
Task 1 Layer 1 — Trend Regime Screen
====================================

This module creates the baseline persistent regime grid used by the Task 1
strategy. It consumes close prices and returns a DataFrame with values in
{-1, 0, +1, NaN}.

Both the moving-average crossover and slow-SMA slope must agree for a regime to
be active. The signal is persistent: it stays +1 or -1 every day the conditions
hold, not only on the crossover day.

Long  (+1): SMA_fast > SMA_slow and SMA_slow slope > 0.
Short (-1): SMA_fast < SMA_slow and SMA_slow slope < 0.
Neutral(0): conditions disagree.
NaN       : insufficient indicator history.

No-lookahead: each row uses prices observed up to that row only.
Pipeline role: prices -> Layer 1 regime -> Layer 2 entry timing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from indicators import sma, sma_slope


def layer1_signal(
    prices: pd.DataFrame,
    filter_fast:    int = 50,
    filter_slow:    int = 200,
    slope_lookback: int = 100,
) -> pd.DataFrame:
    """
    Build the persistent baseline regime signal.

    Returns
    -------
    pd.DataFrame
        Values in {-1, 0, +1, NaN}.
    """
    fast  = sma(prices, filter_fast)
    slow  = sma(prices, filter_slow)
    slope = sma_slope(prices, filter_slow, slope_lookback)

    long_ok  = (fast > slow) & (slope > 0)
    short_ok = (fast < slow) & (slope < 0)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[long_ok]  =  1.0
    signal[short_ok] = -1.0
    signal[fast.isna() | slow.isna() | slope.isna()] = float("nan")

    return signal
