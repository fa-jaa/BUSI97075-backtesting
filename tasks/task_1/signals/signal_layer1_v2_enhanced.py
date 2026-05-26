"""
Layer 1 — Trend Screen (Enhanced)
===================================

Identical to signal_layer1_v2. Reserved for future enhancements.

Current logic (unchanged):
  Long  (+1): SMA50 > SMA200  AND  slope of SMA200 > 0 (over lookback days)
  Short (-1): SMA50 < SMA200  AND  slope of SMA200 < 0 (over lookback days)
  Neutral(0): conditions disagree
  NaN       : insufficient history for any indicator

The signal is persistent: stays +1/-1 every day the conditions hold,
not just on the day of a crossover.
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
    Layer 1 persistent regime signal.

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
