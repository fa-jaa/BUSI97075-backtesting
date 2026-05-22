"""
Layer 1 — Trend Screen

Three conditions must all agree for an entry to fire:

Regime filter (persistent — gates whether entries are eligible)
----------------------------------------------------------------
1. SMA crossover  : SMA(filter_fast) position relative to SMA(filter_slow)
2. Slope confirm  : direction of SMA(filter_slow) slope over slope_lookback days

Entry timing (momentary — fires only on the crossing day)
---------------------------------------------------------
3. EMA cross      : price crosses above/below EMA(ema_window)

Combined (layer1_signal)
------------------------
Long  (+1): SMA_fast > SMA_slow  AND  slope rising  AND  price crosses above EMA
Short (-1): SMA_fast < SMA_slow  AND  slope falling AND  price crosses below EMA
Neutral(0): any condition fails or disagrees
NaN       : insufficient history for any indicator
"""

import numpy as np
import pandas as pd
from indicators import sma, sma_slope, ema


def _sma_regime(
    prices: pd.DataFrame,
    filter_fast: int = 50,
    filter_slow: int = 200,
) -> pd.DataFrame:
    """
    Persistent regime direction from SMA crossover.

    Returns +1 when fast MA is above slow MA (bullish regime),
    -1 when below, NaN before sufficient history is available.
    """
    fast = sma(prices, filter_fast)
    slow = sma(prices, filter_slow)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[fast > slow] =  1.0
    signal[fast < slow] = -1.0
    signal[fast.isna() | slow.isna()] = np.nan

    return signal


def _slope_direction(
    prices: pd.DataFrame,
    slow_window: int = 200,
    slope_lookback: int = 21,
) -> pd.DataFrame:
    """
    Direction of the slow SMA slope over the past `slope_lookback` days.

    Regime condition 2: confirms the long-term trend still has momentum.
    Returns +1 (rising), -1 (falling), or NaN (insufficient history).
    """
    slope = sma_slope(prices, slow_window, slope_lookback)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[slope > 0] =  1.0
    signal[slope < 0] = -1.0
    signal[slope.isna()] = np.nan

    return signal


def _ema_cross(
    prices: pd.DataFrame,
    ema_window: int = 20,
) -> pd.DataFrame:
    """
    Momentary EMA crossover signal — fires only on the day of the cross.

    Long  (+1): close_today > EMA_today  AND  close_yesterday <= EMA_yesterday
    Short (-1): close_today < EMA_today  AND  close_yesterday >= EMA_yesterday
    Zero   (0): no cross occurred today
    NaN       : insufficient history
    """
    e = ema(prices, ema_window)
    e_prev = e.shift(1)
    p_prev = prices.shift(1)

    cross_up   = (prices > e) & (p_prev <= e_prev)
    cross_down = (prices < e) & (p_prev >= e_prev)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[cross_up]   =  1.0
    signal[cross_down] = -1.0
    signal[e.isna() | e_prev.isna()] = np.nan

    return signal


def layer1_signal(
    prices: pd.DataFrame,
    filter_fast:    int = 50,
    filter_slow:    int = 200,
    slope_lookback: int = 21,
    ema_window:     int = 20,
) -> pd.DataFrame:
    """
    Layer 1 entry signal: regime filter + EMA cross timing.

    The SMA crossover and slope conditions gate whether entries are eligible
    (regime). The EMA cross is the timing trigger — it fires only on the day
    price crosses the EMA, so entries are precisely timed rather than
    triggering on every bar the regime is active.

    Returns
    -------
    pd.DataFrame
        Values in {-1, 0, +1, NaN}. +1/-1 only on the crossing day when
        both regime conditions are satisfied.
    """
    regime    = _sma_regime(prices, filter_fast, filter_slow)
    slope_dir = _slope_direction(prices, filter_slow, slope_lookback)
    cross     = _ema_cross(prices, ema_window)

    long_ok  = (regime == 1)  & (slope_dir == 1)  & (cross == 1)
    short_ok = (regime == -1) & (slope_dir == -1) & (cross == -1)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[long_ok]  =  1.0
    signal[short_ok] = -1.0
    signal[regime.isna() | slope_dir.isna() | cross.isna()] = np.nan

    return signal
