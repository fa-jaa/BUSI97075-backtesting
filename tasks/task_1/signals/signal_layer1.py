"""
Layer 1 — Trend Screen

Two independent sub-signals, both must agree for an entry to be eligible.
This layer changes slowly (weeks to months between flips) and is the sole
entry signal for the strategy.

Sub-signals
-----------
_sma_crossover   : SMA50 position relative to SMA200 (price direction)
_slope_direction : direction of SMA200 slope (trend momentum confirmation)

Combined (layer1_signal)
------------------------
Long eligible  (+1): SMA50 > SMA200  AND  SMA200 slope rising
Short eligible (-1): SMA50 < SMA200  AND  SMA200 slope falling
Neutral         (0): sub-signals disagree (e.g. SMA50 > SMA200 but slope falling)
NaN                : insufficient history for either sub-signal
"""

import numpy as np
import pandas as pd
from indicators import sma, sma_slope


def _sma_crossover(
    prices: pd.DataFrame,
    fast_window: int = 30,
    slow_window: int = 200,
) -> pd.DataFrame:
    """
    SMA50 vs SMA200 crossover direction.

    Returns +1 when fast MA is above slow MA (bullish structure),
    -1 when below, NaN before sufficient history is available.
    """
    fast = sma(prices, fast_window)
    slow = sma(prices, slow_window)

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
    Direction of the SMA200 slope over the past `slope_lookback` days.

    Adds a momentum confirmation: even when SMA50 > SMA200, a falling SMA200
    suggests the long-term trend is losing steam. Returns +1 (rising), -1
    (falling), or NaN (insufficient history).
    """
    slope = sma_slope(prices, slow_window, slope_lookback)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[slope > 0] =  1.0
    signal[slope < 0] = -1.0
    signal[slope.isna()] = np.nan

    return signal


def layer1_signal(
    prices: pd.DataFrame,
    fast_window: int = 30,
    slow_window: int = 200,
    slope_lookback: int = 21,
) -> pd.DataFrame:
    """
    Layer 1 trend screen: both sub-signals must agree on direction.

    Requiring agreement between the crossover and the slope prevents entries
    during whipsaw periods where SMA50 crosses SMA200 but the long-term
    trend is still moving in the opposite direction.

    Returns
    -------
    pd.DataFrame
        Values in {-1, 0, +1, NaN}.
    """
    crossover = _sma_crossover(prices, fast_window, slow_window)
    slope_dir = _slope_direction(prices, slow_window, slope_lookback)

    long_ok  = (crossover == 1)  & (slope_dir == 1)
    short_ok = (crossover == -1) & (slope_dir == -1)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[long_ok]  =  1.0
    signal[short_ok] = -1.0
    signal[crossover.isna() | slope_dir.isna()] = np.nan

    return signal
