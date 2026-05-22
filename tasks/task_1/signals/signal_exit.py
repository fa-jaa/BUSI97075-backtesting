"""
Exit Conditions

Three independent exit triggers, applied separately to long and short positions.

Exit conditions:
1. Trend exit      : price crosses the SMA50 in the wrong direction
2. Bollinger exit  : price becomes overextended outside the 100-day band
3. ATR stop exit   : the bar return moves against the held side by more than
                     multiplier × ATR%

The ATR stop is side-aware: long and short stop masks are computed separately,
then the stateful position builder applies only the mask for the side held.
"""

import pandas as pd

from indicators import sma, atr_pct, bollinger_bands


def atr_stop_exit_signal(
    prices: pd.DataFrame,
    side: str,
    atr_window: int = 30,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    ATR-based stop exit: True when the position's daily return breaches the
    ATR30 stop level.

    Long exit : return_t <= -(multiplier × ATR30%)
    Short exit: return_t >=  (multiplier × ATR30%)

    The stateful position builder applies the long/short stop only when that
    side is actually held, avoiding a preliminary no-exit position dependency.
    """
    vol_threshold = atr_pct(prices, window=atr_window)
    asset_ret = prices.pct_change()

    if side == "long":
        exit_signal = asset_ret <= -(multiplier * vol_threshold)
    elif side == "short":
        exit_signal = asset_ret >= multiplier * vol_threshold
    else:
        raise ValueError("side must be 'long' or 'short'")

    return exit_signal.fillna(False)


def long_exit_signal(
    prices: pd.DataFrame,
    sma_window: int = 50,
    bb_window: int = 100,
    bb_std: float = 1.5,
    atr_window: int = 30,
    atr_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Exit conditions for a long position. Exit if any of:

    1. Close < SMA50         — trend has reversed below the medium-term average
    2. Close > BB upper      — price has extended too far above the 100-day band
    3. ATR stop fires        — daily loss exceeds multiplier × ATR30%

    Returns
    -------
    pd.DataFrame
        Boolean. True = exit the long position on this bar.
    """
    sma50 = sma(prices, sma_window)
    _, bb_upper, _ = bollinger_bands(prices, bb_window, bb_std)
    atr_stop = atr_stop_exit_signal(prices, "long", atr_window, atr_multiplier)

    below_sma50    = prices < sma50
    above_bb_upper = prices > bb_upper

    return (below_sma50 | above_bb_upper | atr_stop).fillna(False)


def short_exit_signal(
    prices: pd.DataFrame,
    sma_window: int = 50,
    bb_window: int = 100,
    bb_std: float = 1.5,
    atr_window: int = 30,
    atr_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Exit conditions for a short position. Exit if any of:

    1. Close > SMA50         — trend has reversed above the medium-term average
    2. Close < BB lower      — price has extended too far below the 100-day band
    3. ATR stop fires        — daily loss exceeds multiplier × ATR30%

    Returns
    -------
    pd.DataFrame
        Boolean. True = exit the short position on this bar.
    """
    sma50 = sma(prices, sma_window)
    _, _, bb_lower = bollinger_bands(prices, bb_window, bb_std)
    atr_stop = atr_stop_exit_signal(prices, "short", atr_window, atr_multiplier)

    above_sma50    = prices > sma50
    below_bb_lower = prices < bb_lower

    return (above_sma50 | below_bb_lower | atr_stop).fillna(False)
