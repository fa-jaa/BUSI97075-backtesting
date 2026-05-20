"""
Exit Conditions

Three independent exit triggers, applied separately to long and short positions.

Exit conditions:
1. Trend exit      : price crosses the SMA50 in the wrong direction
2. Bollinger exit  : price becomes overextended outside the 100-day band
3. Volatility exit : ATR30 is both above its historical 80th percentile AND
                     higher than it was 21 days ago (high and rising)

The volatility exit is symmetric — it applies to both long and short positions
because a sharp volatility spike increases risk regardless of direction.
"""

import pandas as pd

from indicators import sma, atr_close_proxy, bollinger_bands, expanding_quantile


def _vol_exit(
    prices: pd.DataFrame,
    atr_window: int = 30,
    lookback: int = 21,
    fallback: float = 0.10,
) -> pd.DataFrame:
    """
    Volatility exit: True when ATR30 is historically elevated AND rising.

    Exits on the combination of two conditions:
      (a) ATR30 > expanding P80 of ATR30  — unusually high relative to history
      (b) ATR30 > ATR30_{t-21}            — still increasing over the past month

    Requiring both conditions avoids exiting on a single vol spike that
    immediately reverts. The expanding P80 threshold uses a 1-day shift
    (inside expanding_quantile) to avoid look-ahead bias; fallback is used
    before 252 days of history are available.
    """
    atr30 = atr_close_proxy(prices, atr_window)

    atr30_p80 = expanding_quantile(
        atr30,
        q=0.80,
        min_periods=252,
    ).fillna(fallback)

    above_p80 = atr30 > atr30_p80
    rising    = atr30 > atr30.shift(lookback)

    return (above_p80 & rising).fillna(False)


def long_exit_signal(
    prices: pd.DataFrame,
    sma_window: int = 50,
    bb_window: int = 100,
    bb_std: float = 1.5,
) -> pd.DataFrame:
    """
    Exit conditions for a long position. Exit if any of:

    1. Close < SMA50         — trend has reversed below the medium-term average
    2. Close > BB upper      — price has extended too far above the 100-day band
    3. Vol exit fires        — volatility is high and still rising

    Returns
    -------
    pd.DataFrame
        Boolean. True = exit the long position on this bar.
    """
    sma50 = sma(prices, sma_window)
    _, bb_upper, _ = bollinger_bands(prices, bb_window, bb_std)
    vol_exit = _vol_exit(prices)

    below_sma50    = prices < sma50
    above_bb_upper = prices > bb_upper

    return (below_sma50 | above_bb_upper | vol_exit).fillna(False)


def short_exit_signal(
    prices: pd.DataFrame,
    sma_window: int = 50,
    bb_window: int = 100,
    bb_std: float = 1.5,
) -> pd.DataFrame:
    """
    Exit conditions for a short position. Exit if any of:

    1. Close > SMA50         — trend has reversed above the medium-term average
    2. Close < BB lower      — price has extended too far below the 100-day band
    3. Vol exit fires        — volatility is high and still rising

    Returns
    -------
    pd.DataFrame
        Boolean. True = exit the short position on this bar.
    """
    sma50 = sma(prices, sma_window)
    _, _, bb_lower = bollinger_bands(prices, bb_window, bb_std)
    vol_exit = _vol_exit(prices)

    above_sma50    = prices > sma50
    below_bb_lower = prices < bb_lower

    return (above_sma50 | below_bb_lower | vol_exit).fillna(False)
