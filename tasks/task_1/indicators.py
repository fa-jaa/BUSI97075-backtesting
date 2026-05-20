"""
Pure indicator functions — no state, no side effects.

Each function takes a prices DataFrame (dates × assets) and returns a
DataFrame of the same shape. All calculations use only data up to and
including the current row (no look-ahead), except expanding_quantile
which explicitly shifts by 1 day.
"""

import numpy as np
import pandas as pd


def sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average over `window` days."""
    return prices.rolling(window, min_periods=window).mean()


def ema(prices: pd.DataFrame, span: int) -> pd.DataFrame:
    """Exponential moving average with span-day half-life."""
    return prices.ewm(span=span, min_periods=span, adjust=False).mean()


def atr_close_proxy(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Close-to-close ATR proxy (rolling mean of absolute daily price change).

    Uses close-to-close differences instead of the full true range (which
    requires high/low data). Returns raw price units, not a percentage.
    Divide by prices to convert to a percentage if needed.
    """
    tr  = (prices - prices.shift(1)).abs()
    atr = tr.rolling(window, min_periods=window).mean()
    return atr


def bollinger_bands(
    prices: pd.DataFrame,
    window: int,
    num_std: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bollinger Bands: rolling mean ± num_std × rolling std.

    Returns
    -------
    (mid, upper, lower) — three DataFrames of the same shape as prices.
    """
    mid   = prices.rolling(window, min_periods=window).mean()
    std   = prices.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def sma_slope(prices: pd.DataFrame, window: int, lookback: int = 21) -> pd.DataFrame:
    """
    Slope of SMA(window) measured over the past `lookback` days.

    slope_t = SMA_t − SMA_{t-lookback}

    Positive → rising trend; negative → falling.
    """
    ma = sma(prices, window)
    return ma - ma.shift(lookback)


def expanding_quantile(series: pd.DataFrame, q: float, min_periods: int = 252) -> pd.DataFrame:
    """
    Expanding-window quantile, shifted by 1 day to avoid look-ahead bias.

    The threshold at time t uses only data available up to t-1, so it can
    safely be compared against today's value. Requires min_periods of history
    before producing a non-NaN result.
    """
    return series.expanding(min_periods=min_periods).quantile(q).shift(1)
