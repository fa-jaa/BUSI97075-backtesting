"""
Task 2 Macro Regime Signals
===========================

This module builds external macro signals used by Task 2 to modulate conviction
and position sizing. It consumes training-period FX, equity, credit, and implied
volatility data. It returns pandas Series/DataFrames aligned by date.

Macro signals do not change trade direction. Commodity Layer 1 and Layer 2 still
drive entries and exits; macro only scales regime_strength before sizing.

No-lookahead: moving averages use data through the current row. The volatility
percentile explicitly shifts by one day so the percentile threshold at t uses
history through t-1.

Signals
-------
  dollar_signal   : USD weakness proxy from FX basket
                    Weak USD → bullish commodities (+1)
  equity_signal   : SPX trend (risk-on / risk-off gate)
                    Uptrend (+1) supports cyclical commodity longs
  credit_signal   : CDX HY spread trend (risk sentiment)
                    Tightening spreads (+1) = risk-on environment

Aggregation
-----------
  macro_regime_score : mean of available signals → continuous [-1, +1]

Blending into regime_strength
------------------------------
  macro_factor = clip(1 + alpha × macro_score × sign(commodity_strength),
                      min_factor, max_factor)
  effective_strength = commodity_strength × macro_factor

  With default alpha=0.5:
    Fully aligned   → 1.5× size  (50 % boost)
    Neutral         → 1.0× size  (unchanged)
    Moderate oppose → 0.75× size
    Fully opposed   → 0.5× size

FX Vol Cap
----------
  apply_vol_cap : when EURUSD 3M ATM vol is in the top percentile of its
                  history, multiply effective_strength by vol_dampen.
                  Prevents large entries during macro crisis periods.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def _sma_crossover(series: pd.Series, fast: int, slow: int) -> pd.Series:
    """Return sign of (SMA_fast − SMA_slow): +1 uptrend, -1 downtrend, 0 flat."""
    sma_fast = series.rolling(int(fast), min_periods=int(fast)).mean()
    sma_slow = series.rolling(int(slow), min_periods=int(slow)).mean()
    cross = np.sign(sma_fast - sma_slow)
    cross[sma_slow.isna()] = np.nan
    return cross


# ── public signal functions ───────────────────────────────────────────────────

def dollar_signal(
    fx_spot: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """
    USD weakness proxy from a 4-pair FX basket.

    +1 when the basket signals a weak USD (bullish for commodities).
    -1 when USD is strong (bearish for commodities).

    Pairs used
    ----------
    EUR/USD, GBP/USD, AUD/USD  — higher price = weaker USD → vote +1
    USD/JPY                    — higher price = stronger USD → vote -1 (inverted)

    Parameters
    ----------
    fx_spot : DataFrame with columns including 'EURUSD CURNCY',
              'GBPUSD Curncy', 'AUDUSD Curncy', 'USDJPY Curncy'
    fast    : SMA fast window (default 50)
    slow    : SMA slow window (default 200)

    Returns
    -------
    pd.Series  values in {-1, 0, +1, NaN}
    """
    col_map = {
        'EURUSD CURNCY': +1,  # USD/quote → higher = weak USD
        'GBPUSD Curncy': +1,
        'AUDUSD Curncy': +1,
        'USDJPY Curncy': -1,  # USD/base  → higher = strong USD, invert
    }

    votes = []
    for col, direction in col_map.items():
        if col not in fx_spot.columns:
            continue
        price = fx_spot[col].copy()
        price = price[~price.index.duplicated()].astype(float)
        cross = _sma_crossover(price, fast, slow)
        votes.append(direction * cross)

    if not votes:
        raise ValueError("No FX spot columns found for dollar signal.")

    basket = pd.concat(votes, axis=1).mean(axis=1)
    signal = np.sign(basket)
    signal[basket.isna()] = np.nan
    signal.name = 'dollar_signal'
    return signal


def equity_signal(
    equity_data: pd.DataFrame,
    ticker: str = 'SPX INDEX',
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """
    S&P 500 trend signal as a risk-on/risk-off gate.

    +1 uptrend (risk-on)  → supports commodity longs
    -1 downtrend (risk-off) → headwind for commodity longs

    Parameters
    ----------
    equity_data : DataFrame with a column named `ticker`
    ticker      : column name (default 'SPX INDEX')
    fast        : SMA fast window (default 50)
    slow        : SMA slow window (default 200)

    Returns
    -------
    pd.Series  values in {-1, 0, +1, NaN}
    """
    if ticker not in equity_data.columns:
        raise ValueError(f"Ticker '{ticker}' not found in equity_data.")

    price = equity_data[ticker].copy()
    price = price[~price.index.duplicated()].astype(float)

    signal = _sma_crossover(price, fast, slow)
    signal.name = 'equity_signal'
    return signal


def credit_signal(
    credit_data: pd.DataFrame,
    ticker: str = 'CDX HY CDSI GEN 5Y CORP',
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """
    CDX High Yield spread trend (inverted).

    CDX HY widening = risk-off → -1 (bearish for commodities)
    CDX HY tightening = risk-on → +1

    NaN-safe: returns NaN for any date where the spread is not available
    (CDX HY data starts ~2007 in the training set).

    Parameters
    ----------
    credit_data : DataFrame with a column named `ticker`
    ticker      : column name (default 'CDX HY CDSI GEN 5Y CORP')
    fast        : SMA fast window (default 50)
    slow        : SMA slow window (default 200)

    Returns
    -------
    pd.Series  values in {-1, 0, +1, NaN}
    """
    if ticker not in credit_data.columns:
        raise ValueError(f"Ticker '{ticker}' not found in credit_data.")

    spread = credit_data[ticker].copy()
    spread = spread[~spread.index.duplicated()].astype(float)

    # Invert: rising spreads = bearish → multiply by -1
    cross = _sma_crossover(spread, fast, slow)
    signal = -cross
    signal.name = 'credit_signal'
    return signal


def vol_percentile(
    fx_vol: pd.DataFrame,
    ticker: str = 'EURUSDV3M Curncy',
    window: int = 252,
) -> pd.Series:
    """
    Expanding-window percentile rank of EUR/USD 3M ATM implied vol.

    Returns a value in [0, 1] representing where current vol sits in its
    own history. High percentile = macro uncertainty = conviction should
    be dampened.

    Parameters
    ----------
    fx_vol  : DataFrame with column `ticker`
    ticker  : column name (default 'EURUSDV3M Curncy')
    window  : minimum history before computing percentile (default 252)

    Returns
    -------
    pd.Series  values in [0, 1], NaN during warm-up
    """
    if ticker not in fx_vol.columns:
        raise ValueError(f"Ticker '{ticker}' not found in fx_vol.")

    vol = fx_vol[ticker].copy()
    vol = vol[~vol.index.duplicated()].astype(float)

    # Shift by 1: threshold at t uses history up to t-1 only,
    # consistent with expanding_quantile() in indicators.py.
    pct = vol.expanding(min_periods=window).rank(pct=True).shift(1)
    pct.name = 'vol_pct'
    return pct


def macro_regime_score(
    dollar:  pd.Series = None,
    equity:  pd.Series = None,
    credit:  pd.Series = None,
) -> pd.Series:
    """
    Aggregate macro signal: mean of available non-NaN component signals.

    Returns a continuous score in [-1, +1]:
      +1 → all signals say bullish macro environment
       0 → signals are mixed or unavailable
      -1 → all signals say bearish macro environment

    Parameters
    ----------
    dollar : output of dollar_signal()
    equity : output of equity_signal()
    credit : output of credit_signal()

    Returns
    -------
    pd.Series  values in [-1, +1]; 0 where no signal is available
    """
    signals = [s for s in [dollar, equity, credit] if s is not None]
    if not signals:
        raise ValueError("At least one signal must be provided.")

    combined = pd.concat(signals, axis=1)
    score = combined.mean(axis=1, skipna=True)
    score = score.fillna(0.0)
    score.name = 'macro_score'
    return score


def blend_regime_strength(
    commodity_strength: pd.DataFrame,
    macro_score:        pd.Series,
    alpha:      float = 0.5,
    min_factor: float = 0.2,
    max_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Multiply each commodity's regime strength by a macro-alignment factor.

    macro_factor = clip(1 + alpha × macro_score × sign(commodity_strength),
                        min_factor, max_factor)
    effective    = commodity_strength × macro_factor

    When macro and commodity agree: factor > 1 (up to max_factor).
    When they disagree:             factor < 1 (down to min_factor).
    When commodity_strength == 0:   factor irrelevant, effective stays 0.

    NaN in macro_score → factor = 1.0 (no adjustment, use raw strength).

    Parameters
    ----------
    commodity_strength : regime_strength DataFrame (dates × assets)
    macro_score        : output of macro_regime_score() (dates,)
    alpha              : macro weight (default 0.5)
    min_factor         : floor on macro_factor (default 0.2)
    max_factor         : ceiling on macro_factor (default 1.5)

    Returns
    -------
    pd.DataFrame  same shape as commodity_strength
    """
    # Align macro_score to commodity_strength index
    macro = macro_score.reindex(commodity_strength.index).fillna(0.0)

    direction = np.sign(commodity_strength)
    # broadcast macro series across columns
    macro_2d = pd.DataFrame(
        np.outer(macro.values, np.ones(commodity_strength.shape[1])),
        index=commodity_strength.index,
        columns=commodity_strength.columns,
    )

    macro_factor = 1.0 + alpha * macro_2d * direction
    macro_factor = macro_factor.clip(lower=min_factor, upper=max_factor)
    # flat positions stay flat regardless of factor
    macro_factor[commodity_strength == 0] = 1.0

    return commodity_strength * macro_factor


def apply_vol_cap(
    effective_strength: pd.DataFrame,
    vol_pct:            pd.Series,
    vol_cap_pct: float = 0.80,
    vol_dampen:  float = 0.5,
) -> pd.DataFrame:
    """
    Reduce effective regime strength on high-volatility macro days.

    On any date where vol_pct > vol_cap_pct, multiply every commodity's
    effective_strength by vol_dampen. This prevents large entries during
    macro crisis periods (e.g., 2008, 2011, 2015).

    Parameters
    ----------
    effective_strength : output of blend_regime_strength()
    vol_pct            : output of vol_percentile() — values in [0, 1]
    vol_cap_pct        : percentile threshold (default 0.80 = top 20 %)
    vol_dampen         : multiplier during high-vol regimes (default 0.5)

    Returns
    -------
    pd.DataFrame  same shape as effective_strength
    """
    pct = vol_pct.reindex(effective_strength.index).fillna(0.0)
    high_vol_mask = (pct > vol_cap_pct).values  # boolean array (n_dates,)

    result = effective_strength.copy()
    result.iloc[high_vol_mask] *= vol_dampen
    return result
