"""
Layer 1 — Regime Strength Filter (Enhanced)
=============================================

Same multi-lookback SMA slope aggregation as the baseline (signal_layer1_v2),
exposed here as the enhanced version so the enhanced pipeline is self-contained.

The continuous regime_strength() output is consumed by portfolio_manager_enhanced
to scale position sizing by regime conviction: stronger regime → larger position.

Signal values with 10 lookbacks (steps of 0.2):
  {-1, -0.8, -0.6, -0.4, -0.2, 0, +0.2, +0.4, +0.6, +0.8, +1}

Interpretation
--------------
  ±1.00  — all 10 lookbacks agree   → strong regime, full conviction
  ±0.60  — 8/10 lookbacks agree     → moderate regime
  ±0.20  — 6/10 lookbacks agree     → weak regime, partial conviction
   0.00  — 5/10 long vs 5/10 short  → genuinely uncertain → flat

Pipeline usage
--------------
  regime_strength()  → continuous [-1, +1]   for conviction-weighted sizing
  layer1_signal()    → binary    {-1, 0, +1} for Layer 2 and slot manager

References
----------
  Elaut & Erdos (2016) — Trends' Signal Strength and the Performance of CTAs
  Koijen, Moskowitz, Pedersen & Vrugt (2016) — Carry
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


def regime_strength(
    prices:    pd.DataFrame,
    fast:      int       = 50,
    slow:      int       = 200,
    lookbacks: list[int] = None,
) -> pd.DataFrame:
    """
    Compute continuous regime strength by aggregating slope signs across
    multiple lookback horizons.

    Parameters
    ----------
    prices    : close prices (dates × assets)
    fast      : fast SMA window — directional gate (default 50)
    slow      : slow SMA window — slope is measured here (default 200)
    lookbacks : list of slope lookback periods to aggregate over.
                Default [20, 30, 40, 50, 60, 70, 80, 90, 100, 110].
                Use an even number so that 0 is achievable when half disagree.

    Returns
    -------
    pd.DataFrame
        Regime strength ∈ [-1, +1]. NaN during warm-up.
        With 10 lookbacks: steps of 0.2 → {-1, -0.8, ..., 0, ..., +0.8, +1}
    """
    if lookbacks is None:
        lookbacks = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

    sma_fast = prices.rolling(fast, min_periods=fast).mean()
    sma_slow = prices.rolling(slow, min_periods=slow).mean()

    # Aggregate sign of SMA_slow slope across all lookbacks
    slope_signs = [np.sign(sma_slow.diff(lb)) for lb in lookbacks]
    slope_avg   = sum(slope_signs) / len(lookbacks)

    # Directional gate: slope strength must agree with MA crossover direction
    trend_dir = np.sign(sma_fast - sma_slow)
    agreement = (slope_avg * trend_dir) > 0
    strength  = slope_avg * agreement.astype(float)

    # Propagate NaN for warm-up period
    strength[sma_slow.isna() | sma_fast.isna()] = np.nan

    return strength


def layer1_signal(
    prices:    pd.DataFrame,
    filter_fast: int       = 50,
    filter_slow: int       = 200,
    lookbacks:   list[int] = None,
) -> pd.DataFrame:
    """
    Layer 1 binary regime signal — enhanced via multi-lookback aggregation.

    Wraps regime_strength() and returns sign() for backward compatibility
    with Layer 2 and slot_manager (which expect {-1, 0, +1}).

    Parameters
    ----------
    prices      : close prices (dates × assets)
    filter_fast : fast SMA window (default 50)
    filter_slow : slow SMA window (default 200)
    lookbacks   : slope lookback periods (default [20, 30, 40, 50, 60, 70, 80, 90, 100, 110])

    Returns
    -------
    pd.DataFrame
        Values in {-1, 0, +1, NaN}.
    """
    strength = regime_strength(prices, fast=filter_fast, slow=filter_slow,
                               lookbacks=lookbacks)
    binary = np.sign(strength)
    binary[strength.isna()] = np.nan
    return binary
