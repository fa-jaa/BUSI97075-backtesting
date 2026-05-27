"""
Task 2 Layer 1 — Enhanced Regime Strength
=========================================

This module creates the enhanced Layer 1 regime signal. It consumes close prices
and returns both a continuous conviction grid and a binary regime grid. The
continuous regime_strength() output is consumed by portfolio_manager_enhanced to
scale position sizing by regime conviction.

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

Macro inputs can scale conviction, but they do not change the binary trading
direction. No-lookahead: all moving averages and slopes use prices observed up
to the current row only.

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
from signals.macro_signal import blend_regime_strength, apply_vol_cap


def regime_strength(
    prices:      pd.DataFrame,
    fast:        int        = 50,
    slow:        int        = 200,
    lookbacks:   list[int]  = None,
    macro_score: pd.Series  = None,
    vol_pct:     pd.Series  = None,
    macro_alpha: float      = 0.5,
    vol_cap_pct: float      = 0.80,
    vol_dampen:  float      = 0.5,
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
        If macro_score is supplied the values are further scaled by a macro
        alignment factor, so the range effectively becomes [min_factor×−1, max_factor×+1].
    """
    if lookbacks is None:
        lookbacks = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

    sma_fast = prices.rolling(int(fast), min_periods=int(fast)).mean()
    sma_slow = prices.rolling(int(slow), min_periods=int(slow)).mean()

    # Aggregate sign of SMA_slow slope across all lookbacks
    slope_signs = [np.sign(sma_slow.diff(lb)) for lb in lookbacks]
    slope_avg   = sum(slope_signs) / len(lookbacks)

    # Directional gate: slope strength must agree with MA crossover direction
    trend_dir = np.sign(sma_fast - sma_slow)
    agreement = (slope_avg * trend_dir) > 0
    strength  = slope_avg * agreement.astype(float)

    # Propagate NaN for warm-up period
    strength[sma_slow.isna() | sma_fast.isna()] = np.nan

    # ── Macro adjustment (optional) ───────────────────────────────────────────
    if macro_score is not None:
        strength = blend_regime_strength(
            strength, macro_score,
            alpha=macro_alpha, min_factor=0.2, max_factor=1.5,
        )
    if vol_pct is not None:
        strength = apply_vol_cap(
            strength, vol_pct,
            vol_cap_pct=vol_cap_pct, vol_dampen=vol_dampen,
        )

    return strength


def layer1_signal(
    prices:      pd.DataFrame,
    filter_fast: int        = 50,
    filter_slow: int        = 200,
    lookbacks:   list[int]  = None,
    macro_score: pd.Series  = None,
    vol_pct:     pd.Series  = None,
    macro_alpha: float      = 0.5,
    vol_cap_pct: float      = 0.80,
    vol_dampen:  float      = 0.5,
) -> pd.DataFrame:
    """
    Build the binary enhanced regime signal from raw commodity strength.

    Wraps regime_strength() and returns sign() for backward compatibility
    with Layer 2 and slot_manager (which expect {-1, 0, +1}).

    Macro parameters are accepted for API consistency but the binary
    direction is always derived from raw commodity strength alone —
    macro only modulates conviction (sizing), not direction.

    Parameters
    ----------
    prices      : close prices (dates × assets)
    filter_fast : fast SMA window (default 50)
    filter_slow : slow SMA window (default 200)
    lookbacks   : slope lookback periods (default [20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    macro_score : output of macro_regime_score() — passed through for API symmetry
    vol_pct     : output of vol_percentile() — passed through for API symmetry
    macro_alpha : macro blend weight (default 0.5)
    vol_cap_pct : vol percentile threshold (default 0.80)
    vol_dampen  : multiplier during high-vol periods (default 0.5)

    Returns
    -------
    pd.DataFrame
        Values in {-1, 0, +1, NaN}.
    """
    raw_strength = regime_strength(
        prices, fast=filter_fast, slow=filter_slow, lookbacks=lookbacks,
    )
    binary = np.sign(raw_strength)
    binary[raw_strength.isna()] = np.nan
    return binary
