"""
Task 2 Portfolio Manager — Conviction Sizing And Costs
======================================================

This module converts Task 2 positions into executed weights and net daily
returns. It receives positions, prices, asset returns, and optional
regime_strength, then returns weights and a returns Series with gross returns
and transaction costs stored in attrs.

Enhancements over the Task 1 portfolio manager:

  Conviction-weighted risk budgeting
  -----------------------------------
  The base portfolio manager always risks a fixed fraction (risk_per_trade)
  regardless of how strong the regime signal is.

  Here, the risk per trade is scaled by the regime conviction at entry:

    effective_risk = risk_per_trade × |regime_strength[t]|

  where regime_strength in [-1, +1] is the continuous multi-lookback signal
  from signal_layer1_enhanced. This means:

    |strength| = 1.0  → full 1% risk  (all 10 lookbacks agree)
    |strength| = 0.6  → 0.6% risk     (moderate conviction)
    |strength| = 0.2  → 0.2% risk     (weak conviction)

  Everything else (stop distance, max_weight cap, leverage cap, exec_lag)
  follows the Task 1 structure.

Sizing formula
--------------
  stop_distance_pct = sl_mult × ATR_pct[t_entry]
  effective_risk    = risk_per_trade × |regime_strength[t_entry]|
  weight            = min(effective_risk / stop_distance_pct, max_weight) × direction

  sl_mult is only the portfolio sizing stop-distance assumption. It is
  separate from sar_initial_mult in the SAR slot manager.

Leverage cap
------------
  gross_leverage = Σ |weight| across all active positions.
  If gross_leverage ≥ leverage_cap when a new position opens → skip it.
  Existing positions are never force-closed by the cap.

Execution lag
-------------
  Weights are shifted by exec_lag (default 1) before being applied to returns.

Transaction costs
-----------------
  Portfolio returns are net of a simple one-way turnover cost:

    daily_cost = one_way_turnover × tx_cost_bps / 10_000

  where one_way_turnover = sum(abs(delta weights)) / 2, computed from executed
  weights after the execution lag.

No-lookahead: weights are shifted by exec_lag before P&L and cost calculation.
Pipeline role: SAR-managed positions -> conviction weights -> net returns.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from indicators import atr_pct as _atr_pct_fn


def build_portfolio(
    positions:       pd.DataFrame,
    prices:          pd.DataFrame,
    returns:         pd.DataFrame,
    regime_strength: pd.DataFrame = None,   # continuous [-1, +1] conviction signal
    atr_window:      int   = 50,
    sl_mult:         float = 3.0,
    risk_per_trade:  float = 0.01,
    max_weight:      float = 0.20,
    leverage_cap:    float = 2.50,
    exec_lag:        int   = 1,
    tx_cost_bps:     float = 2.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute conviction-weighted weights and portfolio returns.

    Parameters
    ----------
    positions       : {-1, 0, +1} per commodity (output of slot_manager)
    prices          : close prices (dates × assets)
    returns         : daily returns (dates × assets)
    regime_strength : continuous regime signal [-1, +1] from signal_layer1_enhanced.
                      If None, falls back to fixed risk_per_trade (base behaviour).
    atr_window      : ATR lookback (default 50)
    sl_mult         : ATR stop-distance assumption used for sizing (default 3.0).
                      Separate from sar_initial_mult in the slot manager.
    risk_per_trade  : maximum capital fraction risked per trade (default 0.01 = 1%)
    max_weight      : hard cap on individual position weight (default 0.20 = 20%)
    leverage_cap    : gross leverage cap; new positions blocked above this (default 2.50)
    exec_lag        : weight shift before P&L calculation (default 1)
    tx_cost_bps     : transaction cost in basis points per one-way turnover
                      (default 2.0)

    Returns
    -------
    weights           : DataFrame of locked weights per asset, shifted by exec_lag
    portfolio_returns : Series of daily portfolio returns
    """
    atr_df = _atr_pct_fn(prices, atr_window)

    cols = list(positions.columns)
    n    = len(positions)

    pos_arr = {c: positions[c].values for c in cols}
    atr_arr = {c: atr_df[c].values if c in atr_df.columns else np.full(n, np.nan) for c in cols}

    # Pre-extract regime strength arrays (None if not provided)
    str_arr = {}
    for c in cols:
        if regime_strength is not None and c in regime_strength.columns:
            str_arr[c] = regime_strength[c].values
        else:
            str_arr[c] = None

    wgt_out       = {c: np.zeros(n) for c in cols}
    locked_weight = {c: 0.0 for c in cols}
    prev_pos      = {c: 0   for c in cols}

    for t in range(n):
        current_gross = sum(abs(locked_weight[c]) for c in cols if prev_pos[c] != 0)

        for col in cols:
            pos_t  = int(pos_arr[col][t])
            prev_t = prev_pos[col]

            if pos_t == 0:
                locked_weight[col] = 0.0

            elif prev_t == 0 and pos_t != 0:
                atr_t = atr_arr[col][t]
                if np.isnan(atr_t) or atr_t < 1e-6 or current_gross >= leverage_cap:
                    locked_weight[col] = 0.0
                else:
                    # Scale risk by regime conviction at entry
                    conviction = 1.0
                    if str_arr[col] is not None:
                        s = str_arr[col][t]
                        if not np.isnan(s):
                            conviction = abs(s)

                    stop_dist      = sl_mult * atr_t
                    effective_risk = risk_per_trade * conviction
                    size           = min(effective_risk / stop_dist, max_weight)
                    locked_weight[col] = pos_t * size
                    current_gross     += abs(locked_weight[col])

            wgt_out[col][t] = locked_weight[col]
            prev_pos[col]   = pos_t

    weights_raw = pd.DataFrame(wgt_out, index=positions.index)
    weights     = weights_raw.shift(exec_lag)

    gross_returns = (weights * returns).sum(axis=1, min_count=1).fillna(0.0)

    applied_weights = weights.fillna(0.0)
    one_way_turnover = applied_weights.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    costs = one_way_turnover * (tx_cost_bps / 10_000.0)

    portfolio_returns = gross_returns - costs
    portfolio_returns.attrs['gross_returns'] = gross_returns
    portfolio_returns.attrs['tx_costs'] = costs
    portfolio_returns.attrs['tx_cost_bps'] = tx_cost_bps

    return weights, portfolio_returns
