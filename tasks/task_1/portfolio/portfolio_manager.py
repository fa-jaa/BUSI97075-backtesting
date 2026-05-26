"""
Portfolio Manager — Weights and Risk Budget
============================================

Takes the positions DataFrame from slot_manager {-1, 0, +1} and computes
position weights using ATR-based bottom-up sizing, applies a gross leverage
cap, and returns daily portfolio returns.

Sizing
------
  stop_distance_pct = sl_mult × ATR_pct[t_entry]   (matches slot_manager SL)
  weight            = min(risk_per_trade / stop_distance_pct, max_weight) × direction

  Weights are locked at entry and do not change until the position closes.
  Capped at max_weight (default 20%) regardless of ATR.

Leverage cap
------------
  gross_leverage = Σ |weight| across all active positions.
  If gross_leverage ≥ leverage_cap when a new position opens → skip it.
  Existing positions are never force-closed by the cap.

Execution lag
-------------
  Weights are shifted by exec_lag (default 1) before being applied to returns,
  so the first bar of P&L is exec_lag bars after position entry.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from indicators import atr_pct as _atr_pct_fn


def build_portfolio(
    positions:      pd.DataFrame,
    prices:         pd.DataFrame,
    returns:        pd.DataFrame,
    atr_window:     int   = 50,
    sl_mult:        float = 3.0,    # must match sl_mult in slot_manager
    risk_per_trade: float = 0.01,   # fraction of capital risked per trade (1%)
    max_weight:     float = 0.20,   # hard cap on individual position weight (20%)
    leverage_cap:   float = 2.50,   # gross leverage cap
    exec_lag:       int   = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute weights and portfolio returns from a positions DataFrame.

    Parameters
    ----------
    positions      : {-1, 0, +1} per commodity (output of slot_manager.build_positions)
    prices         : close prices (dates × assets)
    returns        : daily returns (dates × assets)
    atr_window     : ATR lookback (default 50) — must match slot_manager
    sl_mult        : ATR multiplier for stop distance (default 3.0) — must match slot_manager
    risk_per_trade : capital fraction risked per trade (default 0.01 = 1%)
    max_weight     : hard cap on individual position weight (default 0.20 = 20%)
    leverage_cap   : gross leverage cap; new positions blocked above this (default 2.50)
    exec_lag       : weight shift before P&L calculation (default 1)

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
                # New position: size = risk / stop_distance, capped at max_weight
                atr_t = atr_arr[col][t]
                if np.isnan(atr_t) or atr_t < 1e-6 or current_gross >= leverage_cap:
                    locked_weight[col] = 0.0
                else:
                    stop_dist = sl_mult * atr_t
                    size      = min(risk_per_trade / stop_dist, max_weight)
                    locked_weight[col] = pos_t * size
                    current_gross     += abs(locked_weight[col])

            wgt_out[col][t] = locked_weight[col]
            prev_pos[col]   = pos_t

    weights_raw = pd.DataFrame(wgt_out, index=positions.index)
    weights     = weights_raw.shift(exec_lag)

    portfolio_returns = (weights * returns).sum(axis=1, min_count=1).fillna(0.0)

    return weights, portfolio_returns
