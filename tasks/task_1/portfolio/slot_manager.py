"""
Task 1 Slot Manager — Per-Commodity Positions
=============================================

This module converts Layer 1 and Layer 2 signal grids into a positions DataFrame
with values in {-1, 0, +1}. It tracks one active long or short per commodity.
Sizing, weights, leverage, and returns are handled later by portfolio_manager.

Entry rules
-----------
Flat position, shifted Layer 2 entry signal, and matching Layer 1 regime on the
execution bar. New signals are suppressed while a position is already open.

Exit rules
----------
The first of: fixed ATR stop, Bollinger Band profit exit, or regime mismatch.

No-lookahead: Layer 2 signals are shifted by one bar before entry. The loop then
uses only current and past values as it walks forward through time.
Pipeline role: Layer 2 entries -> positions -> portfolio manager.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from indicators import atr_pct as _atr_pct_fn, bollinger_bands


def build_positions(
    prices:     pd.DataFrame,
    layer1:     pd.DataFrame,
    layer2:     pd.DataFrame,
    bb_window:  int   = 100,
    bb_num_std: float = 2.0,
    atr_window: int   = 50,
    sl_mult:    float = 3.0,
) -> pd.DataFrame:
    """
    Build one position per commodity, bar-by-bar.

    Parameters
    ----------
    prices     : close prices (dates × assets)
    layer1     : regime grid  {-1, 0, +1, NaN}
    layer2     : entry signal {-1, 0, +1, NaN}
    bb_window  : Bollinger Band lookback (default 100)
    bb_num_std : Bollinger Band width in std deviations (default 2.0)
    atr_window : ATR lookback for Stop Loss calculation (default 50)
    sl_mult    : Stop Loss = entry_price × (1 ± sl_mult × ATR_pct) (default 3.0)

    Returns
    -------
    positions : DataFrame {-1, 0, +1} — open direction per commodity
    """
    _, upper_bb, lower_bb = bollinger_bands(prices, bb_window, bb_num_std)
    atr_df = _atr_pct_fn(prices, atr_window)

    # No look-ahead: signal at t -> entry executes at t+1.
    entry_exec = layer2.shift(1)

    cols = list(prices.columns)
    n    = len(prices)

    p_arr     = {c: prices[c].values     for c in cols}
    l1_arr    = {c: layer1[c].values     if c in layer1.columns   else np.full(n, np.nan) for c in cols}
    entry_arr = {c: entry_exec[c].values for c in cols}
    ub_arr    = {c: upper_bb[c].values   if c in upper_bb.columns else np.full(n, np.nan) for c in cols}
    lb_arr    = {c: lower_bb[c].values   if c in lower_bb.columns else np.full(n, np.nan) for c in cols}
    atr_arr   = {c: atr_df[c].values     if c in atr_df.columns   else np.full(n, np.nan) for c in cols}

    pos_out = {c: np.zeros(n) for c in cols}

    # Per-commodity state: None = flat, dict = {'dir', 'sl'} = open position
    active: dict[str, dict | None] = {c: None for c in cols}

    for t in range(n):

        # ── Phase 1: exits ─────────────────────────────────────────────────────
        for col in cols:
            s = active[col]
            if s is None:
                continue

            price_t = p_arr[col][t]
            if np.isnan(price_t):
                active[col] = None
                continue

            # Regime exit
            l1_t = l1_arr[col][t]
            if not np.isnan(l1_t) and l1_t != s['dir']:
                active[col] = None
                continue

            if s['dir'] == 1:       # long
                sl_exit = price_t < s['sl']
                bb_exit = not np.isnan(ub_arr[col][t]) and price_t > ub_arr[col][t]
            elif s['dir'] == -1:    # short
                sl_exit = price_t > s['sl']
                bb_exit = not np.isnan(lb_arr[col][t]) and price_t < lb_arr[col][t]
            else:
                active[col] = None  # corrupted state — close defensively
                continue

            if sl_exit or bb_exit:
                active[col] = None

        # ── Phase 2: entries ───────────────────────────────────────────────────
        for col in cols:
            if active[col] is not None:
                continue    # position open — suppress clustering

            price_t = p_arr[col][t]
            if np.isnan(price_t):
                continue

            entry_t = entry_arr[col][t]   # Shift already applied.
            if np.isnan(entry_t) or entry_t == 0:
                continue

            l1_t = l1_arr[col][t]
            if np.isnan(l1_t) or l1_t != entry_t:
                continue

            atr_t = atr_arr[col][t]
            if np.isnan(atr_t) or atr_t < 1e-6:
                continue

            direction = int(entry_t)
            sl_price  = price_t * (1.0 - direction * sl_mult * atr_t)

            active[col] = {
                'dir': direction,
                'sl':  sl_price,
            }

        # ── Phase 3: record ────────────────────────────────────────────────────
        for col in cols:
            pos_out[col][t] = active[col]['dir'] if active[col] is not None else 0

    return pd.DataFrame(pos_out, index=prices.index)
