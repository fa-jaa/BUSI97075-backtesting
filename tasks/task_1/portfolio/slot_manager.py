"""
Slot Manager — Per-Commodity Position Builder
==============================================

Tracks a single long or short position per commodity.
Receives the pre-computed L1 (regime) and L2 (entry) signal grids and
returns a positions DataFrame {-1, 0, +1}.

No position sizing, no weights, no risk budget — that lives in portfolio_manager.

Entry
-----
  · Flat position AND L2 fires an event (signal at t → execution at t+1)
  · L1 regime at execution bar must match L2 direction
  · New signals are suppressed while a position is already open (no clustering)

Exit — earliest of
------------------
  · Long:  price < Stop Loss  OR  price > Upper BB(bb_window, bb_num_std)
  · Short: price > Stop Loss  OR  price < Lower BB(bb_window, bb_num_std)
  · L1 regime ends (regime no longer matches position direction)

Stop Loss
---------
  SL is fixed at entry:
    Long:  sl = entry_price × (1 − sl_mult × ATR_pct)
    Short: sl = entry_price × (1 + sl_mult × ATR_pct)
  sl_mult defaults to 3.0 (3 × ATR distance from entry price).
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

    # No look-ahead: signal at t → entry executes at t+1
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

    for t in range(n):  # temporal loop function, for each t in time series, read all teh commodities

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

            entry_t = entry_arr[col][t]   # shift already applied, no look ahead bias
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
