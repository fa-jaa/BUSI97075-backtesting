"""
Task 2 Slot Manager — SAR Positions And Short RSI Execution Cap
===============================================================

This module converts enhanced Layer 1 and Layer 2 signals into one open position
per commodity. It receives prices and signal grids, then returns a positions
DataFrame with values in {-1, 0, +1}. Sizing and transaction costs are handled
later by portfolio_manager_enhanced.

Enhancements over the Task 1 slot manager:

- The fixed ATR stop is replaced by a Parabolic SAR trailing stop.
- Short entries are blocked when execution-bar RSI is below rsi_short_cap.

Stop Loss — Parabolic SAR (trailing)
-------------------------------------
  The SAR is initialised at entry using sar_initial_mult × ATR_pct from the
  entry price, then trails the price as the trade moves in our favour.
  sar_initial_mult only controls this initial placement distance; it is not
  the Parabolic SAR acceleration factor.

  Each bar:
    SAR = SAR + AF × (EP - SAR)

  When a new extreme price is recorded (new high for long, new low for short)
  and the position has lived longer than sar_grace_period bars:
    AF = min(AF + sar_af_step, sar_af_max)

  The grace period prevents the AF from accelerating immediately after entry,
  giving the trade room to breathe before the stop starts tightening.

  Long exit:  price < SAR
  Short exit: price > SAR

BB exit and regime exit are unchanged from the Task 1 slot manager.

No-lookahead: Layer 2 signals are shifted by one bar before entry, and the RSI
cap is checked on the actual execution bar. Pipeline role: enhanced Layer 2
events -> SAR-managed positions -> enhanced portfolio manager.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from indicators import atr_pct as _atr_pct_fn, bollinger_bands, rsi as _rsi_fn


def build_positions(
    prices:           pd.DataFrame,
    layer1:           pd.DataFrame,
    layer2:           pd.DataFrame,
    bb_window:        int   = 100,
    bb_num_std:       float = 2.0,
    atr_window:       int   = 50,
    sar_initial_mult: float = 3.0,    # initial SAR placement distance in ATR_pct units
    sar_af_start:     float = 0.02,
    sar_af_step:      float = 0.02,
    sar_af_max:       float = 0.20,
    sar_grace_period: int   = 10,     # bars before AF is allowed to accelerate
    rsi_window:       int   = 14,
    rsi_short_cap:    float = 50.0,   # skip short entry if RSI < this at execution bar
) -> pd.DataFrame:
    """
    Build one position per commodity using Parabolic SAR as trailing stop.

    Parameters
    ----------
    prices           : close prices (dates × assets)
    layer1           : regime grid  {-1, 0, +1, NaN}
    layer2           : entry signal {-1, 0, +1, NaN}
    bb_window        : Bollinger Band lookback (default 100)
    bb_num_std       : Bollinger Band width in std deviations (default 2.0)
    atr_window       : ATR lookback (default 50)
    sar_initial_mult : initial SAR placement distance in ATR_pct units (default 3.0)
                       Distinct from the Parabolic SAR acceleration factors.
    sar_af_start     : initial Parabolic SAR acceleration factor (default 0.02)
    sar_af_step      : AF increment per new extreme price (default 0.02)
    sar_af_max       : maximum acceleration factor (default 0.20)
    sar_grace_period : bars before AF is allowed to increase (default 10)
    rsi_window       : RSI lookback for the short oversold cap (default 14)
    rsi_short_cap    : skip short entry if execution-bar RSI < this value (default 50.0)

    Returns
    -------
    positions : DataFrame {-1, 0, +1} — open direction per commodity
    """
    _, upper_bb, lower_bb = bollinger_bands(prices, bb_window, bb_num_std)
    atr_df  = _atr_pct_fn(prices, atr_window)
    rsi_df  = _rsi_fn(prices, rsi_window)

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
    rsi_arr   = {c: rsi_df[c].values     if c in rsi_df.columns   else np.full(n, 100.0)  for c in cols}

    pos_out = {c: np.zeros(n) for c in cols}

    # Per-commodity state: None = flat, dict = {'dir', 'sar', 'af', 'ep', 'age'}
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

            # SAR update
            s['age'] += 1
            s['sar']  = s['sar'] + s['af'] * (s['ep'] - s['sar'])

            if s['dir'] == 1:       # long
                if price_t > s['ep']:
                    s['ep'] = price_t
                    if s['age'] > sar_grace_period:
                        s['af'] = min(s['af'] + sar_af_step, sar_af_max)
                sl_exit = price_t < s['sar']
                bb_exit = not np.isnan(ub_arr[col][t]) and price_t > ub_arr[col][t]
            elif s['dir'] == -1:    # short
                if price_t < s['ep']:
                    s['ep'] = price_t
                    if s['age'] > sar_grace_period:
                        s['af'] = min(s['af'] + sar_af_step, sar_af_max)
                sl_exit = price_t > s['sar']
                bb_exit = not np.isnan(lb_arr[col][t]) and price_t < lb_arr[col][t]
            else:
                active[col] = None
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

            entry_t = entry_arr[col][t]
            if np.isnan(entry_t) or entry_t == 0:
                continue

            l1_t = l1_arr[col][t]
            if np.isnan(l1_t) or l1_t != entry_t:
                continue

            atr_t = atr_arr[col][t]
            if np.isnan(atr_t) or atr_t < 1e-6:
                continue

            direction = int(entry_t)

            # Oversold guard: skip short if RSI at execution bar is below the cap
            if direction == -1 and rsi_short_cap > 0:
                rsi_t = rsi_arr[col][t]
                if not np.isnan(rsi_t) and rsi_t < rsi_short_cap:
                    continue
            sar_init  = price_t * (1.0 - direction * sar_initial_mult * atr_t)

            active[col] = {
                'dir': direction,
                'sar': sar_init,
                'af':  sar_af_start,
                'ep':  price_t,
                'age': 0,
            }

        # ── Phase 3: record ────────────────────────────────────────────────────
        for col in cols:
            pos_out[col][t] = active[col]['dir'] if active[col] is not None else 0

    return pd.DataFrame(pos_out, index=prices.index)
