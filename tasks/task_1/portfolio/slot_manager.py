"""
Slot Manager — Concurrent Position Tracker (Bottom-Up Sizing)
==============================================================

Each EMA cross (Layer 2 entry event) opens a NEW independent slot with its own
Parabolic SAR. Multiple concurrent slots per asset are fully supported.

Position sizing — bottom-up (fixed risk per slot)
--------------------------------------------------
  stop_distance_pct = sar_initial_mult × ATR_pct[t]   (= initial SAR gap / price)
  size              = min(risk_per_slot / stop_distance_pct, max_slot_size)
  weight_slot       = direction × size

  max_slot_size (default 20%) caps individual slots — prevents a tiny ATR from
  inflating a single slot to an unrealistic size.

  Example: ATR_pct = 2%, sar_initial_mult = 2.5
           stop_distance = 5%,  size = 1% / 5% = 20%  (at the cap)

Gross leverage cap
------------------
  gross_leverage = Σ |weight_slot|  across all active slots and all assets
  If gross_leverage ≥ leverage_cap (default 250%) → new slots are blocked.
  Existing slots are NOT force-closed; the cap only prevents new openings.

  The cap check happens AFTER all exits are processed for the current bar
  and BEFORE any new entries — this requires a time-first loop (all assets
  processed per day) rather than an asset-first loop.

Per-asset slot limit
--------------------
  max_slots_per_asset (default 2) caps concurrent slots on the same commodity.
  A third EMA cross on Brent Crude while 2 slots are already open is skipped.

Weights
-------
  weight_asset[t] = Σ weight_slot for all active slots of that asset
  No daily renormalisation — weights are fixed at entry and do not change
  until the slot closes. This avoids the implicit daily portfolio rebalancing
  of the vol-parity approach.

  Final weights are shifted by exec_lag (default 1 day).

Slot lifecycle
--------------
  Open  : Layer 2 entry event at t-1  →  slot opens at close of t (exec bar)
  Close : earliest of
            · Long:  price < SAR  OR  price > Upper BB(bb_window, bb_num_std)
            · Short: price > SAR
            · Layer 1 regime ends (l1 changes away from slot direction)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from indicators import atr_pct as _atr_pct_fn, bollinger_bands


def build_slot_positions(
    prices:           pd.DataFrame,
    returns:          pd.DataFrame,
    entry_signal:     pd.DataFrame,
    layer1:           pd.DataFrame,
    bb_window:        int   = 100,
    bb_num_std:       float = 2.0,
    atr_window:       int   = 20,
    sar_initial_mult: float = 2.5,
    sar_af_start:     float = 0.02,
    sar_af_step:      float = 0.02,
    sar_af_max:       float = 0.20,
    sar_grace_period: int   = 10,       # bars before AF is allowed to increase
    risk_per_slot:      float = 0.01,   # fraction of capital risked per slot (1%)
    max_slot_size:      float = 0.20,   # max weight per single slot (20%) — low-ATR cap
    leverage_cap:       float = 2.50,   # max gross leverage — blocks new slots above this
    max_slots_per_asset: int  = 2,      # max concurrent slots per commodity
    exec_lag:           int   = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Stateful multi-slot position builder with bottom-up sizing.

    Parameters
    ----------
    prices               : close prices (dates × assets)
    returns              : daily returns (dates × assets)
    entry_signal         : Layer 2 output — entry events in {-1, 0, +1, NaN}
    layer1               : Layer 1 output — regime in {-1, 0, +1, NaN}
    risk_per_slot        : fraction of capital risked per slot (default 0.01 = 1%)
    max_slot_size        : hard cap on individual slot weight (default 0.20 = 20%)
    leverage_cap         : gross leverage cap; new slots blocked when reached (default 2.5 = 250%)
    max_slots_per_asset  : max concurrent open slots per commodity (default 2)

    Returns
    -------
    positions         : DataFrame of integer slot counts (+N long, -N short)
    weights           : bottom-up weights (not normalised), shifted by exec_lag
    portfolio_returns : daily portfolio returns (Series)
    """
    _, upper_bb, _ = bollinger_bands(prices, bb_window, bb_num_std)
    atr_df         = _atr_pct_fn(prices, atr_window)

    # Slot opens at execution bar: Layer 2 signal at t-1 → slot at t
    entry_exec = entry_signal.shift(1)

    cols = list(prices.columns)
    n    = len(prices)

    # Pre-extract numpy arrays for inner-loop speed
    p_arr     = {c: prices[c].values      for c in cols}
    l1_arr    = {c: layer1[c].values      if c in layer1.columns      else np.full(n, np.nan) for c in cols}
    entry_arr = {c: entry_exec[c].values  for c in cols}
    ub_arr    = {c: upper_bb[c].values    if c in upper_bb.columns    else np.full(n, np.nan) for c in cols}
    atr_arr   = {c: atr_df[c].values      if c in atr_df.columns      else np.full(n, np.nan) for c in cols}

    pos_out = {c: np.zeros(n) for c in cols}
    wgt_out = {c: np.zeros(n) for c in cols}

    # active_slots[col]: list of {dir, sar, af, ep, w}
    active_slots: dict[str, list[dict]] = {c: [] for c in cols}

    for t in range(n):

        # ── Phase 1: exits for all assets ─────────────────────────────────────
        # (regime close + SAR/BB exits — must happen before the leverage check)
        for col in cols:
            price_t = p_arr[col][t]
            if np.isnan(price_t):
                continue

            l1_t = l1_arr[col][t]

            # Force-close slots whose direction no longer matches the regime
            if not np.isnan(l1_t):
                active_slots[col] = [
                    s for s in active_slots[col]
                    if (s['dir'] == 1 and l1_t == 1)
                    or (s['dir'] == -1 and l1_t == -1)
                ]

            # Update SAR then check exits
            surviving: list[dict] = []
            for s in active_slots[col]:
                s['age'] += 1
                # SAR update uses yesterday's EP (same order as Layer 3)
                s['sar'] = s['sar'] + s['af'] * (s['ep'] - s['sar'])

                if s['dir'] == 1:   # long slot
                    if price_t > s['ep']:
                        s['ep'] = price_t
                        if s['age'] > sar_grace_period:
                            s['af'] = min(s['af'] + sar_af_step, sar_af_max)
                    bb_exit  = (not np.isnan(ub_arr[col][t])) and (price_t > ub_arr[col][t])
                    sar_exit = price_t < s['sar']
                    if not (bb_exit or sar_exit):
                        surviving.append(s)
                else:               # short slot
                    if price_t < s['ep']:
                        s['ep'] = price_t
                        if s['age'] > sar_grace_period:
                            s['af'] = min(s['af'] + sar_af_step, sar_af_max)
                    sar_exit = price_t > s['sar']
                    if not sar_exit:
                        surviving.append(s)

            active_slots[col] = surviving

        # ── Phase 2: gross leverage after exits ───────────────────────────────
        current_gross = sum(
            abs(s['w']) for slots in active_slots.values() for s in slots
        )

        # ── Phase 3: open new slots (blocked if cap reached) ──────────────────
        for col in cols:
            price_t = p_arr[col][t]
            if np.isnan(price_t):
                continue

            entry_t = entry_arr[col][t]
            if np.isnan(entry_t) or entry_t == 0:
                continue

            atr_t = atr_arr[col][t]
            l1_t  = l1_arr[col][t]

            if np.isnan(atr_t) or np.isnan(l1_t) or l1_t != entry_t:
                continue

            if len(active_slots[col]) >= max_slots_per_asset:
                continue    # per-asset slot limit reached

            if current_gross >= leverage_cap:
                continue    # gross leverage cap reached — block new slot

            direction  = int(entry_t)
            stop_dist  = sar_initial_mult * atr_t
            if stop_dist < 1e-6:
                continue    # degenerate ATR — skip

            size        = min(risk_per_slot / stop_dist, max_slot_size)
            slot_weight = direction * size

            if direction == 1:
                sar_init = price_t * (1.0 - sar_initial_mult * atr_t)
            else:
                sar_init = price_t * (1.0 + sar_initial_mult * atr_t)

            active_slots[col].append({
                'dir': direction,
                'sar': sar_init,
                'af':  sar_af_start,
                'ep':  price_t,
                'w':   slot_weight,     # locked at entry — never changes
                'age': 0,              # bars alive; AF frozen until > sar_grace_period
            })
            current_gross += abs(slot_weight)   # update running total for this bar

        # ── Phase 4: aggregate positions and weights ───────────────────────────
        for col in cols:
            pos_out[col][t] = float(sum(s['dir'] for s in active_slots[col]))
            wgt_out[col][t] = sum(s['w'] for s in active_slots[col])

    positions   = pd.DataFrame(pos_out, index=prices.index)
    weights_raw = pd.DataFrame(wgt_out, index=prices.index)

    # No normalisation — weights are the actual bottom-up sizes
    weights = weights_raw.shift(exec_lag)

    portfolio_returns = (weights * returns).sum(axis=1, min_count=1).fillna(0.0)

    return positions, weights, portfolio_returns
