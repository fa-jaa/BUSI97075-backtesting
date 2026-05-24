"""
Slot Manager — Concurrent Position Tracker
===========================================

Replaces the Layer 3 + build_positions + build_weights pipeline with a single
stateful loop that supports multiple concurrent positions per asset.

Each EMA cross (Layer 2 entry event) opens a NEW independent slot with its own
Parabolic SAR anchored at the execution price. Multiple concurrent slots per
asset (and per direction) are fully supported — if 3 EMA crosses occur within
one regime block, 3 slots run simultaneously, each with its own SAR.

Slot lifecycle
--------------
  Open  : Layer 2 entry event fires at t  →  slot opens at close of t+1 (exec bar)
           (entry_exec = entry_signal.shift(1))
  Close : earliest of
            · Long:  price < SAR  OR  price > Upper BB(bb_window, bb_num_std)
            · Short: price > SAR
            · Layer 1 regime ends (l1 transitions away from the slot direction)

Parabolic SAR update (identical to Layer 3 convention)
-------------------------------------------------------
  SAR[t]  = SAR[t-1] + AF * (EP[t-1] - SAR[t-1])
  EP updated after SAR, so today's new extreme feeds SAR[t+1].
  Long  SAR rises  (EP = running max since entry; initial SAR < entry price)
  Short SAR falls  (EP = running min since entry; initial SAR > entry price)

Vol-parity weighting
--------------------
  w_slot = direction / EWMA_vol(at entry)   locked at opening bar
  weight_asset[t] = Σ w_slot for all active slots in that asset
  Normalised: Σ |weight_asset| = 1 whenever any slot is active.
  Final weights are shifted by exec_lag (default 1).

Look-ahead convention
---------------------
  Layer 2 signal at t  →  slot opens at t+1  →  first P&L at t+2
  (entry_exec = entry_signal.shift(1); weights = weights.shift(exec_lag))
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
    vol_window:       int   = 30,
    exec_lag:         int   = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Stateful multi-slot position builder.

    Each EMA cross in Layer 2 opens a new concurrent slot per asset.
    Slots close independently on SAR/BB hit or regime end.

    Parameters
    ----------
    prices        : close prices (dates × assets)
    returns       : daily returns (dates × assets)
    entry_signal  : Layer 2 output — entry events in {-1, 0, +1, NaN}
    layer1        : Layer 1 output — regime in {-1, 0, +1, NaN}

    Returns
    -------
    positions         : DataFrame of integer slot counts per asset
                        (+N = N active long slots, -N = N active short slots)
    weights           : vol-parity weights, normalised to gross = 1,
                        shifted by exec_lag
    portfolio_returns : daily portfolio returns (Series)
    """
    _, upper_bb, _ = bollinger_bands(prices, bb_window, bb_num_std)
    atr_df         = _atr_pct_fn(prices, atr_window)

    ewma_vol = returns.ewm(span=vol_window, min_periods=vol_window // 2).std()
    ewma_vol = ewma_vol.replace(0, float('nan'))

    # Slot opens at execution bar: signal at t-1  →  slot at t
    entry_exec = entry_signal.shift(1)

    positions   = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights_raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for col in prices.columns:
        p     = prices[col].values
        l1    = layer1[col].values     if col in layer1.columns     else np.full(len(p), np.nan)
        entry = entry_exec[col].values
        ub    = upper_bb[col].values   if col in upper_bb.columns   else np.full(len(p), np.nan)
        atr   = atr_df[col].values     if col in atr_df.columns     else np.full(len(p), np.nan)
        vol   = ewma_vol[col].values   if col in ewma_vol.columns   else np.full(len(p), np.nan)

        pos_arr = np.zeros(len(p))
        wgt_arr = np.zeros(len(p))

        # active_slots: list of {dir, sar, af, ep, w}
        # dir : +1 (long) or -1 (short)
        # sar : current SAR level
        # af  : current acceleration factor
        # ep  : extreme point since entry
        # w   : vol-parity weight locked at entry (direction / ewma_vol)
        active_slots: list[dict] = []

        for t in range(len(p)):
            price_t = p[t]
            l1_t    = l1[t]

            if np.isnan(price_t):
                continue

            # ── 1. Regime-end forced close ─────────────────────────────────
            # Keep long slots only while l1 == +1; short slots only while l1 == -1.
            # If l1 == 0 or flips, all slots of the wrong direction are closed.
            if not np.isnan(l1_t):
                active_slots = [
                    s for s in active_slots
                    if (s['dir'] == 1 and l1_t == 1)
                    or (s['dir'] == -1 and l1_t == -1)
                ]

            # ── 2. Update SAR + check exits ────────────────────────────────
            # SAR updated first (using yesterday's EP), then EP updated from
            # today's price — same order as the standalone Layer 3 loop.
            surviving: list[dict] = []
            for s in active_slots:
                # Update SAR using old EP
                s['sar'] = s['sar'] + s['af'] * (s['ep'] - s['sar'])

                if s['dir'] == 1:   # long slot
                    if price_t > s['ep']:
                        s['ep'] = price_t
                        s['af'] = min(s['af'] + sar_af_step, sar_af_max)
                    bb_exit  = (not np.isnan(ub[t])) and (price_t > ub[t])
                    sar_exit = price_t < s['sar']
                    if not (bb_exit or sar_exit):
                        surviving.append(s)
                else:               # short slot
                    if price_t < s['ep']:
                        s['ep'] = price_t
                        s['af'] = min(s['af'] + sar_af_step, sar_af_max)
                    sar_exit = price_t > s['sar']
                    if not sar_exit:
                        surviving.append(s)

            active_slots = surviving

            # ── 3. Open new slot on entry event ───────────────────────────
            entry_t = entry[t]
            if (
                not np.isnan(entry_t)
                and entry_t != 0
                and not np.isnan(atr[t])
                and not np.isnan(vol[t])
                and not np.isnan(l1_t)
                and l1_t == entry_t          # regime must still match direction
            ):
                direction = int(entry_t)
                atr_val   = atr[t] if not np.isnan(atr[t]) else 0.02

                if direction == 1:
                    sar_init = price_t * (1.0 - sar_initial_mult * atr_val)
                else:
                    sar_init = price_t * (1.0 + sar_initial_mult * atr_val)

                active_slots.append({
                    'dir': direction,
                    'sar': sar_init,
                    'af':  sar_af_start,
                    'ep':  price_t,
                    'w':   direction / vol[t],  # vol-parity weight, locked at entry
                })

            # ── 4. Aggregate ───────────────────────────────────────────────
            pos_arr[t] = float(sum(s['dir'] for s in active_slots))
            wgt_arr[t] = sum(s['w'] for s in active_slots)

        positions[col]   = pos_arr
        weights_raw[col] = wgt_arr

    # Normalise: gross exposure = 1 whenever any slot is active
    gross   = weights_raw.abs().sum(axis=1).replace(0, float('nan'))
    weights = weights_raw.div(gross, axis=0).fillna(0.0)

    # Execution lag: weight used at t was computed from signal at t-1
    weights = weights.shift(exec_lag)

    portfolio_returns = (weights * returns).sum(axis=1, min_count=1).fillna(0.0)

    return positions, weights, portfolio_returns
