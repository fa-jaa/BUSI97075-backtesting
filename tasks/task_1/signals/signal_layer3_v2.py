"""
Layer 3 — Exit Conditions

Requires the entry signal (Layer 2 output, NOT shifted).
The shift by 1 is applied internally so the SAR is initialised at the
correct execution price (close of t+1, not the signal close of t).

Look-ahead convention
---------------------
Signal at t → execution at close of t+1 (backtester responsibility).

Long exits (any one triggers)
------------------------------
1. Close[t] > Upper Bollinger Band(200, 2σ)
2. Close[t] < SAR[t]   (trailing stop moving upward)

Short exits
-----------
1. Close[t] > SAR[t]   (trailing stop moving downward)

Parabolic SAR — initialisation at entry day t_0
------------------------------------------------
Long  : SAR_0 = entry_price * (1 - sar_initial_mult * ATR_pct[t_0])
Short : SAR_0 = entry_price * (1 + sar_initial_mult * ATR_pct[t_0])

Parabolic SAR — update from day t_0+1 onward
---------------------------------------------
SAR[t] = SAR[t-1] + AF * (EP - SAR[t-1])

EP (Extreme Point):
    Long  → highest close since entry  (moves SAR upward)
    Short → lowest  close since entry  (moves SAR downward)

AF (Acceleration Factor):
    Starts at af_start (0.02), increments by af_step (0.02) each time
    a new EP is recorded, capped at af_max (0.20).

Returns
-------
pd.DataFrame  — values in {-1, 0, +1}
    -1  exit long   (sell to close — cancels +1 position → flat)
    +1  exit short  (buy to cover  — cancels -1 position → flat)
     0  no exit
"""

import numpy as np
import pandas as pd
from indicators import atr_pct, bollinger_bands


def layer3_signal(
    prices:           pd.DataFrame,
    entry_signal:     pd.DataFrame,
    bb_window:        int   = 200,
    bb_num_std:       float = 2.0,
    atr_window:       int   = 30,
    sar_initial_mult: float = 2.5,
    sar_af_start:     float = 0.02,
    sar_af_step:      float = 0.02,
    sar_af_max:       float = 0.20,
) -> pd.DataFrame:
    """
    Layer 3 exit signal.

    Parameters
    ----------
    prices           : close prices (dates × assets)
    entry_signal     : Layer 2 output (NOT shifted) — values in {-1, 0, +1, NaN}
    bb_window        : Bollinger Band period
    bb_num_std       : Bollinger Band standard deviation multiplier
    atr_window       : ATR_pct rolling window
    sar_initial_mult : initial SAR distance from entry in ATR_pct units
    sar_af_start     : initial acceleration factor
    sar_af_step      : AF increment per new EP
    sar_af_max       : maximum AF
    """
    _, upper_bb, _ = bollinger_bands(prices, bb_window, bb_num_std)
    atr            = atr_pct(prices, atr_window)

    # shift by 1 so SAR is initialised at the execution price (close t+1),
    # not the signal price (close t)
    entry_exec = entry_signal.shift(1)

    exit_sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for col in prices.columns:
        p       = prices[col].values
        entry   = entry_exec[col].values
        atr_col = atr[col].values
        ub      = upper_bb[col].values
        out     = np.zeros(len(p))

        in_trade  = False
        direction = 0
        sar       = np.nan
        ep        = np.nan
        af        = sar_af_start

        for t in range(len(p)):
            if np.isnan(p[t]):
                continue

            # New entry: initialise SAR, EP, AF
            # Only open if not already in a trade
            if not in_trade and not np.isnan(entry[t]) and entry[t] != 0:
                direction = int(entry[t])
                ep        = p[t]
                af        = sar_af_start
                atr_val   = atr_col[t] if not np.isnan(atr_col[t]) else 0.02
                if direction == 1:
                    sar = p[t] * (1.0 - sar_initial_mult * atr_val)
                else:
                    sar = p[t] * (1.0 + sar_initial_mult * atr_val)
                in_trade = True
                continue

            if not in_trade:
                continue

            # Update SAR
            sar = sar + af * (ep - sar)

            if direction == 1:  # ── LONG ──────────────────────────────────
                # Update EP upward
                if p[t] > ep:
                    ep = p[t]
                    af = min(af + sar_af_step, sar_af_max)

                bb_exit  = (not np.isnan(ub[t])) and (p[t] > ub[t])
                sar_exit = p[t] < sar

                if bb_exit or sar_exit:
                    out[t]   = -1.0   # sell to close long → flat
                    in_trade = False

            else:               # ── SHORT ─────────────────────────────────
                # Update EP downward
                if p[t] < ep:
                    ep = p[t]
                    af = min(af + sar_af_step, sar_af_max)

                if p[t] > sar:
                    out[t]   = 1.0    # buy to cover short → flat
                    in_trade = False

        exit_sig[col] = out

    return exit_sig
