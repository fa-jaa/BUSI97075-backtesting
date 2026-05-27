"""
Task 2 Strategy — Enhanced Trend-Following Pipeline
===================================================

This module orchestrates the Task 2 enhanced strategy. It keeps the Task 1
trend-following structure and adds multi-lookback conviction, optional macro
modulation, stricter short-entry filtering, Parabolic SAR exits, conviction
sizing, and transaction costs.

Strategy overview
-----------------
Enhanced Layer 1 keeps the commodity trend direction but measures conviction
across multiple SMA-slope lookbacks. The binary direction still feeds Layer 2
and the slot manager; the continuous regime_strength feeds position sizing.

Macro signals are optional sizing overlays. Dollar, equity, credit, and FX-vol
inputs can boost or dampen conviction, but they do not flip trade direction.

Enhanced Layer 2 keeps EMA/RSI entry timing and uses the stricter short setup:
RSI must break below 60 within a 5-bar flag window. The short oversold cap is
checked only at execution time in the slot manager; the current default blocks
shorts when execution-bar RSI is below 50.

The enhanced slot manager replaces the fixed ATR stop with Parabolic SAR. The
initial SAR placement uses sar_initial_mult * ATR_pct, while sar_af_start,
sar_af_step, and sar_af_max are the Parabolic SAR acceleration factors. The
portfolio manager applies conviction-weighted sizing, gross leverage control,
execution lag, and transaction costs.

Main inputs
-----------
prices : daily close prices, indexed by date and with one column per commodity.
returns : optional daily returns aligned to prices; computed from prices if None.
external : optional training-period macro data used only to scale conviction.

Main outputs
------------
build_signals() returns Layer 1, Layer 2, regime_strength, and optional macro
diagnostic series. run_strategy() returns positions, weights, and net returns.

No-lookahead convention
-----------------------
Layer 2 records signals at close t. The slot manager shifts entries to t+1 and
checks the short RSI cap on that execution bar. The portfolio manager shifts
weights by exec_lag before applying returns and transaction costs.

Pipeline
--------
prices -> enhanced signals -> SAR-managed positions -> conviction weights ->
net portfolio returns
"""

import pandas as pd

from signals.signal_layer1_enhanced   import layer1_signal, regime_strength as _regime_strength_fn
from signals.signal_layer2_enhanced   import layer2_signal
from signals.macro_signal             import (dollar_signal, equity_signal, credit_signal,
                                              vol_percentile, macro_regime_score,
                                              blend_regime_strength, apply_vol_cap)
from portfolio.slot_manager_enhanced  import build_positions
from portfolio.portfolio_manager_enhanced import build_portfolio


def build_signals(
    prices: pd.DataFrame,
    # Layer 1
    filter_fast:    int        = 50,
    filter_slow:    int        = 200,
    lookbacks:      list[int]  = None,
    # Layer 2
    ema_window:      int   = 14,
    rsi_window:      int   = 14,
    rsi_long_level:  float = 50.0,
    rsi_level:       float = 60.0,
    rsi_flag_window: int   = 5,
    # External macro data
    external:    dict  = None,
    macro_alpha: float = 0.5,
    macro_fast:  int   = 50,
    macro_slow:  int   = 200,
    vol_cap_pct: float = 0.80,
    vol_dampen:  float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Run the enhanced signal stack.

    Layer 1 returns both binary direction and continuous regime strength. Macro
    inputs, when supplied, adjust conviction only. Layer 2 creates entry events;
    the short RSI cap is intentionally checked later at execution time.

    Parameters
    ----------
    external : optional dict with keys:
               'fx_spot'  — DataFrame from fx_spot_levels.csv
               'equity'   — DataFrame from equity_indices_levels.csv
               'credit'   — DataFrame from credit_spreads_levels.csv
               'fx_vol'   — DataFrame from atm_vols_levels.csv
               Any subset of keys is accepted; missing keys are ignored.

    Returns
    -------
    dict with keys 'layer1', 'layer2', 'regime_strength', and optionally
    'macro_score', 'vol_pct'
    """
    # ── Macro signals (computed from external data if provided) ───────────────
    macro_score_s = None
    vol_pct_s     = None

    if external:
        fx_data  = external.get('fx_spot')
        eq_data  = external.get('equity')
        cr_data  = external.get('credit')
        vol_data = external.get('fx_vol')

        components = {}
        if fx_data is not None:
            try:
                components['dollar'] = dollar_signal(fx_data, fast=macro_fast, slow=macro_slow)
            except (ValueError, KeyError):
                pass
        if eq_data is not None:
            try:
                components['equity'] = equity_signal(eq_data, fast=macro_fast, slow=macro_slow)
            except (ValueError, KeyError):
                pass
        if cr_data is not None:
            try:
                components['credit'] = credit_signal(cr_data, fast=macro_fast, slow=macro_slow)
            except (ValueError, KeyError):
                pass

        if components:
            macro_score_s = macro_regime_score(
                dollar=components.get('dollar'),
                equity=components.get('equity'),
                credit=components.get('credit'),
            )

        if vol_data is not None:
            try:
                vol_pct_s = vol_percentile(vol_data)
            except (ValueError, KeyError):
                pass

    # ── Layer 1: commodity regime strength (macro-adjusted if available) ──────
    strength = _regime_strength_fn(
        prices,
        fast        = filter_fast,
        slow        = filter_slow,
        lookbacks   = lookbacks,
        macro_score = macro_score_s,
        vol_pct     = vol_pct_s,
        macro_alpha = macro_alpha,
        vol_cap_pct = vol_cap_pct,
        vol_dampen  = vol_dampen,
    )

    # Binary direction: always commodity-only (macro does not change direction)
    l1 = layer1_signal(
        prices,
        filter_fast = filter_fast,
        filter_slow = filter_slow,
        lookbacks   = lookbacks,
    )

    l2 = layer2_signal(
        prices,
        layer1          = l1,
        ema_window      = ema_window,
        rsi_window      = rsi_window,
        rsi_long_level  = rsi_long_level,
        rsi_level       = rsi_level,
        rsi_flag_window = rsi_flag_window,
    )

    result = {'layer1': l1, 'layer2': l2, 'regime_strength': strength}
    if macro_score_s is not None:
        result['macro_score'] = macro_score_s
    if vol_pct_s is not None:
        result['vol_pct'] = vol_pct_s
    return result


def run_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame = None,
    # Layer 1
    filter_fast:    int        = 50,
    filter_slow:    int        = 200,
    lookbacks:      list[int]  = None,
    # Layer 2
    ema_window:      int   = 14,
    rsi_window:      int   = 14,
    rsi_long_level:  float = 50.0,
    rsi_level:       float = 60.0,
    rsi_flag_window: int   = 5,
    rsi_short_cap:   float = 50.0,   # checked at execution bar in slot manager
    # Slot manager — Parabolic SAR trailing stop
    bb_window:        int   = 100,
    bb_num_std:       float = 2.0,
    atr_window:       int   = 50,
    sar_initial_mult: float = 3.0,
    sar_af_start:     float = 0.02,
    sar_af_step:      float = 0.02,
    sar_af_max:       float = 0.20,
    sar_grace_period: int   = 10,
    # Portfolio manager — sizing and risk
    sl_mult:        float = 3.0,
    risk_per_trade: float = 0.01,
    max_weight:     float = 0.20,
    leverage_cap:   float = 2.50,
    exec_lag:       int   = 1,
    tx_cost_bps:    float = 2.0,
    # External macro data (optional)
    external:    dict  = None,
    macro_alpha: float = 0.5,
    macro_fast:  int   = 50,
    macro_slow:  int   = 200,
    vol_cap_pct: float = 0.80,
    vol_dampen:  float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Run the full enhanced pipeline: signals -> positions -> weights -> returns.

    Returns
    -------
    (positions, weights, portfolio_returns)
    """
    if returns is None:
        returns = prices.pct_change()

    signals = build_signals(
        prices,
        filter_fast     = filter_fast,
        filter_slow     = filter_slow,
        lookbacks       = lookbacks,
        ema_window      = ema_window,
        rsi_window      = rsi_window,
        rsi_long_level  = rsi_long_level,
        rsi_level       = rsi_level,
        rsi_flag_window = rsi_flag_window,
        external        = external,
        macro_alpha     = macro_alpha,
        macro_fast      = macro_fast,
        macro_slow      = macro_slow,
        vol_cap_pct     = vol_cap_pct,
        vol_dampen      = vol_dampen,
    )

    positions = build_positions(
        prices,
        layer1           = signals['layer1'],
        layer2           = signals['layer2'],
        bb_window        = bb_window,
        bb_num_std       = bb_num_std,
        atr_window       = atr_window,
        sar_initial_mult = sar_initial_mult,
        sar_af_start     = sar_af_start,
        sar_af_step      = sar_af_step,
        sar_af_max       = sar_af_max,
        sar_grace_period = sar_grace_period,
        rsi_window       = rsi_window,
        rsi_short_cap    = rsi_short_cap,
    )

    weights, port_returns = build_portfolio(
        positions,
        prices,
        returns,
        regime_strength = signals['regime_strength'],
        atr_window      = atr_window,
        sl_mult         = sl_mult,
        risk_per_trade  = risk_per_trade,
        max_weight      = max_weight,
        leverage_cap    = leverage_cap,
        exec_lag        = exec_lag,
        tx_cost_bps     = tx_cost_bps,
    )

    return positions, weights, port_returns
