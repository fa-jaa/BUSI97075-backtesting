"""
Strategy v2 Enhanced — Trend-Following Pipeline
=================================================

Enhancements over strategy_v2 (base):

  1. Layer 1 — Conviction-weighted sizing
       regime_strength() (continuous [-1,+1]) is passed to the portfolio manager.
       Position size is scaled by |conviction|: full regime → 1% risk, weak → 0.2% risk.
       → signal_layer1_v2_enhanced + portfolio_manager_enhanced

  2. Layer 2 — Short RSI cap
       Short entries blocked when RSI < rsi_short_cap (default 40).
       Avoids shorting already-oversold assets.
       → signal_layer2_v2_enhanced

  3. Slot Manager — Parabolic SAR trailing stop
       Fixed ATR stop replaced by a SAR that trails the price.
       Gives trades room to run while locking in profits as the trend matures.
       → slot_manager_enhanced

Pipeline
--------
prices
  │
  ▼
layer1_signal          multi-lookback SMA slope aggregation → binary {-1,0,+1}
regime_strength        same signal, continuous [-1,+1]      → passed to portfolio
  │
  ▼
layer2_signal          enhanced: short RSI cap (RSI >= rsi_short_cap)
  │
  ▼
build_positions        enhanced: Parabolic SAR trailing stop
  │
  ▼
build_portfolio        enhanced: conviction-weighted sizing
  │
  ▼
portfolio_returns
"""

import pandas as pd

from signals.signal_layer1_v2_enhanced  import layer1_signal, regime_strength as _regime_strength_fn
from signals.signal_layer2_v2_enhanced  import layer2_signal
from portfolio.slot_manager_enhanced    import build_positions
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
    rsi_short_cap:   float = 40.0,
) -> dict[str, pd.DataFrame]:
    """
    Run Layer 1 (regime) and Layer 2 enhanced (entry timing + short RSI cap).

    Returns
    -------
    dict with keys 'layer1', 'layer2', 'regime_strength'
    """
    strength = _regime_strength_fn(
        prices,
        fast      = filter_fast,
        slow      = filter_slow,
        lookbacks = lookbacks,
    )

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
        rsi_short_cap   = rsi_short_cap,
    )

    return {'layer1': l1, 'layer2': l2, 'regime_strength': strength}


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
    rsi_short_cap:   float = 40.0,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Full enhanced pipeline: signals → positions → conviction-weighted weights → returns.

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
        rsi_short_cap   = rsi_short_cap,
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
    )

    return positions, weights, port_returns
