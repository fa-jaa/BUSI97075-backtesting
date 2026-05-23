import pandas as pd

from portfolio.positions  import build_positions
from portfolio.weights    import build_weights, compute_portfolio_returns
from signals.signal_layer1 import layer1_signal
from signals.signal_exit   import long_exit_signal, short_exit_signal


def build_signals(
    prices: pd.DataFrame,
    layer1_fn     = layer1_signal,
    exit_long_fn  = long_exit_signal,
    exit_short_fn = short_exit_signal,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Long Leg
    
    Layer 1
    slope SMA200 > 0   lookback period 10days
    SMA50>SMA200

    Layer 2
    Price cross EMA 20


    Layer 3 Exit

    Price > BB100 st.dev 2

    Trailing stop technique
    Parabolic SAR achored on the entry signal


    IF ELSE = FLAT



    
    
    Short Leg

    Layer1

    SMA50 < SMA200

    Slope SMA 200 < 0 lookback 10 periods

    Layer 2 (reinforeced confirmation)

    Price < EMA20
    + RSI cross down 50 level

    Exit 3

    Parabolic SAR 


    IF ELSE = FLAT







    












    
    
    
    
    
    
    
    Compute entry/exit signals and positions.

    Entry fires on the first day the Layer 1 trend flips into a new direction
    (transition day). The position is then held until an exit signal fires.

    Pass None for any function to disable that component:
      layer1_fn=None      no entries (all positions stay flat)
      exit_long_fn=None   long positions held until an opposite entry fires
      exit_short_fn=None  short positions held until an opposite entry fires

    Returns
    -------
    (entry_long, entry_short, exit_long, exit_short, positions)
    """
    trend = layer1_fn(prices)
    prev  = trend.shift(1)

    # Enter on the first bar the trend changes to +1 or -1.
    entry_l = (trend == 1)  & (prev != 1)
    entry_s = (trend == -1) & (prev != -1)

    _no_exit = pd.DataFrame(False, index=prices.index, columns=prices.columns)

    ex_long  = exit_long_fn(prices)  if exit_long_fn  is not None else _no_exit
    ex_short = exit_short_fn(prices) if exit_short_fn is not None else _no_exit

    positions = build_positions(entry_l, entry_s, ex_long, ex_short)

    return entry_l, entry_s, ex_long, ex_short, positions


def run_strategy(
    prices: pd.DataFrame,
    *,
    returns:       pd.DataFrame = None,
    layer1_fn,
    exit_long_fn,
    exit_short_fn,
    vol_window:    int,
    exec_lag:      int,
    rebal_freq:    str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline: signals → positions → weights → portfolio returns.

    Returns (weights, portfolio_returns).
    """
    if returns is None:
        returns = prices.pct_change()

    _, _, _, _, positions = build_signals(prices, layer1_fn, exit_long_fn, exit_short_fn)

    weights = build_weights(
        positions, returns,
        vol_window=vol_window, exec_lag=exec_lag, rebal_freq=rebal_freq,
    )
    port_r = compute_portfolio_returns(weights, returns)

    return weights, port_r
