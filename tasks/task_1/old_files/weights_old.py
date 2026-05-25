import pandas as pd


def build_weights(
    positions:  pd.DataFrame,
    returns:    pd.DataFrame,
    vol_window: int = 30,
    exec_lag:   int = 1,
) -> pd.DataFrame:
    """
    Vol-parity weights locked at trade entry — event-driven rebalancing only.

    The weight for each asset is computed once when a trade opens
    (position transitions 0 → ±1) and held constant until the position
    closes. This preserves the entry/exit logic from Layer 2/3 without
    any time-based rebalancing cutting trades early or late.

    w_i = position_i / EWMA_vol_i   computed at entry, held fixed

    Rebalancing is event-driven: weights change only when a trade opens
    or closes (Layer 2/3 signals). No time-based rebalancing (e.g. weekly)
    is applied — cutting positions on a fixed schedule would interrupt
    trends still in progress, which is inconsistent with trend-following.

    exec_lag shifts the weight forward by 1 day so execution happens
    at the close following the signal date.
    """
    ewma_vol = returns.ewm(span=vol_window, min_periods=vol_window // 2).std()
    ewma_vol = ewma_vol.replace(0, float('nan'))

    vol_scaled = positions.div(ewma_vol)

    # entry event: position transitions from flat (0) to active (±1)
    entry_event = (positions != 0) & (positions.shift(1).fillna(0) == 0)

    # on entry days  → lock in the vol-scaled weight
    # on flat days   → weight is 0
    # all other days → NaN, forward-filled to carry the entry weight forward
    weights = pd.DataFrame(float('nan'), index=positions.index, columns=positions.columns)
    weights[entry_event]    = vol_scaled[entry_event]
    weights[positions == 0] = 0.0

    weights = weights.ffill().fillna(0.0)

    # normalise: gross exposure = 1 at all times when any position is active
    gross = weights.abs().sum(axis=1).replace(0, float('nan'))
    weights = weights.div(gross, axis=0).fillna(0.0)

    return weights.shift(exec_lag)


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Daily portfolio excess returns."""
    portfolio_returns = (weights * returns).sum(axis=1, min_count=1)
    return portfolio_returns.fillna(0.0)
