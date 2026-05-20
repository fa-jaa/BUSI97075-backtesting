import pandas as pd


def build_weights(
    positions: pd.DataFrame,
    returns:   pd.DataFrame,
    vol_window: int,
    exec_lag:   int,
    rebal_freq: str,
) -> pd.DataFrame:
    """
    Volatility parity weights with weekly rebalancing.

    Each active position is sized as 1/vol so every asset contributes the
    same expected volatility to the portfolio. Weights are then scaled so
    sum(|w|) = 1 (fully invested, gross).

    Step 1 — estimate vol:    ewma_vol_i  (EWMA std over vol_window days)
    Step 2 — vol parity size: raw_i = position_i / ewma_vol_i
    Step 3 — normalise:       w_i   = raw_i / sum(|raw_j|)

    rebal_freq='W' snapshots weights at Friday close and holds them flat
    until the next Friday, avoiding daily drift from vol re-estimation.
    exec_lag shifts weights forward so the trade executes one day after
    the signal date.
    """
    # Step 1: EWMA volatility estimate for each asset.
    ewma_vol = returns.ewm(span=vol_window, min_periods=vol_window // 2).std()
    ewma_vol = ewma_vol.replace(0, float('nan'))

    # Step 2: size each position inversely proportional to its volatility.
    vol_parity = positions.div(ewma_vol)

    # Step 3: normalise so the portfolio is fully invested (gross exposure = 1).
    gross = vol_parity.abs().sum(axis=1).replace(0, float('nan'))
    weights = vol_parity.div(gross, axis=0)

    # Snapshot at end of each rebal period; hold flat until next period.
    if rebal_freq is not None:
        weights = (
            weights
            .resample(rebal_freq).last()
            .reindex(positions.index, method='ffill')
        )

    return weights.shift(exec_lag)


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Daily portfolio excess returns."""
    portfolio_returns = (weights * returns).sum(axis=1, min_count=1)
    return portfolio_returns.fillna(0.0)
