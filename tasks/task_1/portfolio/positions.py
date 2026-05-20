import numpy as np
import pandas as pd


def build_positions(
    entry_long: pd.DataFrame,
    entry_short: pd.DataFrame,
    exit_long: pd.DataFrame,
    exit_short: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stateful position series.

    For each asset at each date:
    - Flat  → enter long if entry_long fires, enter short if entry_short fires
    - Long  → exit to flat if exit_long fires, otherwise stay long
    - Short → exit to flat if exit_short fires, otherwise stay short

    Position at row t is determined from close-t data and represents the
    desired holding going into t+1 (exec_lag in build_weights handles the shift).
    """
    en_long  = entry_long.fillna(False).values.astype(bool)
    en_short = entry_short.fillna(False).values.astype(bool)
    ex_long  = exit_long.fillna(False).values.astype(bool)
    ex_short = exit_short.fillna(False).values.astype(bool)

    n_dates, n_assets = en_long.shape
    pos_array = np.zeros((n_dates, n_assets), dtype=np.float32)

    for j in range(n_assets):
        pos = 0

        for i in range(n_dates):
            if pos == 0:
                if en_long[i, j]:
                    pos = 1
                elif en_short[i, j]:
                    pos = -1

            elif pos == 1:
                if ex_long[i, j]:
                    pos = 0

            elif pos == -1:
                if ex_short[i, j]:
                    pos = 0

            pos_array[i, j] = pos

    return pd.DataFrame(pos_array, index=entry_long.index, columns=entry_long.columns)
