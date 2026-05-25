import numpy as np
import pandas as pd


def build_positions(
    entry_signal: pd.DataFrame,
    exit_signal:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Stateful position series.

    Parameters
    ----------
    entry_signal : output of layer2_signal — values in {-1, 0, +1, NaN}
                   +1 = enter long, -1 = enter short, 0/NaN = no entry
    exit_signal  : output of layer3_signal — values in {-1, 0, +1}
                   -1 = exit long (sell to close), +1 = exit short (buy to cover), 0 = no exit

    Note: layer3_signal must be called with layer2_signal.shift(1) as its
    entry_signal argument so that SAR is initialised at the correct execution price.

    For each asset at each date:
    - Flat  → enter long if entry_signal == +1, enter short if entry_signal == -1
    - Long  → exit to flat if exit_signal == -1, otherwise stay long
    - Short → exit to flat if exit_signal == +1, otherwise stay short

    Position at row t represents the desired holding going into t+1.
    The execution shift is handled downstream in build_weights.
    """
    en_long  = (entry_signal ==  1).fillna(False).values.astype(bool)
    en_short = (entry_signal == -1).fillna(False).values.astype(bool)
    ex_long  = (exit_signal  == -1).fillna(False).values.astype(bool)
    ex_short = (exit_signal  ==  1).fillna(False).values.astype(bool)

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

    return pd.DataFrame(pos_array, index=entry_signal.index, columns=entry_signal.columns)
