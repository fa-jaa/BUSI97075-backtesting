import numpy as np
import pandas as pd


def performance_stats(
    returns_series: pd.Series,
    label:     str = '',
    factor:    pd.Series = None,
    weights:   pd.DataFrame = None,
    positions: pd.DataFrame = None,
) -> pd.Series:
    """
    Annualised performance metrics for a daily returns series.

    Parameters
    ----------
    returns_series : daily portfolio returns.
    label          : row label in the output DataFrame.
    factor         : benchmark series; when provided, adds Alpha, Beta, R².
    weights        : daily portfolio weights; when provided, adds Ann. Turnover %.
                     Turnover = annualised one-way weight change: sum(|Δw|)/2 × 252.
    positions      : daily position matrix (+1/0/-1); when provided, adds
                     Num Trades and Trades/Year. A trade is counted each time
                     any asset transitions from flat (0) to long or short.
    """
    r = returns_series.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + r).cumprod()
    mdd     = ((cum / cum.cummax()) - 1).min()

    out = {
        'Ann. Return %' : round(ann_ret * 100, 2),
        'Ann. Vol %'    : round(ann_vol * 100, 2),
        'Sharpe'        : round(sharpe, 3),
        'Max DD %'      : round(mdd * 100, 2),
        'Skewness'      : round(r.skew(), 3),
        'Hit Rate'      : round((r > 0).mean(), 3),
    }

    if positions is not None:
        pos = positions.reindex(r.index)
        entries = ((pos != 0) & (pos.shift(1) == 0)).sum().sum()
        n_years = len(r) / 252
        out['Num Trades']    = int(entries)
        out['Trades / Year'] = round(entries / n_years, 1)

    if weights is not None:
        # One-way turnover: half the sum of absolute weight changes, annualised.
        daily_to = weights.diff().abs().sum(axis=1) / 2
        out['Ann. Turnover %'] = round(daily_to.mean() * 252 * 100, 1)

    if factor is not None:
        from scipy import stats as scipy_stats

        common = pd.concat([r, factor.reindex(r.index)], axis=1).dropna()
        common.columns = ['strat', 'factor']
        beta, alpha, r_val, _, _ = scipy_stats.linregress(
            common['factor'], common['strat']
        )
        out['Alpha (ann) %'] = round(alpha * 252 * 100, 2)
        out['Beta']          = round(beta, 3)
        out['R²']            = round(r_val ** 2, 3)

    return pd.Series(out, name=label)


def sector_contribution(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    assets:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Daily return contribution broken down by sector.

    Returns a DataFrame with one column per sector; summing across columns
    equals the total portfolio return on each day.
    """
    daily_contrib = weights * returns
    sector_map    = assets.set_index('Commodity')['Sector']

    result = {}
    for sector in sorted(sector_map.unique()):
        cols = sector_map[sector_map == sector].index.intersection(daily_contrib.columns)
        result[sector] = daily_contrib[cols].sum(axis=1)

    return pd.DataFrame(result)


def position_summary(positions: pd.DataFrame) -> pd.DataFrame:
    """Days and percentage of time spent long, short, and flat per commodity."""
    total = len(positions)
    return pd.DataFrame({
        'Days Long'  : (positions == 1).sum(),
        'Days Short' : (positions == -1).sum(),
        'Days Flat'  : (positions == 0).sum(),
        '% Long'     : ((positions == 1).sum()  / total * 100).round(1),
        '% Short'    : ((positions == -1).sum() / total * 100).round(1),
    })


def plot_signals(
    prices: pd.DataFrame,
    configs: dict,
    commodity: str,
    figsize: tuple = (16, 6),
) -> None:
    """
    Signal chart for a single commodity under one or more strategy configurations.

    For each config, produces two full-size figures (Long and Short separately).
    Price is shown as a black line. Entry markers (▲ long, ▽ short) and exit
    markers (×) are overlaid; position-held periods are lightly shaded.

    Parameters
    ----------
    prices    : daily close prices (dates × assets).
    configs   : {label: kwargs} passed to build_signals, e.g. {'Trend': {}}.
    commodity : column name to plot, e.g. 'BRENT CRUDE'.
    figsize   : (width, height) for each individual figure.
    """
    import matplotlib.pyplot as plt
    from strategy import build_signals

    p = prices[commodity]

    for label, kwargs in configs.items():
        entry_l, entry_s, _, _, positions = build_signals(prices, **kwargs)

        pos = positions[commodity]
        el  = entry_l[commodity]
        es  = entry_s[commodity]

        long_exited  = (pos == 0) & (pos.shift(1) == 1)
        short_exited = (pos == 0) & (pos.shift(1) == -1)

        for shade, entries, exits, colour, entry_marker, side, entry_label, exit_label in [
            (pos == 1,  el, long_exited,  'green',     '^', 'Long',  'Long entry',  'Long exit'),
            (pos == -1, es, short_exited, 'firebrick', 'v', 'Short', 'Short entry', 'Short exit'),
        ]:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(f'{commodity}  —  {label}  —  {side}',
                         fontweight='bold', fontsize=13)

            ax.plot(p.index, p.values, color='black', lw=1.0, zorder=3)

            ax.fill_between(p.index, 0, 1,
                            where=shade,
                            transform=ax.get_xaxis_transform(),
                            color=colour, alpha=0.12, lw=0)

            ax.scatter(p[entries].index, p[entries].values,
                       marker=entry_marker, color=colour, s=70,
                       zorder=5, label=entry_label)
            ax.scatter(p[exits].index, p[exits].values,
                       marker='x', color=colour, s=55, lw=2,
                       zorder=5, label=exit_label)

            ax.legend(fontsize=9, ncol=2, loc='upper left')
            ax.set_ylabel('Price')
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            plt.show()


def plot_comparison(
    results: dict,
    factor: pd.Series = None,
    figsize: tuple = (13, 5),
) -> None:
    """
    Cumulative return chart for multiple strategy configurations.

    Parameters
    ----------
    results : {label: portfolio_returns_series}
    factor  : plotted as a dashed black benchmark if provided.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for label, port_r in results.items():
        cum = (1 + port_r.dropna()).cumprod()
        ax.plot(cum.index, cum.values, lw=1.8, label=label)

    if factor is not None:
        cum_f = (1 + factor.dropna()).cumprod()
        ax.plot(cum_f.index, cum_f.values, color='black', lw=1.2,
                linestyle='--', label=factor.name or 'Factor')

    ax.axhline(1, color='grey', lw=0.5)
    ax.set_yscale('log')
    ax.set_title('Cumulative Return — Strategy Comparison', fontweight='bold')
    ax.set_ylabel('Cumulative Return (log scale)')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    plt.show()


def compare_strategies(
    results: dict,
    factor: pd.Series = None,
    weights_dict: dict = None,
) -> pd.DataFrame:
    """
    Compare performance metrics across multiple strategy configurations.

    Parameters
    ----------
    results      : {label: portfolio_returns_series}
    factor       : benchmark for alpha/beta calculation.
    weights_dict : {label: weights_DataFrame} — same keys as results.

    Returns
    -------
    pd.DataFrame with one row per configuration.
    """
    rows = [
        performance_stats(
            port_r, label=label, factor=factor,
            weights=weights_dict.get(label) if weights_dict else None,
        )
        for label, port_r in results.items()
    ]
    return pd.concat(rows, axis=1).T
