# Task 2 Enhanced Strategy

## File Map

This document explains the enhanced strategy implemented in `strategy_enhanced.py`.
The signal stack lives in `signals/`, position management lives in
`portfolio/slot_manager_enhanced.py`, sizing and costs live in
`portfolio/portfolio_manager_enhanced.py`, and training-only research checks live
in `diagnostics.py`.

## Objective

Task 2 is an enhanced commodity trend-following strategy. It keeps the Task 1 trend-following identity, but adds conviction-weighted sizing, macro regime modulation, a stricter short RSI setup, Parabolic SAR exits, and transaction-cost-aware analysis.

The current research default uses net returns with `2.0` bps of cost per one-way turnover. Gross returns can still be reproduced with `tx_cost_bps = 0.0`.

## Signal Stack

### Layer 1: Trend Regime

Layer 1 defines the directional regime for each commodity.

- A fast and slow SMA pair defines the broad trend direction.
- Multiple slope lookbacks are aggregated into a continuous conviction score.
- The binary regime is still one of `+1`, `0`, or `-1`.
- The continuous regime strength is used later for position sizing.

The default Layer 1 settings are:

- `filter_fast = 50`
- `filter_slow = 200`
- lookbacks from `20` to `110` days

### Layer 2: Entry Timing

Layer 2 controls when a position may be opened inside the Layer 1 regime.

- Long entries require price/EMA confirmation and RSI strength.
- Short entries require price/EMA confirmation and a Task 1-style RSI break below `60`.
- The short RSI break flag stays active for `5` bars.
- Layer 2 does not apply the oversold short cap.

The default RSI settings are:

- `rsi_long_level = 50.0`
- `rsi_level = 60.0`
- `rsi_flag_window = 5`

### Short RSI Execution Cap

The short oversold cap is checked only when the short entry executes.

- Default: `rsi_short_cap = 50.0`
- If the execution-bar RSI is below `50`, the short entry is skipped.
- This prevents over-trading shorts after the move is already too stretched.

This is intentionally an execution filter, not a Layer 2 signal filter.

## Position Management

### Parabolic SAR Exit

Task 2 replaces the fixed ATR stop with a Parabolic SAR trailing stop.

- Initial SAR distance is set by `sar_initial_mult * ATR_pct`.
- The default initial distance is `3.0 * ATR_pct`.
- SAR then moves toward the trade's extreme price using the acceleration factor.
- The acceleration factor starts at `0.02`, steps by `0.02`, and is capped at `0.20`.
- The acceleration factor only increases after the grace period.

Current defaults:

- `sar_initial_mult = 3.0`
- `sar_af_start = 0.02`
- `sar_af_step = 0.02`
- `sar_af_max = 0.20`
- `sar_grace_period = 10`

Important scale note:

- `3.0` is an ATR stop-distance multiplier.
- `0.02` and `0.20` are SAR acceleration-factor values.
- These are different concepts and should not be compared as the same type of multiplier.

### Other Exits

Task 2 keeps the other Task 1-style exits:

- Exit when the Layer 1 regime no longer matches the open position.
- Exit longs above the upper Bollinger Band.
- Exit shorts below the lower Bollinger Band.
- Suppress clustering by allowing only one active position per commodity.

## Portfolio Construction

Position size is based on ATR risk and regime conviction.

```text
stop_distance_pct = sl_mult * ATR_pct
effective_risk    = risk_per_trade * abs(regime_strength)
weight            = min(effective_risk / stop_distance_pct, max_weight) * direction
```

Current defaults:

- `risk_per_trade = 0.01`
- `sl_mult = 3.0`
- `max_weight = 0.20`
- `leverage_cap = 2.50`
- `exec_lag = 1`

The execution lag shifts weights before P&L is calculated.

## Macro Modulation

Macro signals adjust conviction, not direction.

- Dollar, equity, and credit signals are blended into a macro regime score.
- Macro agreement can increase conviction.
- Macro disagreement can dampen conviction.
- High FX volatility can cap conviction.

Current defaults:

- `macro_alpha = 0.5`
- `macro_fast = 50`
- `macro_slow = 200`
- `vol_cap_pct = 0.80`
- `vol_dampen = 0.5`

## Transaction Costs

Task 2 now reports strategy returns net of transaction costs by default.

```text
one_way_turnover = sum(abs(delta executed weights)) / 2
daily_cost       = one_way_turnover * tx_cost_bps / 10_000
net_return       = gross_return - daily_cost
```

Current default:

- `tx_cost_bps = 2.0`

The notebook also checks sensitivity at `0`, `2`, `5`, and `10` bps.

## SAR Optimization Method

SAR optimization should be treated as a training-only robustness exercise, not a pure in-sample Sharpe maximization.

The diagnostic grid tests:

- `sar_initial_mult`: `2.0`, `2.5`, `3.0`, `3.5`, `4.0`
- `sar_af_start`: `0.01`, `0.02`, `0.03`
- `sar_af_step`: `0.01`, `0.02`, `0.03`
- `sar_af_max`: `0.10`, `0.15`, `0.20`, `0.25`
- `sar_grace_period`: `5`, `10`, `15`

A candidate should be preferred only if it:

- improves net Sharpe after costs,
- does not materially worsen max drawdown,
- does not collapse trade count,
- is stable across nearby SAR parameter values,
- does not rely entirely on one side of the book.

The diagnostics flag candidates that cut trade count below `70%` of the current SAR baseline or worsen max drawdown by more than `5` percentage points.

The selected SAR parameters must be frozen before any final holdout evaluation.

## Current Validation Checklist

Before accepting any new Task 2 result:

- Confirm `tx_cost_bps = 0.0` reproduces the current gross baseline.
- Confirm net Sharpe declines as costs increase.
- Confirm no executed short entry has execution-bar RSI below `50`.
- Confirm long and short contribution separately.
- Confirm selected SAR parameters are chosen from the predefined training grid with robustness checks.
- Freeze final parameters before any final holdout evaluation.
- Confirm Task 1 remains unchanged.
