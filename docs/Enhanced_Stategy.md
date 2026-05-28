# Task 2 Enhanced Strategy

## Purpose And Strategy Identity

Task 2 is an enhanced commodity trend-following strategy. It keeps the Task 1
identity: find a directional trend, wait for an entry-timing confirmation, hold
one position per commodity, and size positions from volatility-based risk.

Task 2 adds five enhancements:

- Multi-lookback Layer 1 conviction instead of a single slope lookback.
- Optional macro modulation that changes position size, not trade direction.
- A stricter short RSI setup with an execution-time oversold cap.
- A close-price Parabolic SAR-style trailing stop with ATR-based initial
  placement.
- Transaction-cost-aware returns and training diagnostics.

Final holdout/test evaluation is not part of tuning. The strategy and selected
parameters must be frozen before the final test-set run.

## File Map

This document explains the enhanced strategy implemented in
`strategy_enhanced.py`.

- Signal stack: `signals/`
- Macro sizing overlay: `signals/macro_signal.py`
- Position management: `portfolio/slot_manager_enhanced.py`
- Sizing and transaction costs: `portfolio/portfolio_manager_enhanced.py`
- Training-only research diagnostics: `diagnostics.py`
- Notebook control panel and training analysis: `main.ipynb`

## End-To-End Pipeline

The strategy starts from daily commodity close prices. Returns are either passed
in or computed from prices. Asset metadata is used only for analysis tables and
sector contribution. Optional macro inputs are used to scale conviction.

The execution flow is:

1. Build Layer 1:
   - Binary regime direction: `+1`, `0`, `-1`.
   - Continuous `regime_strength`: conviction in `[-1, +1]`.
2. Build Layer 2:
   - Momentary entry events inside the active Layer 1 regime.
   - Signal is observed at close of day `t`.
3. Shift entries by one bar:
   - Layer 2 signal at `t` can execute at `t+1`.
   - The short RSI cap is checked on the actual execution bar.
4. Build positions:
   - At most one active position per commodity.
   - Exits come from SAR, Bollinger Band profit exits, or regime mismatch.
5. Build portfolio:
   - Position size is scaled by conviction.
   - Weights are shifted by `exec_lag` before P&L.
   - Transaction costs are subtracted from gross returns.

Primary outputs are:

- `positions`: one column per commodity, values `-1`, `0`, or `+1`.
- `weights`: executed portfolio weights after sizing and lag.
- `gross_returns`: strategy returns before transaction costs.
- `portfolio_returns`: net returns after transaction costs.
- Diagnostics: cost sensitivity, SAR grid results, contribution and trade counts.

## Layer 1: Trend Regime And Conviction

Layer 1 decides whether a commodity is in a long regime, short regime, or no
trade regime.

Task 1 used a single slow-SMA slope lookback. Task 2 instead measures the slope
of the slow SMA across several lookbacks and aggregates the signs into a
continuous conviction score.

Default Layer 1 settings:

- `filter_fast = 50`
- `filter_slow = 200`
- `lookbacks = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]`

The enhanced Layer 1 has two outputs:

- Binary direction:
  - `+1`: long regime.
  - `0`: neutral or mixed regime.
  - `-1`: short regime.
- Continuous `regime_strength`:
  - `+1.0` or `-1.0`: all lookbacks agree.
  - Around `+0.6` or `-0.6`: moderate agreement.
  - Around `+0.2` or `-0.2`: weak agreement.
  - `0`: no clear directional conviction.

The binary direction feeds Layer 2 and the slot manager. The continuous
`regime_strength` feeds the portfolio manager, where it scales position size.

Macro inputs can adjust `regime_strength`, but macro does not flip the trade
direction. Commodity trend direction remains the source of long versus short.

## Layer 2: Entry Timing And RSI Logic

Layer 2 controls when the strategy may open a position inside the Layer 1
regime. It is an entry-timing layer, not a position-sizing layer and not an exit
layer.

Default Layer 2 settings:

- `ema_window = 14`
- `rsi_window = 14`
- `rsi_long_level = 50.0`
- `rsi_level = 60.0`
- `rsi_flag_window = 5`
- `rsi_short_cap = 50.0`

Long entry condition at signal bar `t`:

- Layer 1 is `+1`.
- Price crosses above EMA.
- RSI recently crossed above `rsi_long_level`.
- The RSI long flag is active.

Short entry condition at signal bar `t`:

- Layer 1 is `-1`.
- Price crosses below EMA.
- RSI recently crossed below `rsi_level = 60`.
- The RSI short flag is active for up to `5` bars.

Layer 2 does not apply the short oversold cap. It only says that a short entry
setup exists. The cap is applied later in `slot_manager_enhanced.py`, after the
one-bar execution shift.

## Short RSI Execution Cap

The short RSI cap is checked only on the execution bar.

- Layer 2 signal appears at close `t`.
- Slot manager shifts the entry to `t+1`.
- On `t+1`, the strategy checks execution-bar RSI.
- If execution-bar RSI is below `rsi_short_cap`, the short is skipped.

Current cap:

- `rsi_short_cap = 50.0`
- A short can execute only if execution-bar RSI is at least `50`.

This is important because checking the cap at signal time can be stale. A short
setup may look acceptable at `t`, but by `t+1` the move may already be too
stretched. The execution-time cap prevents over-trading shorts after RSI has
fallen too far.

## Position Management

The slot manager converts entry events into actual positions. It allows only one
active position per commodity, so repeated signals do not stack into clustered
trades.

Entries require:

- Flat commodity position.
- Shifted Layer 2 signal.
- Matching Layer 1 regime on the execution bar.
- Valid ATR value.
- For shorts, execution-bar RSI must pass the short cap.

Exits happen on the first applicable condition:

- SAR stop hit.
- Bollinger Band profit exit.
- Layer 1 regime no longer matches the open position.
- Missing price data closes the position defensively.

## Parabolic SAR-Style Exit

Task 2 uses a close-price Parabolic SAR-style trailing stop with ATR-based
initial placement. It is deliberately not the high/low version often described
in Wilder's original rules: the code uses close prices, not intraday highs/lows,
and it does not reverse positions automatically when SAR is hit.

In this strategy, SAR is an exit mechanism only. It closes an existing position.
New positions still require Layer 1 and Layer 2 entry logic.

### SAR Parameters

- `atr_window`: lookback used to compute `ATR_pct`.
- `sar_initial_mult`: initial SAR placement distance in `ATR_pct` units.
- `sar_af_start`: starting Parabolic SAR acceleration factor.
- `sar_af_step`: amount added to AF after new favorable extremes.
- `sar_af_max`: maximum allowed AF.
- `sar_grace_period`: number of bars before AF is allowed to increase.

### Initial SAR Placement

At entry, the stop starts a volatility-scaled distance away from the entry
price.

Long initial SAR:

```text
initial_sar = entry_price * (1 - sar_initial_mult * ATR_pct)
```

Short initial SAR:

```text
initial_sar = entry_price * (1 + sar_initial_mult * ATR_pct)
```

Example with entry price `100`, `ATR_pct = 0.02`, and
`sar_initial_mult = 2.5`:

- Long initial SAR: `100 * (1 - 2.5 * 0.02) = 95`
- Short initial SAR: `100 * (1 + 2.5 * 0.02) = 105`

### Daily SAR Update

Each day while the position is open:

```text
SAR = SAR + AF * (EP - SAR)
```

`EP` is the favorable extreme price:

- Long: highest close since entry.
- Short: lowest close since entry.

`AF` starts at `sar_af_start`. After the trade has lived longer than
`sar_grace_period`, AF increases by `sar_af_step` whenever a new favorable
extreme is made. AF cannot exceed `sar_af_max`.

Exit rules:

- Long exits when close `< SAR`.
- Short exits when close `> SAR`.
- SAR does not reverse the position.

## Code Defaults Versus Notebook-Selected Research Settings

`strategy_enhanced.py` keeps conservative function defaults:

- `sar_initial_mult = 3.0`
- `sar_af_start = 0.02`
- `sar_af_step = 0.02`
- `sar_af_max = 0.20`
- `sar_grace_period = 10`

The current Task 2 notebook control panel uses the training-selected SAR
candidate from the predefined SAR grid:

- `ENH_ATR_WINDOW = 50`
- `ENH_SAR_INITIAL_MULT = 2.5`
- `ENH_SAR_AF_START = 0.01`
- `ENH_SAR_AF_STEP = 0.03`
- `ENH_SAR_AF_MAX = 0.20`
- `ENH_SAR_GRACE_PERIOD = 5`

These settings were selected using training-only diagnostics. They should be
frozen before any final holdout/test evaluation.

## Portfolio Construction

The portfolio manager converts positions into weights. The sizing logic uses
ATR risk and conviction.

```text
stop_distance_pct = sl_mult * ATR_pct
effective_risk    = risk_per_trade * abs(regime_strength)
weight            = min(effective_risk / stop_distance_pct, max_weight) * direction
```

Current portfolio settings:

- `risk_per_trade = 0.01`
- `sl_mult = 3.0`
- Notebook control: `ENH_SL_MULT = 3.0`
- `max_weight = 0.20`
- `leverage_cap = 2.50`
- `exec_lag = 1`

Important distinction:

- `sl_mult` is a sizing assumption.
- `sar_initial_mult` is the actual initial SAR placement distance.
- `sar_af_*` controls SAR acceleration.

Because Task 2 exits with SAR, `sl_mult` is not the actual stop used by the slot
manager. It is the portfolio manager's estimate of stop distance for sizing the
trade. This keeps risk sizing stable while the actual SAR stop can trail over
time.

Lower `sl_mult` means a smaller assumed stop distance, so the same risk budget
translates into a larger position. Higher `sl_mult` means a wider assumed stop
distance, so the same risk budget translates into a smaller position.

Weights are locked when a position opens and remain fixed until the position
closes. A new position is skipped if adding it would breach the gross leverage
cap. Existing positions are not force-closed by the leverage cap.

`exec_lag = 1` shifts weights before P&L is calculated, so returns are earned
after the modeled execution delay.

## SL Multiplier Sizing Grid

The `sl_mult` grid is a training-only sizing diagnostic. It does not change
entries, SAR exits, RSI logic, macro settings, or transaction-cost logic. It
only changes the portfolio manager's sizing assumption:

```text
stop_distance_pct = sl_mult * ATR_pct
```

The predefined grid is:

- `sl_mult`: `1.5`, `2.0`, `2.5`, `3.0`, `3.5`, `4.0`, `4.5`, `5.0`

The current default remains:

- `ENH_SL_MULT = 3.0`

Decision rule:

- Do not choose the top training Sharpe automatically.
- Prefer a new `sl_mult` only if it improves net Sharpe after costs while
  preserving drawdown, trade count, short-entry count, turnover, cost drag, and
  long/short contribution balance.
- If the improvement is small or unstable, keep `ENH_SL_MULT = 3.0`.

The SL multiplier must be frozen before any final holdout/test evaluation.

## Macro Modulation

Macro signals adjust conviction, not direction.

Inputs can include:

- FX spot data for the dollar signal.
- Equity index data for risk-on/risk-off.
- Credit spread data for credit conditions.
- FX implied volatility for high-volatility dampening.

Macro settings:

- `macro_alpha = 0.5`
- `macro_fast = 50`
- `macro_slow = 200`
- `vol_cap_pct = 0.80`
- `vol_dampen = 0.5`

Macro agreement can increase conviction. Macro disagreement can dampen
conviction. High FX volatility can reduce conviction. None of these macro
signals changes a commodity from long to short or short to long.

## Transaction Costs

Task 2 reports net returns after transaction costs by default.

The portfolio manager computes one-way turnover from executed weights:

```text
one_way_turnover = sum(abs(delta executed weights)) / 2
daily_cost       = one_way_turnover * tx_cost_bps / 10_000
net_return       = gross_return - daily_cost
```

Current default:

- `tx_cost_bps = 2.0`
- Notebook control: `ENH_TX_COST_BPS = 2.0`

Gross returns can still be reproduced with:

```text
tx_cost_bps = 0.0
```

The notebook checks transaction-cost sensitivity at:

- `0` bps
- `2` bps
- `5` bps
- `10` bps

Net Sharpe should decline as transaction costs increase.

## SAR Optimization And Anti-Overfitting Policy

SAR optimization is a training-only robustness diagnostic. It is not trading
logic and it should not touch final holdout/test data.

The predefined grid is intentionally narrow:

- `sar_initial_mult`: `2.0`, `2.5`, `3.0`, `3.5`, `4.0`
- `sar_af_start`: `0.01`, `0.02`, `0.03`
- `sar_af_step`: `0.01`, `0.02`, `0.03`
- `sar_af_max`: `0.10`, `0.15`, `0.20`, `0.25`
- `sar_grace_period`: `5`, `10`, `15`

This grid has `540` combinations.

The selected candidate should not be the top training Sharpe by default. It
should be preferred only if it is robust across:

- Net Sharpe after costs.
- Max drawdown.
- Total trade count.
- Short-entry count.
- Turnover and cost drag.
- Nearby parameter values.
- Balance between long and short contribution.

The diagnostics flag candidates that cut trade count below `70%` of the current
SAR baseline or worsen max drawdown by more than `5` percentage points.

Current training-selected SAR candidate:

- `sar_initial_mult = 2.5`
- `sar_af_start = 0.01`
- `sar_af_step = 0.03`
- `sar_af_max = 0.20`
- `sar_grace_period = 5`

These parameters must be locked before the final test-set evaluation. If the
grid is expanded after seeing results, or if test results are used to revise
parameters, that becomes overfitting risk.

## Validation Checklist

Before accepting a Task 2 research result:

- Confirm `tx_cost_bps = 0.0` reproduces the gross baseline.
- Confirm net Sharpe declines as transaction costs increase.
- Confirm no executed short entry has execution-bar RSI below `50`.
- Confirm long and short contribution separately.
- Confirm SAR settings come from the predefined training grid.
- Confirm selected SAR settings are robust, not just the best Sharpe row.
- Confirm the final holdout/test set has not been used during tuning.
- Confirm Task 1 remains unchanged.

## Common Confusions

`sar_initial_mult` is not a Parabolic SAR acceleration factor. It controls where
the initial SAR stop is placed, measured in `ATR_pct` units.

`sar_af_start`, `sar_af_step`, and `sar_af_max` are the Parabolic SAR
acceleration-factor settings.

`sl_mult` is not the actual SAR stop. It is only the portfolio sizing
stop-distance assumption.

SAR is an exit mechanism. It does not create Layer 2 entries, and it does not
reverse positions.

Macro affects conviction and sizing. It does not change the long/short
direction.

The short RSI cap is execution-time only. Layer 2 creates the short setup; the
slot manager decides whether the short is still acceptable when it executes.
