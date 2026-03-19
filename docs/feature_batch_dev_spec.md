# Feature Batch Dev Spec

## Scope

This batch implements only:

1. Deterministic slippage modeling
2. Bounded parameter sensitivity scan
3. Optional non-strict gap entry mode
4. Partial-exit ordering review plus regression tests

## Preserved behavior

- Default gap mode remains `strict_break`
- Partial-exit ordering semantics remain unchanged
- Offline-first architecture remains unchanged
- Slippage remains separate from buy/sell costs
- Scan execution stays sequential and bounded

## Accepted behavior

### Slippage

- Long entry buy: `open * (1 + buy_slippage_ratio)`
- Long exit sell: `reference_exit_price * (1 - sell_slippage_ratio)`
- Short entry sell: `open * (1 - sell_slippage_ratio)`
- Short exit cover: `reference_exit_price * (1 + buy_slippage_ratio)`
- Slippage is applied after trigger determination and does not change signal/exit trigger conditions.

### Gap entry mode

- `strict_break`:
  - long: `open > prev_high * (1 + gap_ratio)`
  - short: `open < prev_low * (1 - gap_ratio)`
- `open_vs_prev_close_threshold`:
  - long: `gap_pct_vs_prev_close >= gap_pct`
  - short: `gap_pct_vs_prev_close <= -gap_pct`
- Both modes still respect `max_gap_filter_pct` and optional MA filter.

### Partial-exit ordering verdict

Current behavior is coherent and remains unchanged:

1. stop loss
2. partial exits by ascending priority
3. legacy whole-position exits only when partial exits are disabled
4. time exit
5. strict / force_close end-of-data handling

Same-day multiple partial exits are allowed and processed by ascending priority.

### Parameter scan

- Disabled by default
- Sequential only
- Cartesian combinations only
- Hard capped by `max_combinations`
- Full detail/daily/equity outputs retained only for the best combination
- Optional export sheet included only when scan is enabled

## Implementation seams

- `models.py`: new params, scan config, parsing, validation
- `rules.py`: gap mode branching, slippage application
- `analyzer.py`: sequential scan runner
- `app.py`: new controls and scan visualization
- `exporter.py`: optional scan sheet
- `tests/`: regression coverage for semantics, slippage, gap mode, and scan behavior

## Rollout order

1. Lock same-bar partial-exit semantics with tests
2. Add non-strict gap entry mode
3. Add deterministic slippage
4. Add scan backend and validations
5. Add optional scan export
6. Add Streamlit wiring and display
