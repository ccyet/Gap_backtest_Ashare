from __future__ import annotations

from typing import Any

import pandas as pd

from models import AnalysisParams
from rules import apply_gap_filters, simulate_trade


def make_params(**overrides: Any) -> AnalysisParams:
    base: dict[str, Any] = {
        "data_source_type": "local_parquet",
        "db_path": "",
        "table_name": None,
        "column_overrides": {},
        "excel_sheet_name": None,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "stock_codes": (),
        "gap_direction": "up",
        "entry_factor": "gap",
        "gap_entry_mode": "strict_break",
        "gap_pct": 2.0,
        "max_gap_filter_pct": 9.9,
        "trend_breakout_lookback": 2,
        "vcb_range_lookback": 2,
        "vcb_breakout_lookback": 2,
        "use_ma_filter": False,
        "fast_ma_period": 5,
        "slow_ma_period": 20,
        "time_stop_days": 1,
        "time_stop_target_pct": -100.0,
        "stop_loss_pct": 50.0,
        "take_profit_pct": 5.0,
        "enable_take_profit": False,
        "enable_profit_drawdown_exit": False,
        "profit_drawdown_pct": 40.0,
        "enable_ma_exit": False,
        "exit_ma_period": 10,
        "ma_exit_batches": 2,
        "partial_exit_enabled": False,
        "partial_exit_count": 2,
        "partial_exit_rules": (),
        "buy_cost_pct": 0.0,
        "sell_cost_pct": 0.0,
        "buy_slippage_pct": 0.0,
        "sell_slippage_pct": 0.0,
        "time_exit_mode": "force_close",
    }
    base.update(overrides)
    return AnalysisParams(**base)


def make_df(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(rows), freq="D"),
            "stock_code": ["000001.SZ"] * len(rows),
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": [row[4] for row in rows],
        }
    )


def test_entry_factor_default_is_gap() -> None:
    assert getattr(make_params(), "entry_factor", None) == "gap"


def test_explicit_gap_entry_factor_matches_default_gap_behavior() -> None:
    df = make_df([(100, 106, 99, 100, 1000), (103, 104, 102, 103, 1000)])
    default_result = apply_gap_filters(df, make_params())
    explicit_gap_result = apply_gap_filters(df, make_params(entry_factor="gap"))

    assert bool(default_result.loc[1, "is_signal"]) == bool(
        explicit_gap_result.loc[1, "is_signal"]
    )


def test_trend_breakout_long_trigger_uses_only_prior_bars() -> None:
    df = make_df(
        [
            (9.5, 10.0, 9.0, 9.8, 1000),
            (10.5, 11.0, 10.0, 10.8, 1000),
            (11.5, 50.0, 11.0, 12.0, 1000),
            (12.0, 12.5, 11.5, 12.2, 1000),
        ]
    )
    result = apply_gap_filters(
        df,
        make_params(entry_factor="trend_breakout", trend_breakout_lookback=2),
    )

    assert float(result.loc[2, "entry_trigger_price"]) == 11.0
    assert bool(result.loc[2, "is_signal"])


def test_trend_breakout_short_trigger_uses_only_prior_bars() -> None:
    df = make_df(
        [
            (10.0, 11.0, 9.0, 10.0, 1000),
            (9.5, 10.0, 8.0, 9.0, 1000),
            (7.9, 8.5, 1.0, 7.8, 1000),
            (7.8, 8.0, 7.0, 7.2, 1000),
        ]
    )
    result = apply_gap_filters(
        df,
        make_params(
            gap_direction="down",
            entry_factor="trend_breakout",
            trend_breakout_lookback=2,
        ),
    )

    assert float(result.loc[2, "entry_trigger_price"]) == 8.0
    assert bool(result.loc[2, "is_signal"])


def test_vcb_breakout_requires_contraction_gate() -> None:
    df = make_df(
        [
            (100.0, 105.0, 100.0, 104.0, 1000),
            (104.0, 111.0, 104.0, 110.0, 1000),
            (112.0, 113.0, 109.0, 112.0, 1000),
            (114.0, 116.0, 113.0, 115.0, 1000),
            (115.0, 116.0, 114.0, 115.0, 1000),
        ]
    )
    result = apply_gap_filters(
        df,
        make_params(
            entry_factor="volatility_contraction_breakout",
            vcb_range_lookback=2,
            vcb_breakout_lookback=2,
        ),
    )

    assert not bool(result.loc[2, "is_signal"])
    assert bool(result.loc[3, "is_signal"])
    assert not bool(result.loc[2, "is_contraction"])
    assert bool(result.loc[3, "is_contraction"])


def test_trend_breakout_long_open_fill_uses_open_price() -> None:
    df = make_df(
        [
            (98.0, 100.0, 95.0, 99.0, 1000),
            (105.0, 106.0, 104.0, 105.0, 1000),
            (104.0, 104.0, 103.0, 103.0, 1000),
        ]
    )
    enriched = apply_gap_filters(
        df,
        make_params(entry_factor="trend_breakout", trend_breakout_lookback=1),
    )
    trade, reason = simulate_trade(
        enriched, 1, make_params(entry_factor="trend_breakout")
    )

    assert reason is None
    assert trade is not None
    assert trade["entry_factor"] == "trend_breakout"
    assert float(trade["entry_trigger_price"]) == 100.0
    assert trade["entry_fill_type"] == "open"
    assert float(trade["buy_price"]) == 105.0
    assert int(trade["holding_days"]) >= 1


def test_trend_breakout_long_trigger_fill_uses_trigger_price() -> None:
    df = make_df(
        [
            (98.0, 100.0, 95.0, 99.0, 1000),
            (99.0, 101.0, 98.0, 100.0, 1000),
            (100.0, 101.0, 99.0, 100.0, 1000),
        ]
    )
    enriched = apply_gap_filters(
        df,
        make_params(entry_factor="trend_breakout", trend_breakout_lookback=1),
    )
    trade, reason = simulate_trade(
        enriched, 1, make_params(entry_factor="trend_breakout")
    )

    assert reason is None
    assert trade is not None
    assert trade["entry_factor"] == "trend_breakout"
    assert float(trade["entry_trigger_price"]) == 100.0
    assert trade["entry_fill_type"] == "trigger"
    assert float(trade["buy_price"]) == 100.0


def test_trend_breakout_long_no_fill_is_skipped() -> None:
    df = make_df(
        [
            (98.0, 100.0, 95.0, 99.0, 1000),
            (99.0, 99.5, 98.0, 99.0, 1000),
            (99.0, 99.5, 98.0, 99.0, 1000),
        ]
    )
    enriched = apply_gap_filters(
        df,
        make_params(entry_factor="trend_breakout", trend_breakout_lookback=1),
    )
    trade, reason = simulate_trade(
        enriched, 1, make_params(entry_factor="trend_breakout")
    )

    assert trade is None
    assert reason == "entry_not_filled"


def test_trend_breakout_short_open_fill_uses_open_price() -> None:
    df = make_df(
        [
            (100.0, 101.0, 95.0, 99.0, 1000),
            (94.0, 95.0, 93.0, 94.0, 1000),
            (95.0, 96.0, 94.0, 95.0, 1000),
        ]
    )
    params = make_params(
        gap_direction="down", entry_factor="trend_breakout", trend_breakout_lookback=1
    )
    enriched = apply_gap_filters(df, params)
    trade, reason = simulate_trade(enriched, 1, params, direction="short")

    assert reason is None
    assert trade is not None
    assert trade["entry_factor"] == "trend_breakout"
    assert float(trade["entry_trigger_price"]) == 95.0
    assert trade["entry_fill_type"] == "open"
    assert float(trade["buy_price"]) == 94.0


def test_trend_breakout_short_trigger_fill_uses_trigger_price() -> None:
    df = make_df(
        [
            (100.0, 101.0, 95.0, 99.0, 1000),
            (97.0, 98.0, 94.0, 96.0, 1000),
            (96.0, 97.0, 95.0, 96.0, 1000),
        ]
    )
    params = make_params(
        gap_direction="down", entry_factor="trend_breakout", trend_breakout_lookback=1
    )
    enriched = apply_gap_filters(df, params)
    trade, reason = simulate_trade(enriched, 1, params, direction="short")

    assert reason is None
    assert trade is not None
    assert trade["entry_factor"] == "trend_breakout"
    assert float(trade["entry_trigger_price"]) == 95.0
    assert trade["entry_fill_type"] == "trigger"
    assert float(trade["buy_price"]) == 95.0


def test_trend_breakout_locked_bar_is_unfillable() -> None:
    df = make_df(
        [
            (98.0, 100.0, 95.0, 99.0, 1000),
            (100.0, 100.0, 100.0, 100.0, 1000),
            (100.0, 101.0, 99.0, 100.0, 1000),
        ]
    )
    enriched = apply_gap_filters(
        df,
        make_params(entry_factor="trend_breakout", trend_breakout_lookback=1),
    )
    trade, reason = simulate_trade(
        enriched, 1, make_params(entry_factor="trend_breakout")
    )

    assert trade is None
    assert reason == "locked_bar_unfillable"


def test_trend_breakout_zero_volume_is_unfillable() -> None:
    df = make_df(
        [
            (98.0, 100.0, 95.0, 99.0, 1000),
            (105.0, 106.0, 104.0, 105.0, 0.0),
            (104.0, 104.0, 103.0, 103.0, 1000),
        ]
    )
    enriched = apply_gap_filters(
        df,
        make_params(entry_factor="trend_breakout", trend_breakout_lookback=1),
    )
    trade, reason = simulate_trade(
        enriched, 1, make_params(entry_factor="trend_breakout")
    )

    assert trade is None
    assert reason == "locked_bar_unfillable"
