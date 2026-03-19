from __future__ import annotations

from typing import Any

import pandas as pd

from models import AnalysisParams
from rules import apply_gap_filters


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
        "gap_entry_mode": "strict_break",
        "gap_pct": 2.0,
        "max_gap_filter_pct": 9.9,
        "use_ma_filter": False,
        "fast_ma_period": 5,
        "slow_ma_period": 20,
        "time_stop_days": 2,
        "time_stop_target_pct": 1.0,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 5.0,
        "enable_take_profit": True,
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
        "time_exit_mode": "strict",
    }
    supported_fields = set(AnalysisParams.__dataclass_fields__)
    optional_rollout_defaults = {
        "entry_factor": "gap",
        "trend_breakout_lookback": 2,
        "vcb_range_lookback": 2,
        "vcb_breakout_lookback": 2,
    }
    for field_name, field_value in optional_rollout_defaults.items():
        if field_name in supported_fields:
            base[field_name] = field_value
    base.update(overrides)
    return AnalysisParams(**base)


def make_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(rows), freq="D"),
            "stock_code": ["000001.SZ"] * len(rows),
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": [1000] * len(rows),
        }
    )


def test_long_non_strict_mode_can_pass_when_strict_break_fails() -> None:
    df = make_df([(100, 106, 99, 100), (103, 104, 102, 103)])
    strict_result = apply_gap_filters(df, make_params(gap_entry_mode="strict_break"))
    relaxed_result = apply_gap_filters(
        df, make_params(gap_entry_mode="open_vs_prev_close_threshold")
    )
    assert not bool(strict_result.loc[1, "is_signal"])
    assert bool(relaxed_result.loc[1, "is_signal"])


def test_short_non_strict_mode_can_pass_when_strict_break_fails() -> None:
    df = make_df([(100, 101, 94, 100), (97, 98, 96, 97)])
    strict_result = apply_gap_filters(
        df, make_params(gap_direction="down", gap_entry_mode="strict_break")
    )
    relaxed_result = apply_gap_filters(
        df,
        make_params(
            gap_direction="down", gap_entry_mode="open_vs_prev_close_threshold"
        ),
    )
    assert not bool(strict_result.loc[1, "is_signal"])
    assert bool(relaxed_result.loc[1, "is_signal"])


def test_non_strict_mode_still_respects_max_gap_filter() -> None:
    df = make_df([(100, 101, 99, 100), (112, 113, 111, 112)])
    result = apply_gap_filters(
        df,
        make_params(
            gap_entry_mode="open_vs_prev_close_threshold",
            gap_pct=2.0,
            max_gap_filter_pct=9.9,
        ),
    )
    assert not bool(result.loc[1, "is_signal"])


def test_default_gap_mode_is_strict_break() -> None:
    params = make_params()
    assert params.gap_entry_mode == "strict_break"
