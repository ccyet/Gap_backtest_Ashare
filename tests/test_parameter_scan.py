from __future__ import annotations

import pandas as pd

from analyzer import run_parameter_scan
from models import AnalysisParams, ParamScanAxis, ParamScanConfig, validate_params


def make_params(scan_config: ParamScanConfig) -> AnalysisParams:
    return AnalysisParams(
        data_source_type="local_parquet",
        db_path="",
        table_name=None,
        column_overrides={},
        excel_sheet_name=None,
        start_date="2024-01-01",
        end_date="2024-01-31",
        stock_codes=(),
        gap_direction="up",
        gap_entry_mode="strict_break",
        gap_pct=2.0,
        max_gap_filter_pct=9.9,
        use_ma_filter=False,
        fast_ma_period=5,
        slow_ma_period=20,
        time_stop_days=1,
        time_stop_target_pct=-10.0,
        stop_loss_pct=50.0,
        take_profit_pct=5.0,
        enable_take_profit=True,
        enable_profit_drawdown_exit=False,
        profit_drawdown_pct=40.0,
        enable_ma_exit=False,
        exit_ma_period=10,
        ma_exit_batches=2,
        partial_exit_enabled=False,
        partial_exit_count=2,
        partial_exit_rules=(),
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        buy_slippage_pct=0.0,
        sell_slippage_pct=0.0,
        time_exit_mode="strict",
        scan_config=scan_config,
    )


def make_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [100.0, 104.0, 109.0],
            "high": [101.0, 106.0, 110.0],
            "low": [99.0, 103.0, 108.0],
            "close": [100.0, 105.0, 109.0],
            "volume": [1000.0, 1200.0, 900.0],
        }
    )


def test_parameter_scan_ranks_best_combo_and_keeps_best_outputs() -> None:
    scan_config = ParamScanConfig(
        enabled=True,
        axes=(ParamScanAxis(field_name="gap_pct", values=(2.0, 5.0)),),
        metric="total_return_pct",
        max_combinations=25,
    )
    scan_df, detail_df, daily_df, equity_df, stats, best_overrides = run_parameter_scan(
        make_market_data(), make_params(scan_config)
    )
    assert len(scan_df) == 2
    assert int(scan_df.iloc[0]["rank"]) == 1
    assert float(scan_df.iloc[0]["gap_pct"]) == 2.0
    assert not detail_df.empty
    assert not daily_df.empty
    assert not equity_df.empty
    assert float(best_overrides["gap_pct"]) == 2.0
    assert float(stats["total_return_pct"]) > 0.0


def test_parameter_scan_validation_rejects_oversized_grid() -> None:
    params = make_params(
        ParamScanConfig(
            enabled=True,
            axes=(
                ParamScanAxis(field_name="gap_pct", values=(1.0, 2.0, 3.0, 4.0, 5.0)),
                ParamScanAxis(
                    field_name="stop_loss_pct", values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
                ),
            ),
            metric="total_return_pct",
            max_combinations=20,
        )
    )
    errors, _ = validate_params(params)
    assert any("组合数超出上限" in error for error in errors)
