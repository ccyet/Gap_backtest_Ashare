from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_local_parquet_data
from models import AnalysisParams, validate_params


def test_validate_params_local_parquet_and_adjust():
    params = AnalysisParams(
        data_source_type="local_parquet",
        db_path="",
        table_name=None,
        column_overrides={},
        excel_sheet_name=None,
        start_date="2024-01-01",
        end_date="2024-01-31",
        stock_codes=(),
        gap_direction="up",
        gap_pct=2.0,
        max_gap_filter_pct=9.9,
        use_ma_filter=False,
        fast_ma_period=5,
        slow_ma_period=20,
        time_stop_days=2,
        time_stop_target_pct=1.0,
        stop_loss_pct=5.0,
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
        time_exit_mode="strict",
        local_data_root="data/market/daily",
        adjust="qfq",
    )
    errors, _ = validate_params(params)
    assert not errors


def test_validate_params_adjust_invalid():
    params = AnalysisParams(
        data_source_type="local_parquet",
        db_path="",
        table_name=None,
        column_overrides={},
        excel_sheet_name=None,
        start_date="2024-01-01",
        end_date="2024-01-31",
        stock_codes=(),
        gap_direction="up",
        gap_pct=2.0,
        max_gap_filter_pct=9.9,
        use_ma_filter=False,
        fast_ma_period=5,
        slow_ma_period=20,
        time_stop_days=2,
        time_stop_target_pct=1.0,
        stop_loss_pct=5.0,
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
        time_exit_mode="strict",
        local_data_root="data/market/daily",
        adjust="bad",
    )
    errors, _ = validate_params(params)
    assert any("复权方式" in msg for msg in errors)


def test_local_parquet_loader_missing_dir(tmp_path: Path):
    try:
        load_local_parquet_data("2024-01-01", "2024-01-31", local_data_root=str(tmp_path / "missing"), adjust="qfq")
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_local_parquet_loader_reads_and_filters(tmp_path: Path):
    import pytest

    pytest.importorskip("pyarrow")
    root = tmp_path / "daily"
    qfq = root / "qfq"
    qfq.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.0, 2.0, 3.0],
            "volume": [10, 20, 30],
        }
    ).to_parquet(qfq / "000001.SZ.parquet", index=False)

    out = load_local_parquet_data(
        "2024-01-02",
        "2024-01-02",
        stock_codes=("000001.SZ",),
        local_data_root=str(root),
        adjust="qfq",
    )
    assert not out.empty
    assert out["stock_code"].nunique() == 1
    assert out["stock_code"].iloc[0] == "000001.SZ"
    assert out["date"].min() <= pd.Timestamp("2024-01-02") <= out["date"].max()
