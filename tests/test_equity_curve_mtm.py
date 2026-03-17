from __future__ import annotations

import pandas as pd

from analyzer import build_equity_curve
from models import AnalysisParams


def make_params() -> AnalysisParams:
    from models import PartialExitRule

    return AnalysisParams(
        data_source_type="sqlite",
        db_path="/tmp/a.db",
        table_name=None,
        column_overrides={},
        excel_sheet_name=None,
        start_date="2024-01-01",
        end_date="2024-01-05",
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
        partial_exit_rules=(PartialExitRule(False, 0, "fixed_tp", 1, target_profit_pct=1.0),),
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        time_exit_mode="strict",
    )


def test_equity_curve_marks_to_market_during_holding():
    all_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "close": [100.0, 110.0, 120.0],
        }
    )
    strategy_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "sell_date": [pd.Timestamp("2024-01-03")],
            "stock_code": ["000001.SZ"],
            "buy_price": [100.0],
            "nav_before_trade": [1.0],
            "nav_after_trade": [1.2],
            "trade_no": [1],
            "exit_type": ["take_profit"],
        }
    )

    eq = build_equity_curve(all_data, strategy_df, make_params())
    day2 = eq.loc[eq["date"] == pd.Timestamp("2024-01-02"), "net_value"].iloc[0]
    day3 = eq.loc[eq["date"] == pd.Timestamp("2024-01-03"), "net_value"].iloc[0]

    assert day2 > 1.0
    assert day2 < day3
    assert abs(day3 - 1.2) < 1e-12
