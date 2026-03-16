import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from models import AnalysisParams, PartialExitRule
from rules import simulate_trade


def make_params(**kwargs):
    base = dict(
        data_source_type="file",
        db_path="",
        table_name=None,
        column_overrides={},
        excel_sheet_name=None,
        start_date="2024-01-01",
        end_date="2024-12-31",
        stock_codes=(),
        gap_direction="up",
        gap_pct=2.0,
        max_gap_filter_pct=9.9,
        use_ma_filter=False,
        fast_ma_period=5,
        slow_ma_period=20,
        time_stop_days=2,
        time_stop_target_pct=2.0,
        stop_loss_pct=5.0,
        take_profit_pct=5.0,
        enable_take_profit=False,
        enable_profit_drawdown_exit=False,
        profit_drawdown_pct=40.0,
        enable_ma_exit=False,
        exit_ma_period=3,
        ma_exit_batches=2,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        time_exit_mode="strict",
        partial_exit_enabled=False,
        partial_exit_count=2,
        partial_exit_rules=(),
    )
    base.update(kwargs)
    return AnalysisParams(**base)


def make_df(prices):
    rows = []
    for i, (o, h, l, c) in enumerate(prices):
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "stock_code": "000001.SZ",
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000,
                "prev_close": 9.5,
                "prev_high": 9.8,
                "prev_low": 9.2,
                "gap_pct_vs_prev_close": 2.0,
            }
        )
    return pd.DataFrame(rows)


def test_time_exit_case_a():
    df = make_df([(10, 10, 10, 10), (10, 10.1, 9.9, 10.1), (10, 10.1, 9.8, 10.0)])
    trade, reason = simulate_trade(df, 0, make_params(time_stop_days=2, time_stop_target_pct=1.5))
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "time_exit"


def test_time_exit_case_b_and_c():
    df = make_df([(10, 10, 10, 10), (10, 10.3, 9.9, 10.2), (10, 10.3, 9.9, 10.3), (10, 10.1, 9.9, 10.0)])
    trade_b, _ = simulate_trade(df, 0, make_params(time_stop_days=2, time_stop_target_pct=1.0, stop_loss_pct=1.0))
    assert trade_b["fills"][-1]["exit_type"] != "time_exit"

    trade_c, _ = simulate_trade(df, 0, make_params(time_stop_days=2, time_stop_target_pct=2.5))
    assert trade_c["fills"][-1]["exit_type"] == "time_exit"
    assert trade_c["fills"][-1]["holding_days"] == 3


def test_partial_two_batches_fixed_and_ma():
    rules = (
        PartialExitRule(True, 50.0, "fixed_tp", 1, target_profit_pct=5.0),
        PartialExitRule(True, 50.0, "ma_exit", 2, ma_period=2),
    )
    df = make_df([(10, 10, 10, 10), (10, 10.6, 9.9, 10.5), (10, 10.2, 9.9, 9.8)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_rules=rules))
    assert reason is None
    assert [f["exit_type"] for f in trade["fills"]] == ["fixed_tp", "ma_exit"]


def test_partial_three_batches_with_drawdown_and_activation():
    rules = (
        PartialExitRule(True, 30.0, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 30.0, "fixed_tp", 2, target_profit_pct=6.0),
        PartialExitRule(True, 40.0, "profit_drawdown", 3, drawdown_pct=5.0, min_profit_to_activate_drawdown=5.0),
    )
    df = make_df([(10, 10, 10, 10), (10, 10.4, 9.9, 10.3), (10, 10.8, 10.0, 10.7), (10, 10.8, 10.0, 10.1)])
    trade, _ = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=3, partial_exit_rules=rules))
    assert [f["exit_type"] for f in trade["fills"]] == ["fixed_tp", "fixed_tp", "profit_drawdown"]


def test_stop_loss_priority_over_partial():
    rules = (
        PartialExitRule(True, 50.0, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50.0, "ma_exit", 2, ma_period=2),
    )
    df = make_df([(10, 10, 10, 10), (10, 10.4, 9.9, 10.3), (10, 10.5, 9.0, 9.2)])
    trade, _ = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_rules=rules, stop_loss_pct=5.0))
    assert trade["fills"][-1]["exit_type"] == "stop_loss"
    assert abs(sum(f["weight"] for f in trade["fills"]) - 1.0) < 1e-12


def test_same_day_multi_fixed_tp_priority():
    rules = (
        PartialExitRule(True, 30.0, "fixed_tp", 2, target_profit_pct=3.0),
        PartialExitRule(True, 70.0, "fixed_tp", 1, target_profit_pct=3.0),
    )
    df = make_df([(10, 10, 10, 10), (10, 10.5, 9.9, 10.2)])
    trade, _ = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_rules=rules))
    assert trade["fills"][0]["weight"] == 0.7
    assert trade["fills"][1]["weight"] == 0.3


def test_strict_unclosed_trade_case_h_and_force_close_case_i():
    rules = (
        PartialExitRule(True, 50.0, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50.0, "ma_exit", 2, ma_period=10),
    )
    df = make_df([(10, 10, 10, 10), (10, 10.4, 9.9, 10.2), (10, 10.3, 10.1, 10.2)])
    trade_h, reason_h = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_rules=rules, time_exit_mode="strict", time_stop_days=99))
    assert trade_h is None and reason_h == "unclosed_trade"

    trade_i, reason_i = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_rules=rules, time_exit_mode="force_close", time_stop_days=99))
    assert reason_i is None
    assert trade_i["fills"][-1]["exit_type"] == "force_close"


def test_profit_drawdown_not_trigger_before_activation_case_j_and_zero_value_case_k():
    rules = (
        PartialExitRule(True, 100.0, "profit_drawdown", 1, drawdown_pct=1.0, min_profit_to_activate_drawdown=10.0),
        PartialExitRule(False, 0.0, "fixed_tp", 2, target_profit_pct=5.0),
    )
    df = make_df([(10, 10, 10, 10), (10, 10.7, 10.0, 10.6), (10, 10.7, 10.0, 10.5)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_rules=rules, time_exit_mode="strict", time_stop_days=99))
    assert trade is None and reason == "unclosed_trade"

    zero_rule = (
        PartialExitRule(True, 100.0, "fixed_tp", 1, target_profit_pct=0.0),
        PartialExitRule(False, 0.0, "ma_exit", 2, ma_period=2),
    )
    df2 = make_df([(10, 10, 10, 10), (10, 10.0, 9.9, 9.95)])
    trade2, reason2 = simulate_trade(df2, 0, make_params(partial_exit_enabled=True, partial_exit_rules=zero_rule))
    assert reason2 is None
    assert trade2["fills"][0]["exit_type"] == "fixed_tp"
