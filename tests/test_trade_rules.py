from __future__ import annotations

import pandas as pd

from models import AnalysisParams, PartialExitRule, validate_params
from rules import simulate_trade


def make_params(**overrides):
    base = dict(
        data_source_type="sqlite",
        db_path="/tmp/a.db",
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
    )
    base.update(overrides)
    return AnalysisParams(**base)


def make_stock_df(rows):
    data = []
    for i, (o, h, l, c) in enumerate(rows):
        data.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "stock_code": "000001.SZ",
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000,
                "prev_close": 99.0,
                "prev_high": 99.0,
                "prev_low": 98.0,
                "gap_pct_vs_prev_close": 0.0,
            }
        )
    return pd.DataFrame(data)


def test_time_exit_case_a_trigger_on_day_n_below_target():
    df = make_stock_df([(100, 101, 99, 100), (100, 101, 99, 101), (101, 101, 100, 101.5), (101, 101, 99, 100)])
    trade, reason = simulate_trade(df, 0, make_params(enable_take_profit=False, time_stop_days=2, time_stop_target_pct=2.0, stop_loss_pct=50.0))
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "time_exit"


def test_time_exit_case_b_meet_target_then_other_rule_exit():
    df = make_stock_df([(100, 101, 99, 100), (100, 103, 99, 102), (102, 103, 101, 102), (102, 102, 94, 95)])
    trade, reason = simulate_trade(df, 0, make_params(enable_take_profit=False, time_stop_days=2, time_stop_target_pct=1.0, stop_loss_pct=5.0))
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "stop_loss"


def test_time_exit_case_c_fall_below_target_later():
    df = make_stock_df([(100, 101, 99, 100), (100, 104, 99, 103), (103, 103, 102, 102), (102, 102, 100, 101)])
    trade, reason = simulate_trade(df, 0, make_params(enable_take_profit=False, time_stop_days=2, time_stop_target_pct=1.5, stop_loss_pct=50.0))
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "time_exit"
    assert trade["fills"][-1]["holding_days"] == 3


def test_partial_case_d_two_batch_fixed_tp_then_ma_exit():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=5.0),
        PartialExitRule(True, 50, "ma_exit", 2, ma_period=2),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 105, 99, 104), (104, 104, 99, 100), (100, 100, 99, 99)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=2, partial_exit_rules=rules, enable_take_profit=True, enable_ma_exit=True, stop_loss_pct=50.0))
    assert reason is None
    assert [f["exit_type"] for f in trade["fills"]] == ["fixed_tp", "ma_exit"]


def test_partial_case_e_three_batch_with_drawdown():
    rules = (
        PartialExitRule(True, 30, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 30, "fixed_tp", 2, target_profit_pct=6.0),
        PartialExitRule(True, 40, "profit_drawdown", 3, drawdown_pct=5.0, min_profit_to_activate_drawdown=5.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 104, 99, 103), (103, 108, 103, 107), (107, 107, 100, 101)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=3, partial_exit_rules=rules, enable_take_profit=False, stop_loss_pct=50.0))
    assert reason is None
    assert [f["exit_type"] for f in trade["fills"]] == ["fixed_tp", "fixed_tp", "profit_drawdown"]


def test_partial_case_f_stop_loss_has_highest_priority():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50, "fixed_tp", 2, target_profit_pct=5.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 106, 94, 95), (95, 96, 94, 95)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=2, partial_exit_rules=rules, stop_loss_pct=5.0))
    assert reason is None
    assert len(trade["fills"]) == 1
    assert trade["fills"][0]["exit_type"] == "stop_loss"


def test_partial_case_g_same_day_two_fixed_tp_by_priority():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 2, target_profit_pct=3.0),
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=2.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 106, 99, 105), (105, 106, 104, 105)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=2, partial_exit_rules=rules, stop_loss_pct=50.0))
    assert reason is None
    assert trade["fills"][0]["sell_price"] == 102.0
    assert trade["fills"][1]["sell_price"] == 103.0


def test_partial_case_h_strict_returns_unclosed_trade():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50, "ma_exit", 2, ma_period=3),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 104, 99, 103), (103, 103, 102, 103)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=2, partial_exit_rules=rules, time_exit_mode="strict", enable_take_profit=False, time_stop_days=2, time_stop_target_pct=-10.0, stop_loss_pct=50.0))
    assert trade is None
    assert reason == "unclosed_trade"


def test_partial_case_i_force_close_adds_fill():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50, "ma_exit", 2, ma_period=3),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 104, 99, 103), (103, 103, 102, 103)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=2, partial_exit_rules=rules, time_exit_mode="force_close", enable_take_profit=False, time_stop_days=2, time_stop_target_pct=-10.0, stop_loss_pct=50.0))
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "force_close"


def test_partial_case_j_activation_threshold_prevents_drawdown_trigger():
    rules = (
        PartialExitRule(True, 100, "profit_drawdown", 1, drawdown_pct=5.0, min_profit_to_activate_drawdown=10.0),
        PartialExitRule(False, 0, "fixed_tp", 2, target_profit_pct=1.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 106, 99, 101), (101, 101, 99, 100), (100, 100, 99, 99)])
    trade, reason = simulate_trade(df, 0, make_params(partial_exit_enabled=True, partial_exit_count=2, partial_exit_rules=rules, time_stop_days=2, time_stop_target_pct=0.0, stop_loss_pct=50.0))
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "time_exit"


def test_case_k_zero_value_param_not_ignored():
    params = make_params(
        partial_exit_enabled=True,
        partial_exit_count=2,
        partial_exit_rules=(
            PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=0.0),
            PartialExitRule(True, 50, "fixed_tp", 2, target_profit_pct=1.0),
        ),
    )
    errors, _ = validate_params(params)
    assert not errors


def test_partial_case_l_drawdown_uses_updated_trailing_peak():
    rules = (
        PartialExitRule(True, 100, "profit_drawdown", 1, drawdown_pct=20.0, min_profit_to_activate_drawdown=5.0),
        PartialExitRule(False, 0, "fixed_tp", 2, target_profit_pct=1.0),
    )
    # buy=100, 峰值先到 110（若按旧峰值20%回撤阈值=88会误触发），后续新峰值到 120，
    # 再回撤到 96（=120*(1-20%)）时才应触发。
    df = make_stock_df([
        (100, 101, 99, 100),
        (100, 110, 99, 100),
        (100, 120, 99, 119),
        (119, 119, 95, 96),
    ])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            enable_take_profit=False,
            stop_loss_pct=50.0,
            time_stop_days=3,
            time_stop_target_pct=-50.0,
        ),
    )
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "profit_drawdown"
    assert trade["fills"][-1]["holding_days"] == 3
    assert trade["fills"][-1]["sell_price"] == 96


def test_short_stop_loss_mirror():
    df = make_stock_df([(100, 101, 99, 100), (100, 106, 99, 105), (105, 106, 104, 105)])
    trade, reason = simulate_trade(df, 0, make_params(enable_take_profit=False, stop_loss_pct=5.0), direction="short")
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "stop_loss"


def test_short_fixed_tp_mirror():
    df = make_stock_df([(100, 101, 99, 100), (100, 101, 94, 95), (95, 96, 94, 95)])
    trade, reason = simulate_trade(df, 0, make_params(enable_take_profit=True, take_profit_pct=5.0, stop_loss_pct=50.0), direction="short")
    assert reason is None
    assert trade["fills"][-1]["exit_type"] == "take_profit"
