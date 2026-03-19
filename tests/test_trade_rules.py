from __future__ import annotations

from typing import Any

import pandas as pd

from models import AnalysisParams, PartialExitRule, validate_params
from rules import simulate_trade


def make_params(**overrides: Any) -> AnalysisParams:
    base: dict[str, Any] = dict(
        data_source_type="sqlite",
        db_path="/tmp/a.db",
        table_name=None,
        column_overrides={},
        excel_sheet_name=None,
        start_date="2024-01-01",
        end_date="2024-12-31",
        stock_codes=(),
        gap_direction="up",
        gap_entry_mode="strict_break",
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
        buy_slippage_pct=0.0,
        sell_slippage_pct=0.0,
        time_exit_mode="strict",
    )
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


def require_trade(trade: dict[str, Any] | None, reason: str | None) -> dict[str, Any]:
    assert reason is None
    assert trade is not None
    return trade


def test_time_exit_case_a_trigger_on_day_n_below_target():
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 101, 99, 101),
            (101, 101, 100, 101.5),
            (101, 101, 99, 100),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=False,
            time_stop_days=2,
            time_stop_target_pct=2.0,
            stop_loss_pct=50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "time_exit"


def test_time_exit_case_b_meet_target_then_other_rule_exit():
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 103, 99, 102),
            (102, 103, 101, 102),
            (102, 102, 94, 95),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=False,
            time_stop_days=2,
            time_stop_target_pct=1.0,
            stop_loss_pct=5.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "stop_loss"


def test_time_exit_case_c_fall_below_target_later():
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 104, 99, 103),
            (103, 103, 102, 102),
            (102, 102, 100, 101),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=False,
            time_stop_days=2,
            time_stop_target_pct=1.5,
            stop_loss_pct=50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "time_exit"
    assert trade["fills"][-1]["holding_days"] == 3


def test_partial_case_d_two_batch_fixed_tp_then_ma_exit():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=5.0),
        PartialExitRule(True, 50, "ma_exit", 2, ma_period=2),
    )
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 105, 99, 104),
            (104, 104, 99, 100),
            (100, 100, 99, 99),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            enable_take_profit=True,
            enable_ma_exit=True,
            stop_loss_pct=50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert [f["exit_type"] for f in trade["fills"]] == ["fixed_tp", "ma_exit"]


def test_partial_case_e_three_batch_with_drawdown():
    rules = (
        PartialExitRule(True, 30, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 30, "fixed_tp", 2, target_profit_pct=6.0),
        PartialExitRule(
            True,
            40,
            "profit_drawdown",
            3,
            drawdown_pct=5.0,
            min_profit_to_activate_drawdown=5.0,
        ),
    )
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 104, 99, 103),
            (103, 108, 103, 107),
            (107, 107, 100, 101),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=3,
            partial_exit_rules=rules,
            enable_take_profit=False,
            stop_loss_pct=50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert [f["exit_type"] for f in trade["fills"]] == [
        "fixed_tp",
        "fixed_tp",
        "profit_drawdown",
    ]


def test_partial_case_f_stop_loss_has_highest_priority():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50, "fixed_tp", 2, target_profit_pct=5.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 106, 94, 95), (95, 96, 94, 95)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            stop_loss_pct=5.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert len(trade["fills"]) == 1
    assert trade["fills"][0]["exit_type"] == "stop_loss"


def test_partial_case_g_same_day_two_fixed_tp_by_priority():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 2, target_profit_pct=3.0),
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=2.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 106, 99, 105), (105, 106, 104, 105)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            stop_loss_pct=50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][0]["sell_price"] == 102.0
    assert trade["fills"][1]["sell_price"] == 103.0


def test_partial_case_h_strict_returns_unclosed_trade():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50, "ma_exit", 2, ma_period=3),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 104, 99, 103), (103, 103, 102, 103)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            time_exit_mode="strict",
            enable_take_profit=False,
            time_stop_days=2,
            time_stop_target_pct=-10.0,
            stop_loss_pct=50.0,
        ),
    )
    assert trade is None
    assert reason == "unclosed_trade"


def test_partial_case_i_force_close_adds_fill():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=3.0),
        PartialExitRule(True, 50, "ma_exit", 2, ma_period=3),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 104, 99, 103), (103, 103, 102, 103)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            time_exit_mode="force_close",
            enable_take_profit=False,
            time_stop_days=2,
            time_stop_target_pct=-10.0,
            stop_loss_pct=50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "force_close"


def test_partial_case_j_activation_threshold_prevents_drawdown_trigger():
    rules = (
        PartialExitRule(
            True,
            100,
            "profit_drawdown",
            1,
            drawdown_pct=5.0,
            min_profit_to_activate_drawdown=10.0,
        ),
        PartialExitRule(False, 0, "fixed_tp", 2, target_profit_pct=1.0),
    )
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 106, 99, 101),
            (101, 101, 99, 100),
            (100, 100, 99, 99),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            time_stop_days=2,
            time_stop_target_pct=0.0,
            stop_loss_pct=50.0,
        ),
    )
    trade = require_trade(trade, reason)
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
        PartialExitRule(
            True,
            100,
            "profit_drawdown",
            1,
            drawdown_pct=20.0,
            min_profit_to_activate_drawdown=5.0,
        ),
        PartialExitRule(False, 0, "fixed_tp", 2, target_profit_pct=1.0),
    )
    # 新语义下看的是整笔利润回撤而非最高价回撤：
    # buy=100, 峰值到120时峰值利润=20%，若利润回撤20%，则当前利润降到16%即触发。
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 110, 99, 109),
            (109, 120, 108, 119),
            (119, 119, 115, 116),
        ]
    )
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
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "profit_drawdown"
    assert trade["fills"][-1]["holding_days"] == 3
    assert trade["fills"][-1]["sell_price"] == 116


def test_short_stop_loss_mirror():
    df = make_stock_df([(100, 101, 99, 100), (100, 106, 99, 105), (105, 106, 104, 105)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(enable_take_profit=False, stop_loss_pct=5.0),
        direction="short",
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "stop_loss"


def test_short_fixed_tp_mirror():
    df = make_stock_df([(100, 101, 99, 100), (100, 101, 94, 95), (95, 96, 94, 95)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(enable_take_profit=True, take_profit_pct=5.0, stop_loss_pct=50.0),
        direction="short",
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "take_profit"


def test_partial_exits_execute_before_same_day_time_exit():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=2.0),
        PartialExitRule(True, 50, "fixed_tp", 2, target_profit_pct=5.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 103, 99, 101), (101, 101, 100, 101)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            time_stop_days=1,
            time_stop_target_pct=2.0,
            stop_loss_pct=50.0,
            enable_take_profit=False,
        ),
    )
    trade = require_trade(trade, reason)
    assert [fill["exit_type"] for fill in trade["fills"]] == ["fixed_tp", "time_exit"]


def test_long_slippage_adjusts_entry_and_exit_prices():
    df = make_stock_df([(100, 101, 99, 100), (100, 105, 99, 104), (104, 104, 103, 103)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=True,
            take_profit_pct=5.0,
            stop_loss_pct=50.0,
            buy_slippage_pct=1.0,
            sell_slippage_pct=2.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert abs(trade["buy_price"] - 101.0) < 1e-12
    assert abs(trade["fills"][-1]["sell_price"] - 102.9) < 1e-12


def test_short_slippage_adjusts_entry_and_cover_prices():
    df = make_stock_df([(100, 101, 99, 100), (100, 101, 94, 95), (95, 96, 94, 95)])
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=True,
            take_profit_pct=5.0,
            stop_loss_pct=50.0,
            buy_slippage_pct=1.0,
            sell_slippage_pct=2.0,
        ),
        direction="short",
    )
    trade = require_trade(trade, reason)
    assert abs(trade["buy_price"] - 98.0) < 1e-12
    assert abs(trade["fills"][-1]["sell_price"] - 95.95) < 1e-12


def test_slippage_does_not_change_partial_fixed_tp_trigger_timing():
    rules = (
        PartialExitRule(True, 100, "fixed_tp", 1, target_profit_pct=5.0),
        PartialExitRule(False, 0, "fixed_tp", 2, target_profit_pct=1.0),
    )
    df = make_stock_df([(100, 101, 99, 100), (100, 105, 99, 104), (104, 104, 103, 103)])
    base_trade, base_reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            stop_loss_pct=50.0,
        ),
    )
    slipped_trade, slipped_reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            stop_loss_pct=50.0,
            buy_slippage_pct=1.0,
            sell_slippage_pct=2.0,
        ),
    )
    base_trade = require_trade(base_trade, base_reason)
    slipped_trade = require_trade(slipped_trade, slipped_reason)
    assert (
        base_trade["fills"][-1]["holding_days"]
        == slipped_trade["fills"][-1]["holding_days"]
        == 1
    )


def test_slippage_does_not_change_whole_position_drawdown_trigger_timing():
    df = make_stock_df([(100, 101, 99, 100), (100, 110, 99, 109), (109, 109, 103, 104)])
    base_trade, base_reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=False,
            enable_profit_drawdown_exit=True,
            profit_drawdown_pct=40.0,
            stop_loss_pct=50.0,
            time_stop_days=2,
            time_stop_target_pct=-50.0,
        ),
    )
    slipped_trade, slipped_reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=False,
            enable_profit_drawdown_exit=True,
            profit_drawdown_pct=40.0,
            stop_loss_pct=50.0,
            time_stop_days=2,
            time_stop_target_pct=-50.0,
            buy_slippage_pct=1.0,
            sell_slippage_pct=2.0,
        ),
    )
    base_trade = require_trade(base_trade, base_reason)
    slipped_trade = require_trade(slipped_trade, slipped_reason)
    assert (
        base_trade["fills"][-1]["exit_type"]
        == slipped_trade["fills"][-1]["exit_type"]
        == "profit_drawdown_exit"
    )
    assert (
        base_trade["fills"][-1]["holding_days"]
        == slipped_trade["fills"][-1]["holding_days"]
        == 2
    )


def test_partial_total_profit_drawdown_uses_locked_first_batch_profit():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=10.0),
        PartialExitRule(
            True,
            50,
            "profit_drawdown",
            2,
            drawdown_pct=50.0,
            min_profit_to_activate_drawdown=8.0,
        ),
    )
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 111, 99, 110),
            (110, 111, 99, 100),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            enable_take_profit=False,
            stop_loss_pct=50.0,
            time_stop_days=2,
            time_stop_target_pct=-50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert [fill["exit_type"] for fill in trade["fills"]] == [
        "fixed_tp",
        "profit_drawdown",
    ]
    assert abs(trade["fills"][0]["sell_price"] - 110.0) < 1e-12
    assert trade["fills"][1]["sell_price"] == 100.0


def test_total_profit_drawdown_can_trigger_when_peak_price_drawdown_would_not():
    rules = (
        PartialExitRule(
            True,
            100,
            "profit_drawdown",
            1,
            drawdown_pct=20.0,
            min_profit_to_activate_drawdown=5.0,
        ),
        PartialExitRule(False, 0, "fixed_tp", 2, target_profit_pct=1.0),
    )
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 150, 99, 149),
            (149, 149, 134, 135),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            enable_take_profit=False,
            stop_loss_pct=50.0,
            time_stop_days=2,
            time_stop_target_pct=-50.0,
        ),
    )
    trade = require_trade(trade, reason)
    assert trade["fills"][-1]["exit_type"] == "profit_drawdown"
    assert trade["fills"][-1]["holding_days"] == 2
    assert trade["fills"][-1]["sell_price"] == 135.0


def test_total_profit_drawdown_activation_uses_total_trade_peak_profit():
    rules = (
        PartialExitRule(True, 50, "fixed_tp", 1, target_profit_pct=10.0),
        PartialExitRule(
            True,
            50,
            "profit_drawdown",
            2,
            drawdown_pct=20.0,
            min_profit_to_activate_drawdown=12.0,
        ),
    )
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 111, 99, 110),
            (110, 111, 104, 105),
        ]
    )
    trade, reason = simulate_trade(
        df,
        0,
        make_params(
            partial_exit_enabled=True,
            partial_exit_count=2,
            partial_exit_rules=rules,
            enable_take_profit=False,
            stop_loss_pct=50.0,
            time_stop_days=2,
            time_stop_target_pct=-50.0,
            time_exit_mode="force_close",
        ),
    )
    trade = require_trade(trade, reason)
    assert [fill["exit_type"] for fill in trade["fills"]] == ["fixed_tp", "force_close"]


def test_slippage_does_not_change_time_exit_trigger_timing():
    df = make_stock_df(
        [
            (100, 101, 99, 100),
            (100, 101, 99, 101),
            (101, 101, 100, 101.5),
            (101, 101, 99, 100),
        ]
    )
    base_trade, base_reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=False,
            time_stop_days=2,
            time_stop_target_pct=2.0,
            stop_loss_pct=50.0,
        ),
    )
    slipped_trade, slipped_reason = simulate_trade(
        df,
        0,
        make_params(
            enable_take_profit=False,
            time_stop_days=2,
            time_stop_target_pct=2.0,
            stop_loss_pct=50.0,
            buy_slippage_pct=1.0,
            sell_slippage_pct=2.0,
        ),
    )
    base_trade = require_trade(base_trade, base_reason)
    slipped_trade = require_trade(slipped_trade, slipped_reason)
    assert (
        base_trade["fills"][-1]["exit_type"]
        == slipped_trade["fills"][-1]["exit_type"]
        == "time_exit"
    )
    assert (
        base_trade["fills"][-1]["holding_days"]
        == slipped_trade["fills"][-1]["holding_days"]
    )
