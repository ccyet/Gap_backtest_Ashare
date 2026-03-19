from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

import analyzer
from models import AnalysisParams


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
        "entry_factor": "trend_breakout",
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


def make_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "stock_code": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "open": [10.0, 10.5, 20.0, 21.0],
            "high": [10.2, 10.7, 20.2, 21.3],
            "low": [9.8, 10.2, 19.8, 20.6],
            "close": [10.1, 10.6, 20.1, 21.1],
            "volume": [1000.0, 1200.0, 900.0, 1100.0],
        }
    )


def _build_trade(
    *,
    trade_date: date,
    stock_code: str,
    buy_price: float,
    sell_date: str,
    fill_type: str,
) -> dict[str, object]:
    fills = [
        {
            "sell_date": sell_date,
            "sell_price": buy_price,
            "weight": 1.0,
            "exit_type": "force_close",
            "holding_days": 1,
        }
    ]
    return {
        "date": trade_date,
        "stock_code": stock_code,
        "prev_close": buy_price,
        "prev_high": buy_price,
        "prev_low": buy_price,
        "open": buy_price,
        "close": buy_price,
        "volume": 1000.0,
        "gap_pct_vs_prev_close": 0.0,
        "buy_date": str(trade_date),
        "buy_price": buy_price,
        "sell_price": buy_price,
        "sell_date": pd.to_datetime(sell_date).date(),
        "exit_type": "force_close",
        "holding_days": 1,
        "fills": fills,
        "gross_return_pct": 0.0,
        "net_return_pct": 0.0,
        "win_flag": 0,
        "mfe_pct": 0.0,
        "mae_pct": 0.0,
        "max_profit_pct": 0.0,
        "exit_ma_value": float("nan"),
        "profit_drawdown_ratio": float("nan"),
        "entry_factor": "trend_breakout",
        "entry_trigger_price": buy_price,
        "entry_fill_type": fill_type,
    }


def test_scan_trade_candidates_propagates_skip_counters_and_entry_fields(
    monkeypatch,
) -> None:
    def fake_apply_gap_filters(
        stock_df: pd.DataFrame, _params: AnalysisParams
    ) -> pd.DataFrame:
        enriched = stock_df.sort_values("date").reset_index(drop=True).copy()
        enriched["is_signal"] = True
        return enriched

    def fake_simulate_trade(
        stock_df: pd.DataFrame,
        signal_idx: int,
        _params: AnalysisParams,
        direction: str = "long",
    ) -> tuple[dict[str, object] | None, str | None]:
        del direction
        stock_code = str(stock_df.iloc[signal_idx]["stock_code"])
        if stock_code == "000001.SZ" and signal_idx == 0:
            return None, "entry_not_filled"
        if stock_code == "000002.SZ" and signal_idx == 0:
            return None, "locked_bar_unfillable"
        if stock_code == "000001.SZ":
            return (
                _build_trade(
                    trade_date=date(2024, 1, 3),
                    stock_code=stock_code,
                    buy_price=10.5,
                    sell_date="2024-01-06",
                    fill_type="open",
                ),
                None,
            )
        return (
            _build_trade(
                trade_date=date(2024, 1, 2),
                stock_code=stock_code,
                buy_price=21.0,
                sell_date="2024-01-05",
                fill_type="trigger",
            ),
            None,
        )

    monkeypatch.setattr(analyzer, "apply_gap_filters", fake_apply_gap_filters)
    monkeypatch.setattr(analyzer, "simulate_trade", fake_simulate_trade)

    detail_df, stats = analyzer.scan_trade_candidates(make_market_data(), make_params())

    assert stats["signal_count"] == 4
    assert stats["closed_trade_candidates"] == 2
    assert stats["skipped_entry_not_filled"] == 1
    assert stats["skipped_locked_bar_unfillable"] == 1

    assert "entry_factor" in detail_df.columns
    assert "entry_trigger_price" in detail_df.columns
    assert "entry_fill_type" in detail_df.columns
    assert set(detail_df["entry_factor"]) == {"trend_breakout"}
    assert set(detail_df["entry_fill_type"]) == {"open", "trigger"}


def test_scan_trade_candidates_keeps_date_stock_sell_date_sort_order(
    monkeypatch,
) -> None:
    def fake_apply_gap_filters(
        stock_df: pd.DataFrame, _params: AnalysisParams
    ) -> pd.DataFrame:
        enriched = stock_df.sort_values("date").reset_index(drop=True).copy()
        enriched["is_signal"] = [False, True]
        return enriched

    def fake_simulate_trade(
        stock_df: pd.DataFrame,
        signal_idx: int,
        _params: AnalysisParams,
        direction: str = "long",
    ) -> tuple[dict[str, object] | None, str | None]:
        del direction
        stock_code = str(stock_df.iloc[signal_idx]["stock_code"])
        if stock_code == "000001.SZ":
            return (
                _build_trade(
                    trade_date=date(2024, 1, 3),
                    stock_code=stock_code,
                    buy_price=10.5,
                    sell_date="2024-01-06",
                    fill_type="open",
                ),
                None,
            )
        return (
            _build_trade(
                trade_date=date(2024, 1, 2),
                stock_code=stock_code,
                buy_price=21.0,
                sell_date="2024-01-05",
                fill_type="trigger",
            ),
            None,
        )

    monkeypatch.setattr(analyzer, "apply_gap_filters", fake_apply_gap_filters)
    monkeypatch.setattr(analyzer, "simulate_trade", fake_simulate_trade)

    detail_df, _ = analyzer.scan_trade_candidates(make_market_data(), make_params())

    observed = detail_df[["date", "stock_code", "sell_date"]].copy()
    observed["date"] = pd.Series(pd.to_datetime(observed["date"])).map(
        lambda ts: ts.date()
    )
    observed["sell_date"] = pd.Series(pd.to_datetime(observed["sell_date"])).map(
        lambda ts: ts.date()
    )
    expected = pd.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "stock_code": ["000002.SZ", "000001.SZ"],
            "sell_date": [
                pd.Timestamp("2024-01-05").date(),
                pd.Timestamp("2024-01-06").date(),
            ],
        }
    )
    pd.testing.assert_frame_equal(observed.reset_index(drop=True), expected)


def test_empty_scan_stats_include_new_skip_counters() -> None:
    detail_df, stats = analyzer.scan_trade_candidates(pd.DataFrame(), make_params())

    assert detail_df.empty
    assert stats["skipped_entry_not_filled"] == 0
    assert stats["skipped_locked_bar_unfillable"] == 0
