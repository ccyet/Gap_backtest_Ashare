from __future__ import annotations

import math
from typing import Any

import pandas as pd

from models import AnalysisParams


SIGNAL_COLUMNS = [
    "date",
    "stock_code",
    "prev_close",
    "prev_high",
    "prev_low",
    "open",
    "close",
    "volume",
    "gap_pct_vs_prev_close",
]


def apply_gap_filters(df: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    stock_df = df.sort_values("date").reset_index(drop=True).copy()

    stock_df["prev_close"] = stock_df["close"].shift(1)
    stock_df["prev_high"] = stock_df["high"].shift(1)
    stock_df["prev_low"] = stock_df["low"].shift(1)
    stock_df["gap_pct_vs_prev_close"] = (stock_df["open"] / stock_df["prev_close"] - 1.0) * 100.0

    if params.use_ma_filter:
        # 开单过滤只能使用信号当日开盘前已经知道的数据，因此均线向后移一日。
        stock_df["fast_ma"] = stock_df["close"].rolling(params.fast_ma_period).mean().shift(1)
        stock_df["slow_ma"] = stock_df["close"].rolling(params.slow_ma_period).mean().shift(1)
    else:
        stock_df["fast_ma"] = math.nan
        stock_df["slow_ma"] = math.nan

    signal_mask = stock_df["prev_close"].notna() & stock_df["prev_high"].notna() & stock_df["prev_low"].notna()

    if params.gap_direction == "up":
        signal_mask &= stock_df["open"] > stock_df["prev_high"] * (1.0 + params.gap_ratio)
        signal_mask &= stock_df["gap_pct_vs_prev_close"] <= params.max_gap_filter_pct
        if params.use_ma_filter:
            signal_mask &= stock_df["fast_ma"].notna() & stock_df["slow_ma"].notna()
            signal_mask &= stock_df["open"] > stock_df["fast_ma"]
            signal_mask &= stock_df["open"] > stock_df["slow_ma"]
    else:
        signal_mask &= stock_df["open"] < stock_df["prev_low"] * (1.0 - params.gap_ratio)
        signal_mask &= stock_df["gap_pct_vs_prev_close"] >= -params.max_gap_filter_pct
        if params.use_ma_filter:
            signal_mask &= stock_df["fast_ma"].notna() & stock_df["slow_ma"].notna()
            signal_mask &= stock_df["open"] < stock_df["fast_ma"]
            signal_mask &= stock_df["open"] < stock_df["slow_ma"]

    stock_df["is_signal"] = signal_mask
    return stock_df


def simulate_trade(
    stock_df: pd.DataFrame,
    signal_idx: int,
    params: AnalysisParams,
) -> tuple[dict[str, Any] | None, str | None]:
    if signal_idx + params.time_stop_days >= len(stock_df):
        return None, "insufficient_future"

    signal_row = stock_df.iloc[signal_idx]
    buy_price = float(signal_row["open"])
    buy_date = pd.Timestamp(signal_row["date"])

    stop_loss_price = buy_price * (1.0 - params.stop_loss_ratio)
    take_profit_price = buy_price * (1.0 + params.take_profit_ratio)

    for holding_days in range(1, params.time_stop_days + 1):
        day_row = stock_df.iloc[signal_idx + holding_days]
        day_date = pd.Timestamp(day_row["date"])

        if day_row["low"] <= stop_loss_price:
            sell_price = stop_loss_price
            exit_type = "stop_loss"
        elif day_row["high"] >= take_profit_price:
            sell_price = take_profit_price
            exit_type = "take_profit"
        elif holding_days == params.time_stop_days:
            holding_return_at_n = float(day_row["close"]) / buy_price - 1.0
            if holding_return_at_n < params.time_stop_target_ratio:
                sell_price = float(day_row["close"])
                exit_type = "time_exit"
            elif params.time_exit_mode == "force_close":
                sell_price = float(day_row["close"])
                exit_type = "time_exit"
            else:
                return None, "time_target_met"
        else:
            continue

        actual_buy_cost = buy_price * (1.0 + params.buy_cost_ratio)
        actual_sell_value = sell_price * (1.0 - params.sell_cost_ratio)
        gross_return = sell_price / buy_price - 1.0
        net_return = actual_sell_value / actual_buy_cost - 1.0
        holding_slice = stock_df.iloc[signal_idx + 1 : signal_idx + holding_days + 1]
        mfe = holding_slice["high"].max() / buy_price - 1.0
        mae = holding_slice["low"].min() / buy_price - 1.0

        return {
            "date": buy_date.date(),
            "stock_code": signal_row["stock_code"],
            "prev_close": float(signal_row["prev_close"]),
            "prev_high": float(signal_row["prev_high"]),
            "prev_low": float(signal_row["prev_low"]),
            "open": float(signal_row["open"]),
            "close": float(signal_row["close"]),
            "volume": float(signal_row["volume"]) if pd.notna(signal_row["volume"]) else math.nan,
            "gap_pct_vs_prev_close": float(signal_row["gap_pct_vs_prev_close"]),
            "buy_price": buy_price,
            "sell_price": float(sell_price),
            "sell_date": day_date.date(),
            "exit_type": exit_type,
            "holding_days": int(holding_days),
            "gross_return_pct": gross_return * 100.0,
            "net_return_pct": net_return * 100.0,
            "win_flag": 1 if net_return > 0 else 0,
            "mfe_pct": float(mfe) * 100.0,
            "mae_pct": float(mae) * 100.0,
        }, None

    return None, "no_exit"
