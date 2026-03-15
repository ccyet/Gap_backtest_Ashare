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
    exit_ma_series = (
        stock_df["close"].rolling(params.exit_ma_period).mean() if params.enable_ma_exit else pd.Series(math.nan, index=stock_df.index)
    )

    sell_price: float | None = None
    exit_type: str | None = None
    sell_day_idx: int | None = None
    sell_date: pd.Timestamp | None = None
    triggered_profit_drawdown_ratio = math.nan
    triggered_exit_ma_value = math.nan

    max_profit_pct = 0.0

    for holding_days in range(1, params.time_stop_days + 1):
        day_idx = signal_idx + holding_days
        day_row = stock_df.iloc[day_idx]
        day_date = pd.Timestamp(day_row["date"])
        day_close = float(day_row["close"])

        # 1) 固定止损
        if day_row["low"] <= stop_loss_price:
            sell_price = stop_loss_price
            exit_type = "stop_loss"
        # 2) 固定止盈（可选）
        elif params.enable_take_profit and day_row["high"] >= take_profit_price:
            sell_price = take_profit_price
            exit_type = "take_profit"
        else:
            # 3) 更新最大浮盈
            max_profit_pct = max(max_profit_pct, float(day_row["high"]) / buy_price - 1.0)

            # 4) 盈利回撤止盈（可选）
            if params.enable_profit_drawdown_exit and max_profit_pct > 0:
                current_profit_pct = day_close / buy_price - 1.0
                profit_drawdown_ratio = (max_profit_pct - current_profit_pct) / max_profit_pct
                if profit_drawdown_ratio >= params.profit_drawdown_ratio:
                    sell_price = day_close
                    exit_type = "profit_drawdown_exit"
                    triggered_profit_drawdown_ratio = float(profit_drawdown_ratio)

            # 5) 均线趋势离场（可选）
            if sell_price is None and params.enable_ma_exit:
                day_exit_ma = exit_ma_series.iloc[day_idx]
                if pd.notna(day_exit_ma):
                    day_exit_ma_value = float(day_exit_ma)
                    if day_close < day_exit_ma_value:
                        sell_price = day_close
                        exit_type = "ma_exit"
                        triggered_exit_ma_value = day_exit_ma_value

            # 6) 时间退出
            if sell_price is None and holding_days == params.time_stop_days:
                holding_return_at_n = day_close / buy_price - 1.0
                if holding_return_at_n < params.time_stop_target_ratio:
                    sell_price = day_close
                    exit_type = "time_exit"
                elif params.time_exit_mode == "force_close":
                    sell_price = day_close
                    exit_type = "time_exit"
                else:
                    return None, "time_target_met"

        if sell_price is not None and exit_type is not None:
            sell_day_idx = day_idx
            sell_date = day_date
            if params.enable_ma_exit and pd.isna(triggered_exit_ma_value):
                day_exit_ma = exit_ma_series.iloc[day_idx]
                if pd.notna(day_exit_ma):
                    triggered_exit_ma_value = float(day_exit_ma)
            break

    if sell_price is None or exit_type is None or sell_day_idx is None or sell_date is None:
        return None, "no_exit"

    actual_buy_cost = buy_price * (1.0 + params.buy_cost_ratio)
    actual_sell_value = sell_price * (1.0 - params.sell_cost_ratio)
    gross_return = sell_price / buy_price - 1.0
    net_return = actual_sell_value / actual_buy_cost - 1.0
    holding_slice = stock_df.iloc[signal_idx + 1 : sell_day_idx + 1]
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
        "sell_date": sell_date.date(),
        "exit_type": exit_type,
        "holding_days": int(sell_day_idx - signal_idx),
        "gross_return_pct": gross_return * 100.0,
        "net_return_pct": net_return * 100.0,
        "win_flag": 1 if net_return > 0 else 0,
        "mfe_pct": float(mfe) * 100.0,
        "mae_pct": float(mae) * 100.0,
        "max_profit_pct": float(mfe) * 100.0,
        "exit_ma_value": float(triggered_exit_ma_value) if pd.notna(triggered_exit_ma_value) else math.nan,
        "profit_drawdown_ratio": (
            float(triggered_profit_drawdown_ratio) * 100.0 if pd.notna(triggered_profit_drawdown_ratio) else math.nan
        ),
    }, None
