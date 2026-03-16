from __future__ import annotations

import math
from typing import Any

import pandas as pd

from models import AnalysisParams, PartialExitRule


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

FLOAT_TOLERANCE = 1e-12


def apply_gap_filters(df: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    stock_df = df.sort_values("date").reset_index(drop=True).copy()

    stock_df["prev_close"] = stock_df["close"].shift(1)
    stock_df["prev_high"] = stock_df["high"].shift(1)
    stock_df["prev_low"] = stock_df["low"].shift(1)
    stock_df["gap_pct_vs_prev_close"] = (stock_df["open"] / stock_df["prev_close"] - 1.0) * 100.0

    if params.use_ma_filter:
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


def _build_partial_ma_map(stock_df: pd.DataFrame, params: AnalysisParams) -> dict[int, pd.Series]:
    if not params.partial_exit_enabled:
        return {}
    periods = sorted(
        {
            int(rule.ma_period)
            for rule in params.partial_exit_rules
            if rule.enabled and rule.mode == "ma_exit" and rule.ma_period is not None
        }
    )
    return {period: stock_df["close"].rolling(period).mean() for period in periods}


def _trigger_partial_rule(
    rule: PartialExitRule,
    day_row: pd.Series,
    day_idx: int,
    buy_price: float,
    peak_price: float,
    ma_map: dict[int, pd.Series],
) -> bool:
    day_close = float(day_row["close"])
    day_high = float(day_row["high"])

    if rule.mode == "fixed_tp" and rule.target_profit_pct is not None:
        target_profit_ratio = rule.target_profit_pct / 100.0
        return day_high >= buy_price * (1.0 + target_profit_ratio)

    if rule.mode == "ma_exit" and rule.ma_period is not None:
        ma_series = ma_map.get(int(rule.ma_period))
        if ma_series is None:
            return False
        ma_value = ma_series.iloc[day_idx]
        return pd.notna(ma_value) and day_close < float(ma_value)

    if rule.mode == "profit_drawdown" and rule.drawdown_pct is not None:
        activation = 0.05
        if rule.min_profit_to_activate_drawdown is not None:
            activation = rule.min_profit_to_activate_drawdown / 100.0
        if buy_price <= 0:
            return False
        if (peak_price / buy_price - 1.0) < activation:
            return False
        if peak_price <= 0:
            return False
        drawdown_ratio = (peak_price - day_close) / peak_price
        return drawdown_ratio >= (rule.drawdown_pct / 100.0)

    return False


def simulate_trade(stock_df: pd.DataFrame, signal_idx: int, params: AnalysisParams) -> tuple[dict[str, Any] | None, str | None]:
    if signal_idx + 1 >= len(stock_df):
        return None, "insufficient_future"

    signal_row = stock_df.iloc[signal_idx]
    buy_price = float(signal_row["open"])
    buy_date = pd.Timestamp(signal_row["date"])

    stop_loss_price = buy_price * (1.0 - params.stop_loss_ratio)
    take_profit_price = buy_price * (1.0 + params.take_profit_ratio)
    legacy_ma_series = stock_df["close"].rolling(params.exit_ma_period).mean() if params.enable_ma_exit else pd.Series(math.nan, index=stock_df.index)
    partial_ma_map = _build_partial_ma_map(stock_df, params)

    fills: list[dict[str, Any]] = []
    remaining_weight = 1.0
    peak_price = buy_price
    max_profit_pct = 0.0
    triggered_exit_ma_value = math.nan
    triggered_profit_drawdown_ratio = math.nan
    exited = False

    enabled_partial_rules = sorted(
        [rule for rule in params.partial_exit_rules if rule.enabled],
        key=lambda item: item.priority,
    )
    partial_rule_executed: set[int] = set()

    max_holding_days = len(stock_df) - signal_idx - 1

    for holding_days in range(1, max_holding_days + 1):
        day_idx = signal_idx + holding_days
        day_row = stock_df.iloc[day_idx]
        day_date = pd.Timestamp(day_row["date"])
        day_close = float(day_row["close"])
        day_high = float(day_row["high"])

        peak_price = max(peak_price, day_high)
        max_profit_pct = max(max_profit_pct, peak_price / buy_price - 1.0)

        # 1) 全仓止损
        if float(day_row["low"]) <= stop_loss_price and remaining_weight > FLOAT_TOLERANCE:
            fills.append(
                {
                    "sell_date": str(day_date.date()),
                    "sell_price": stop_loss_price,
                    "weight": remaining_weight,
                    "exit_type": "stop_loss",
                    "holding_days": holding_days,
                }
            )
            remaining_weight = 0.0
            exited = True
            break

        # 2) 分批退出（按 priority）
        if params.partial_exit_enabled and remaining_weight > FLOAT_TOLERANCE:
            for idx, rule in enumerate(enabled_partial_rules):
                if idx in partial_rule_executed:
                    continue
                if remaining_weight <= FLOAT_TOLERANCE:
                    break

                if _trigger_partial_rule(rule, day_row, day_idx, buy_price, peak_price, partial_ma_map):
                    target_weight = rule.weight_pct / 100.0
                    fill_weight = min(target_weight, remaining_weight)
                    fills.append(
                        {
                            "sell_date": str(day_date.date()),
                            "sell_price": day_close,
                            "weight": fill_weight,
                            "exit_type": rule.mode,
                            "holding_days": holding_days,
                        }
                    )
                    remaining_weight -= fill_weight
                    partial_rule_executed.add(idx)

            if remaining_weight <= FLOAT_TOLERANCE:
                exited = True
                break

        # 3) 未启用分批时旧版单次退出逻辑
        if not params.partial_exit_enabled and remaining_weight > FLOAT_TOLERANCE:
            if params.enable_take_profit and day_high >= take_profit_price:
                fills.append(
                    {
                        "sell_date": str(day_date.date()),
                        "sell_price": take_profit_price,
                        "weight": remaining_weight,
                        "exit_type": "take_profit",
                        "holding_days": holding_days,
                    }
                )
                remaining_weight = 0.0
                exited = True
                break

            if params.enable_profit_drawdown_exit and max_profit_pct > 0:
                current_profit_pct = day_close / buy_price - 1.0
                profit_drawdown_ratio = (max_profit_pct - current_profit_pct) / max_profit_pct
                if profit_drawdown_ratio >= params.profit_drawdown_ratio:
                    fills.append(
                        {
                            "sell_date": str(day_date.date()),
                            "sell_price": day_close,
                            "weight": remaining_weight,
                            "exit_type": "profit_drawdown_exit",
                            "holding_days": holding_days,
                        }
                    )
                    triggered_profit_drawdown_ratio = float(profit_drawdown_ratio)
                    remaining_weight = 0.0
                    exited = True
                    break

            if params.enable_ma_exit:
                day_exit_ma = legacy_ma_series.iloc[day_idx]
                if pd.notna(day_exit_ma):
                    day_exit_ma_value = float(day_exit_ma)
                    triggered_exit_ma_value = day_exit_ma_value
                    if day_close < day_exit_ma_value:
                        fills.append(
                            {
                                "sell_date": str(day_date.date()),
                                "sell_price": day_close,
                                "weight": remaining_weight,
                                "exit_type": "ma_exit",
                                "holding_days": holding_days,
                            }
                        )
                        remaining_weight = 0.0
                        exited = True
                        break

        # 4) 时间退出（持续检查）
        if remaining_weight > FLOAT_TOLERANCE and holding_days >= params.time_stop_days:
            if day_close / buy_price - 1.0 < params.time_stop_target_ratio:
                fills.append(
                    {
                        "sell_date": str(day_date.date()),
                        "sell_price": day_close,
                        "weight": remaining_weight,
                        "exit_type": "time_exit",
                        "holding_days": holding_days,
                    }
                )
                remaining_weight = 0.0
                exited = True
                break

    # 5) 数据结束处理
    if remaining_weight > FLOAT_TOLERANCE and not exited:
        if params.time_exit_mode == "strict":
            return None, "unclosed_trade"
        if params.time_exit_mode == "force_close":
            last_row = stock_df.iloc[-1]
            last_date = pd.Timestamp(last_row["date"])
            fills.append(
                {
                    "sell_date": str(last_date.date()),
                    "sell_price": float(last_row["close"]),
                    "weight": remaining_weight,
                    "exit_type": "force_close",
                    "holding_days": len(stock_df) - signal_idx - 1,
                }
            )
            remaining_weight = 0.0

    if not fills:
        return None, "no_exit"

    total_weight = sum(fill["weight"] for fill in fills)
    if total_weight <= FLOAT_TOLERANCE:
        return None, "no_exit"

    weighted_sell_price = sum(fill["sell_price"] * fill["weight"] for fill in fills) / total_weight
    last_fill = fills[-1]
    sell_day_idx = signal_idx + int(last_fill["holding_days"])

    actual_buy_cost = buy_price * (1.0 + params.buy_cost_ratio)
    actual_sell_value = weighted_sell_price * (1.0 - params.sell_cost_ratio)
    gross_return = weighted_sell_price / buy_price - 1.0
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
        "buy_date": str(buy_date.date()),
        "fills": fills,
        "sell_price": float(weighted_sell_price),
        "sell_date": last_fill["sell_date"],
        "exit_type": "|".join(fill["exit_type"] for fill in fills),
        "holding_days": int(last_fill["holding_days"]),
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
