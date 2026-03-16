from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import pandas as pd

from models import EPSILON, AnalysisParams, PartialExitRule, TradeFill


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


def _build_partial_ma_series(stock_df: pd.DataFrame, rules: list[PartialExitRule]) -> dict[int, pd.Series]:
    periods = {rule.ma_period for rule in rules if rule.mode == "ma_exit" and rule.ma_period is not None}
    return {period: stock_df["close"].rolling(period).mean() for period in periods}


def _rule_triggered(
    rule: PartialExitRule,
    day_row: pd.Series,
    buy_price: float,
    peak_price: float,
    ma_series: dict[int, pd.Series],
    day_idx: int,
) -> bool:
    if rule.mode == "fixed_tp":
        if rule.target_profit_ratio is None:
            return False
        return float(day_row["high"]) >= buy_price * (1.0 + rule.target_profit_ratio)

    if rule.mode == "ma_exit":
        if rule.ma_period is None:
            return False
        day_ma = ma_series.get(rule.ma_period)
        if day_ma is None:
            return False
        ma_value = day_ma.iloc[day_idx]
        if pd.isna(ma_value):
            return False
        return float(day_row["close"]) < float(ma_value)

    if rule.mode == "profit_drawdown":
        if rule.drawdown_ratio is None:
            return False
        activation_price = buy_price * (1.0 + rule.min_profit_to_activate_drawdown_ratio)
        if peak_price < activation_price:
            return False
        day_close = float(day_row["close"])
        if peak_price <= EPSILON:
            return False
        price_drawdown = (peak_price - day_close) / peak_price
        return price_drawdown >= rule.drawdown_ratio

    return False


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
    exit_ma_series = stock_df["close"].rolling(params.exit_ma_period).mean() if params.enable_ma_exit else None

    fills: list[TradeFill] = []
    remaining_weight = 1.0
    max_holding_days = len(stock_df) - signal_idx - 1
    peak_price = buy_price

    partial_rules = sorted(
        [rule for rule in params.partial_exit_rules if rule.enabled],
        key=lambda item: item.priority,
    ) if params.partial_exit_enabled else []
    partial_ma_series = _build_partial_ma_series(stock_df, partial_rules) if partial_rules else {}
    triggered_rule_priority: set[int] = set()

    triggered_profit_drawdown_ratio = math.nan
    triggered_exit_ma_value = math.nan

    for holding_days in range(1, max_holding_days + 1):
        day_idx = signal_idx + holding_days
        day_row = stock_df.iloc[day_idx]
        day_date = pd.Timestamp(day_row["date"])
        day_close = float(day_row["close"])
        day_high = float(day_row["high"])

        peak_price = max(peak_price, day_high)

        # 1) 全仓止损
        if float(day_row["low"]) <= stop_loss_price and remaining_weight > EPSILON:
            fills.append(
                TradeFill(
                    sell_date=str(day_date.date()),
                    sell_price=stop_loss_price,
                    weight=remaining_weight,
                    exit_type="stop_loss",
                    holding_days=holding_days,
                )
            )
            remaining_weight = 0.0
            break

        # 2) 分批退出（按 priority）
        if params.partial_exit_enabled and remaining_weight > EPSILON:
            for rule in partial_rules:
                if rule.priority in triggered_rule_priority:
                    continue

                if not _rule_triggered(rule, day_row, buy_price, peak_price, partial_ma_series, day_idx):
                    continue

                rule_weight = min(remaining_weight, rule.weight_ratio)
                if rule_weight <= EPSILON:
                    triggered_rule_priority.add(rule.priority)
                    continue

                fill_price = day_close
                if rule.mode == "fixed_tp" and rule.target_profit_ratio is not None:
                    fill_price = buy_price * (1.0 + rule.target_profit_ratio)
                if rule.mode == "ma_exit" and rule.ma_period is not None:
                    ma_value = partial_ma_series[rule.ma_period].iloc[day_idx]
                    if pd.notna(ma_value):
                        triggered_exit_ma_value = float(ma_value)
                if rule.mode == "profit_drawdown":
                    price_drawdown = (peak_price - day_close) / peak_price if peak_price > EPSILON else 0.0
                    triggered_profit_drawdown_ratio = price_drawdown

                fills.append(
                    TradeFill(
                        sell_date=str(day_date.date()),
                        sell_price=float(fill_price),
                        weight=float(rule_weight),
                        exit_type=rule.mode,
                        holding_days=holding_days,
                    )
                )
                remaining_weight -= rule_weight
                triggered_rule_priority.add(rule.priority)

                if remaining_weight <= EPSILON:
                    remaining_weight = 0.0
                    break

        # 3) 旧版整笔退出逻辑（仅未启用分批时生效）
        if (not params.partial_exit_enabled) and remaining_weight > EPSILON:
            if params.enable_take_profit and float(day_row["high"]) >= take_profit_price:
                fills.append(
                    TradeFill(
                        sell_date=str(day_date.date()),
                        sell_price=take_profit_price,
                        weight=remaining_weight,
                        exit_type="take_profit",
                        holding_days=holding_days,
                    )
                )
                remaining_weight = 0.0

            if remaining_weight > EPSILON and params.enable_profit_drawdown_exit:
                if peak_price >= buy_price:
                    current_profit_pct = day_close / buy_price - 1.0
                    max_profit_pct = peak_price / buy_price - 1.0
                    if max_profit_pct > 0:
                        profit_drawdown_ratio = (max_profit_pct - current_profit_pct) / max_profit_pct
                        if profit_drawdown_ratio >= params.profit_drawdown_ratio:
                            fills.append(
                                TradeFill(
                                    sell_date=str(day_date.date()),
                                    sell_price=day_close,
                                    weight=remaining_weight,
                                    exit_type="profit_drawdown_exit",
                                    holding_days=holding_days,
                                )
                            )
                            triggered_profit_drawdown_ratio = profit_drawdown_ratio
                            remaining_weight = 0.0

            if remaining_weight > EPSILON and params.enable_ma_exit and exit_ma_series is not None:
                day_exit_ma = exit_ma_series.iloc[day_idx]
                if pd.notna(day_exit_ma) and day_close < float(day_exit_ma):
                    fills.append(
                        TradeFill(
                            sell_date=str(day_date.date()),
                            sell_price=day_close,
                            weight=remaining_weight,
                            exit_type="ma_exit",
                            holding_days=holding_days,
                        )
                    )
                    triggered_exit_ma_value = float(day_exit_ma)
                    remaining_weight = 0.0

        # 4) 时间退出：从 holding_days>=N 开始持续检查
        if remaining_weight > EPSILON and holding_days >= params.time_stop_days:
            holding_return = day_close / buy_price - 1.0
            if holding_return < params.time_stop_target_ratio:
                fills.append(
                    TradeFill(
                        sell_date=str(day_date.date()),
                        sell_price=day_close,
                        weight=remaining_weight,
                        exit_type="time_exit",
                        holding_days=holding_days,
                    )
                )
                remaining_weight = 0.0

        if remaining_weight <= EPSILON:
            remaining_weight = 0.0
            break

    # 5) 数据结束处理
    if remaining_weight > EPSILON:
        if params.time_exit_mode == "strict":
            return None, "unclosed_trade"

        if params.time_exit_mode == "force_close":
            last_idx = len(stock_df) - 1
            last_row = stock_df.iloc[last_idx]
            fills.append(
                TradeFill(
                    sell_date=str(pd.Timestamp(last_row["date"]).date()),
                    sell_price=float(last_row["close"]),
                    weight=remaining_weight,
                    exit_type="force_close",
                    holding_days=last_idx - signal_idx,
                )
            )
            remaining_weight = 0.0

    if not fills:
        return None, "no_exit"

    if remaining_weight > EPSILON:
        return None, "unclosed_trade"

    fill_dicts = [asdict(fill) for fill in fills]
    total_weight = sum(fill["weight"] for fill in fill_dicts)
    if total_weight <= EPSILON:
        return None, "no_exit"

    weighted_sell_price = sum(fill["sell_price"] * fill["weight"] for fill in fill_dicts) / total_weight
    sell_date = pd.to_datetime(fill_dicts[-1]["sell_date"]).date()
    exit_type = "+".join(fill["exit_type"] for fill in fill_dicts)
    sell_day_idx = signal_idx + int(fill_dicts[-1]["holding_days"])

    actual_buy_cost = buy_price * (1.0 + params.buy_cost_ratio)
    actual_sell_value = weighted_sell_price * (1.0 - params.sell_cost_ratio)
    gross_return = weighted_sell_price / buy_price - 1.0
    net_return = actual_sell_value / actual_buy_cost - 1.0

    holding_slice = stock_df.iloc[signal_idx + 1 : sell_day_idx + 1]
    mfe = holding_slice["high"].max() / buy_price - 1.0 if not holding_slice.empty else 0.0
    mae = holding_slice["low"].min() / buy_price - 1.0 if not holding_slice.empty else 0.0

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
        "buy_date": str(buy_date.date()),
        "buy_price": buy_price,
        "sell_price": float(weighted_sell_price),
        "sell_date": sell_date,
        "exit_type": exit_type,
        "holding_days": int(fill_dicts[-1]["holding_days"]),
        "fills": fill_dicts,
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
