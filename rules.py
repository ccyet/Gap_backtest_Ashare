from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, cast

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

BREAKOUT_ENTRY_FACTORS = frozenset(
    {"trend_breakout", "volatility_contraction_breakout"}
)


def _column(stock_df: pd.DataFrame, column_name: str) -> pd.Series:
    return cast(pd.Series, stock_df[column_name])


def _is_missing_scalar(value: object) -> bool:
    return bool(pd.isna(cast(Any, value)))


def _compute_breakout_trigger_price(
    stock_df: pd.DataFrame, params: AnalysisParams
) -> pd.Series:
    if params.entry_factor == "trend_breakout":
        lookback = params.trend_breakout_lookback
    else:
        lookback = params.vcb_breakout_lookback

    if params.gap_direction == "down":
        return pd.Series(
            _column(stock_df, "low").shift(1).rolling(lookback).min(),
            index=stock_df.index,
        )
    return pd.Series(
        _column(stock_df, "high").shift(1).rolling(lookback).max(),
        index=stock_df.index,
    )


def _resolve_entry_execution(
    signal_row: pd.Series,
    params: AnalysisParams,
    direction: str,
) -> tuple[float | None, float, str | None, str | None, str]:
    entry_factor = str(signal_row.get("entry_factor", params.entry_factor))
    if entry_factor not in BREAKOUT_ENTRY_FACTORS:
        return float(signal_row["open"]), math.nan, "open", None, entry_factor

    trigger_price_raw = signal_row["entry_trigger_price"]
    trigger_price = (
        math.nan if _is_missing_scalar(trigger_price_raw) else float(trigger_price_raw)
    )
    if pd.isna(trigger_price):
        return None, math.nan, None, "entry_not_filled", entry_factor

    day_open = float(signal_row["open"])
    day_high = float(signal_row["high"])
    day_low = float(signal_row["low"])
    day_close = float(signal_row["close"])

    reference_buy_price: float | None = None
    entry_fill_type: str | None = None
    if direction == "short":
        if day_open <= trigger_price:
            reference_buy_price = day_open
            entry_fill_type = "open"
        elif day_open > trigger_price >= day_low:
            reference_buy_price = trigger_price
            entry_fill_type = "trigger"
    else:
        if day_open >= trigger_price:
            reference_buy_price = day_open
            entry_fill_type = "open"
        elif day_open < trigger_price <= day_high:
            reference_buy_price = trigger_price
            entry_fill_type = "trigger"

    if reference_buy_price is None or entry_fill_type is None:
        return None, trigger_price, None, "entry_not_filled", entry_factor

    volume = signal_row["volume"] if "volume" in signal_row else math.nan
    is_one_price_bar = day_open == day_high == day_low == day_close
    if _is_missing_scalar(volume) or float(volume) <= 0.0 or is_one_price_bar:
        return None, trigger_price, None, "locked_bar_unfillable", entry_factor

    return reference_buy_price, trigger_price, entry_fill_type, None, entry_factor


def apply_gap_filters(df: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    stock_df = df.sort_values("date").reset_index(drop=True).copy()

    stock_df["prev_close"] = _column(stock_df, "close").shift(1)
    stock_df["prev_high"] = _column(stock_df, "high").shift(1)
    stock_df["prev_low"] = _column(stock_df, "low").shift(1)
    stock_df["gap_pct_vs_prev_close"] = (
        stock_df["open"] / stock_df["prev_close"] - 1.0
    ) * 100.0

    if params.use_ma_filter:
        stock_df["fast_ma"] = pd.Series(
            _column(stock_df, "close").rolling(params.fast_ma_period).mean(),
            index=stock_df.index,
        ).shift(1)
        stock_df["slow_ma"] = pd.Series(
            _column(stock_df, "close").rolling(params.slow_ma_period).mean(),
            index=stock_df.index,
        ).shift(1)
    else:
        stock_df["fast_ma"] = math.nan
        stock_df["slow_ma"] = math.nan

    stock_df["entry_factor"] = params.entry_factor
    stock_df["entry_trigger_price"] = math.nan
    stock_df["is_contraction"] = False

    signal_mask = (
        stock_df["prev_close"].notna()
        & stock_df["prev_high"].notna()
        & stock_df["prev_low"].notna()
    )

    if params.entry_factor == "gap":
        if params.gap_direction == "up":
            if params.gap_entry_mode == "open_vs_prev_close_threshold":
                signal_mask &= stock_df["gap_pct_vs_prev_close"] >= params.gap_pct
            else:
                signal_mask &= stock_df["open"] > stock_df["prev_high"] * (
                    1.0 + params.gap_ratio
                )
            signal_mask &= (
                stock_df["gap_pct_vs_prev_close"] <= params.max_gap_filter_pct
            )
            if params.use_ma_filter:
                signal_mask &= stock_df["fast_ma"].notna() & stock_df["slow_ma"].notna()
                signal_mask &= stock_df["open"] > stock_df["fast_ma"]
                signal_mask &= stock_df["open"] > stock_df["slow_ma"]
        else:
            if params.gap_entry_mode == "open_vs_prev_close_threshold":
                signal_mask &= stock_df["gap_pct_vs_prev_close"] <= -params.gap_pct
            else:
                signal_mask &= stock_df["open"] < stock_df["prev_low"] * (
                    1.0 - params.gap_ratio
                )
            signal_mask &= (
                stock_df["gap_pct_vs_prev_close"] >= -params.max_gap_filter_pct
            )
            if params.use_ma_filter:
                signal_mask &= stock_df["fast_ma"].notna() & stock_df["slow_ma"].notna()
                signal_mask &= stock_df["open"] < stock_df["fast_ma"]
                signal_mask &= stock_df["open"] < stock_df["slow_ma"]
    else:
        stock_df["entry_trigger_price"] = _compute_breakout_trigger_price(
            stock_df, params
        )
        if params.entry_factor == "volatility_contraction_breakout":
            prior_range = _column(stock_df, "high").shift(1) - _column(
                stock_df, "low"
            ).shift(1)
            contraction_floor = prior_range.rolling(params.vcb_range_lookback).min()
            stock_df["is_contraction"] = prior_range.eq(contraction_floor)
            signal_mask &= stock_df["is_contraction"]
        signal_mask &= stock_df["entry_trigger_price"].notna()
        if params.use_ma_filter:
            signal_mask &= stock_df["fast_ma"].notna() & stock_df["slow_ma"].notna()
            if params.gap_direction == "down":
                signal_mask &= stock_df["open"] < stock_df["fast_ma"]
                signal_mask &= stock_df["open"] < stock_df["slow_ma"]
            else:
                signal_mask &= stock_df["open"] > stock_df["fast_ma"]
                signal_mask &= stock_df["open"] > stock_df["slow_ma"]

    stock_df["is_signal"] = signal_mask
    return stock_df


def _build_partial_ma_series(
    stock_df: pd.DataFrame, rules: list[PartialExitRule]
) -> dict[int, pd.Series]:
    periods = {
        rule.ma_period
        for rule in rules
        if rule.mode == "ma_exit" and rule.ma_period is not None
    }
    close_series = _column(stock_df, "close")
    return {
        period: pd.Series(close_series.rolling(period).mean(), index=stock_df.index)
        for period in periods
    }


def _apply_entry_slippage(
    reference_price: float, params: AnalysisParams, direction: str
) -> float:
    if direction == "short":
        return reference_price * (1.0 - params.sell_slippage_ratio)
    return reference_price * (1.0 + params.buy_slippage_ratio)


def _apply_exit_slippage(
    reference_price: float, params: AnalysisParams, direction: str
) -> float:
    if direction == "short":
        return reference_price * (1.0 + params.buy_slippage_ratio)
    return reference_price * (1.0 - params.sell_slippage_ratio)


def _position_profit_ratio(
    reference_entry_price: float, mark_price: float, direction: str
) -> float:
    if direction == "short":
        return (reference_entry_price - mark_price) / reference_entry_price
    return mark_price / reference_entry_price - 1.0


def _total_trade_profit_ratio(
    realized_profit_ratio: float,
    remaining_weight: float,
    reference_entry_price: float,
    mark_price: float,
    direction: str,
) -> float:
    return realized_profit_ratio + remaining_weight * _position_profit_ratio(
        reference_entry_price, mark_price, direction
    )


def _update_peak_total_profit_ratio(
    peak_total_profit_ratio: float,
    realized_profit_ratio: float,
    remaining_weight: float,
    reference_entry_price: float,
    favorable_price: float,
    direction: str,
) -> float:
    candidate_peak = _total_trade_profit_ratio(
        realized_profit_ratio,
        remaining_weight,
        reference_entry_price,
        favorable_price,
        direction,
    )
    return max(peak_total_profit_ratio, candidate_peak)


def _compute_profit_drawdown_ratio(
    peak_total_profit_ratio: float,
    current_total_profit_ratio: float,
) -> float:
    if peak_total_profit_ratio <= EPSILON:
        return math.nan
    return (
        peak_total_profit_ratio - current_total_profit_ratio
    ) / peak_total_profit_ratio


def _rule_triggered(
    rule: PartialExitRule,
    day_row: pd.Series,
    buy_price: float,
    ma_series: dict[int, pd.Series],
    day_idx: int,
    direction: str,
    peak_total_profit_ratio: float,
    current_total_profit_ratio: float,
) -> bool:
    day_close = float(day_row["close"])
    day_high = float(day_row["high"])
    day_low = float(day_row["low"])

    if rule.mode == "fixed_tp":
        if rule.target_profit_ratio is None:
            return False
        if direction == "short":
            return day_low <= buy_price * (1.0 - rule.target_profit_ratio)
        return day_high >= buy_price * (1.0 + rule.target_profit_ratio)

    if rule.mode == "ma_exit":
        if rule.ma_period is None:
            return False
        day_ma = ma_series.get(rule.ma_period)
        if day_ma is None:
            return False
        ma_value = day_ma.iloc[day_idx]
        if pd.isna(ma_value):
            return False
        return (
            day_close > float(ma_value)
            if direction == "short"
            else day_close < float(ma_value)
        )

    if rule.mode == "profit_drawdown":
        if (
            rule.drawdown_ratio is None
            or rule.min_profit_to_activate_drawdown_ratio is None
        ):
            return False
        if peak_total_profit_ratio < rule.min_profit_to_activate_drawdown_ratio:
            return False
        profit_drawdown = _compute_profit_drawdown_ratio(
            peak_total_profit_ratio, current_total_profit_ratio
        )
        if pd.isna(profit_drawdown):
            return False
        return profit_drawdown >= rule.drawdown_ratio

    return False


def simulate_trade(
    stock_df: pd.DataFrame,
    signal_idx: int,
    params: AnalysisParams,
    direction: str = "long",
) -> tuple[dict[str, Any] | None, str | None]:
    if direction not in {"long", "short"}:
        return None, "invalid_direction"

    if signal_idx + params.time_stop_days >= len(stock_df):
        return None, "insufficient_future"

    signal_row = stock_df.iloc[signal_idx]
    (
        reference_buy_price,
        entry_trigger_price,
        entry_fill_type,
        entry_skip_reason,
        entry_factor,
    ) = _resolve_entry_execution(signal_row, params, direction)
    if reference_buy_price is None:
        return None, entry_skip_reason

    buy_price = _apply_entry_slippage(reference_buy_price, params, direction)
    buy_date = pd.Timestamp(signal_row["date"])

    stop_loss_price = (
        reference_buy_price * (1.0 + params.stop_loss_ratio)
        if direction == "short"
        else reference_buy_price * (1.0 - params.stop_loss_ratio)
    )
    take_profit_price = (
        reference_buy_price * (1.0 - params.take_profit_ratio)
        if direction == "short"
        else reference_buy_price * (1.0 + params.take_profit_ratio)
    )
    exit_ma_series = (
        pd.Series(
            _column(stock_df, "close").rolling(params.exit_ma_period).mean(),
            index=stock_df.index,
        )
        if params.enable_ma_exit
        else None
    )

    fills: list[TradeFill] = []
    remaining_weight = 1.0
    max_holding_days = len(stock_df) - signal_idx - 1
    realized_trigger_profit_ratio = 0.0
    peak_total_profit_ratio = 0.0

    partial_rules = (
        sorted(
            [rule for rule in params.partial_exit_rules if rule.enabled],
            key=lambda item: item.priority,
        )
        if params.partial_exit_enabled
        else []
    )
    partial_ma_series = (
        _build_partial_ma_series(stock_df, partial_rules) if partial_rules else {}
    )
    triggered_rule_priority: set[int] = set()

    triggered_profit_drawdown_ratio = math.nan
    triggered_exit_ma_value = math.nan

    for holding_days in range(1, max_holding_days + 1):
        day_idx = signal_idx + holding_days
        day_row = stock_df.iloc[day_idx]
        day_date = pd.Timestamp(day_row["date"])
        day_close = float(day_row["close"])
        day_high = float(day_row["high"])
        day_low = float(day_row["low"])
        favorable_price = day_low if direction == "short" else day_high

        # 1) 全仓止损
        stop_hit = (
            day_high >= stop_loss_price
            if direction == "short"
            else day_low <= stop_loss_price
        )
        if stop_hit and remaining_weight > EPSILON:
            fills.append(
                TradeFill(
                    str(day_date.date()),
                    _apply_exit_slippage(stop_loss_price, params, direction),
                    remaining_weight,
                    "stop_loss",
                    holding_days,
                )
            )
            remaining_weight = 0.0
            break

        # 2) 分批退出
        if params.partial_exit_enabled and remaining_weight > EPSILON:
            for rule in partial_rules:
                if rule.priority in triggered_rule_priority:
                    continue
                current_peak_total_profit_ratio = _update_peak_total_profit_ratio(
                    peak_total_profit_ratio,
                    realized_trigger_profit_ratio,
                    remaining_weight,
                    reference_buy_price,
                    favorable_price,
                    direction,
                )
                current_total_profit_ratio = _total_trade_profit_ratio(
                    realized_trigger_profit_ratio,
                    remaining_weight,
                    reference_buy_price,
                    day_close,
                    direction,
                )
                if not _rule_triggered(
                    rule,
                    day_row,
                    reference_buy_price,
                    partial_ma_series,
                    day_idx,
                    direction,
                    current_peak_total_profit_ratio,
                    current_total_profit_ratio,
                ):
                    continue

                rule_weight = min(remaining_weight, rule.weight_ratio)
                if rule_weight <= EPSILON:
                    triggered_rule_priority.add(rule.priority)
                    continue

                fill_price = _apply_exit_slippage(day_close, params, direction)
                reference_fill_price = day_close
                if rule.mode == "fixed_tp" and rule.target_profit_ratio is not None:
                    target_price = (
                        reference_buy_price * (1.0 - rule.target_profit_ratio)
                        if direction == "short"
                        else reference_buy_price * (1.0 + rule.target_profit_ratio)
                    )
                    reference_fill_price = target_price
                    fill_price = _apply_exit_slippage(target_price, params, direction)
                if rule.mode == "ma_exit" and rule.ma_period is not None:
                    ma_value = partial_ma_series[rule.ma_period].iloc[day_idx]
                    if pd.notna(ma_value):
                        triggered_exit_ma_value = float(ma_value)
                if rule.mode == "profit_drawdown":
                    triggered_profit_drawdown_ratio = _compute_profit_drawdown_ratio(
                        current_peak_total_profit_ratio, current_total_profit_ratio
                    )

                fills.append(
                    TradeFill(
                        str(day_date.date()),
                        float(fill_price),
                        float(rule_weight),
                        rule.mode,
                        holding_days,
                    )
                )
                remaining_weight -= rule_weight
                realized_trigger_profit_ratio += rule_weight * _position_profit_ratio(
                    reference_buy_price, reference_fill_price, direction
                )
                triggered_rule_priority.add(rule.priority)
                peak_total_profit_ratio = _update_peak_total_profit_ratio(
                    current_peak_total_profit_ratio,
                    realized_trigger_profit_ratio,
                    remaining_weight,
                    reference_buy_price,
                    favorable_price,
                    direction,
                )
                if remaining_weight <= EPSILON:
                    remaining_weight = 0.0
                    break
        peak_total_profit_ratio = _update_peak_total_profit_ratio(
            peak_total_profit_ratio,
            realized_trigger_profit_ratio,
            remaining_weight,
            reference_buy_price,
            favorable_price,
            direction,
        )

        # 3) 旧版整笔退出
        if (not params.partial_exit_enabled) and remaining_weight > EPSILON:
            tp_hit = (
                day_low <= take_profit_price
                if direction == "short"
                else day_high >= take_profit_price
            )
            if params.enable_take_profit and tp_hit:
                fills.append(
                    TradeFill(
                        str(day_date.date()),
                        _apply_exit_slippage(take_profit_price, params, direction),
                        remaining_weight,
                        "take_profit",
                        holding_days,
                    )
                )
                remaining_weight = 0.0

            if remaining_weight > EPSILON and params.enable_profit_drawdown_exit:
                current_total_profit_ratio = _total_trade_profit_ratio(
                    realized_trigger_profit_ratio,
                    remaining_weight,
                    reference_buy_price,
                    day_close,
                    direction,
                )
                profit_drawdown_ratio = _compute_profit_drawdown_ratio(
                    peak_total_profit_ratio, current_total_profit_ratio
                )
                if (
                    peak_total_profit_ratio > EPSILON
                    and not pd.isna(profit_drawdown_ratio)
                    and profit_drawdown_ratio >= params.profit_drawdown_ratio
                ):
                    fills.append(
                        TradeFill(
                            str(day_date.date()),
                            _apply_exit_slippage(day_close, params, direction),
                            remaining_weight,
                            "profit_drawdown_exit",
                            holding_days,
                        )
                    )
                    triggered_profit_drawdown_ratio = profit_drawdown_ratio
                    remaining_weight = 0.0

            if (
                remaining_weight > EPSILON
                and params.enable_ma_exit
                and exit_ma_series is not None
            ):
                day_exit_ma = exit_ma_series.iloc[day_idx]
                if pd.notna(day_exit_ma):
                    exit_hit = (
                        day_close > float(day_exit_ma)
                        if direction == "short"
                        else day_close < float(day_exit_ma)
                    )
                    if exit_hit:
                        fills.append(
                            TradeFill(
                                str(day_date.date()),
                                _apply_exit_slippage(day_close, params, direction),
                                remaining_weight,
                                "ma_exit",
                                holding_days,
                            )
                        )
                        triggered_exit_ma_value = float(day_exit_ma)
                        remaining_weight = 0.0

        # 4) 时间退出
        if remaining_weight > EPSILON and holding_days >= params.time_stop_days:
            holding_return = (
                (reference_buy_price - day_close) / reference_buy_price
                if direction == "short"
                else day_close / reference_buy_price - 1.0
            )
            if holding_return < params.time_stop_target_ratio:
                fills.append(
                    TradeFill(
                        str(day_date.date()),
                        _apply_exit_slippage(day_close, params, direction),
                        remaining_weight,
                        "time_exit",
                        holding_days,
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
            reference_force_close = float(last_row["close"])
            fills.append(
                TradeFill(
                    str(pd.Timestamp(last_row["date"]).date()),
                    _apply_exit_slippage(reference_force_close, params, direction),
                    remaining_weight,
                    "force_close",
                    last_idx - signal_idx,
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

    weighted_sell_price = (
        sum(fill["sell_price"] * fill["weight"] for fill in fill_dicts) / total_weight
    )
    sell_date = pd.to_datetime(fill_dicts[-1]["sell_date"]).date()
    exit_type = "+".join(fill["exit_type"] for fill in fill_dicts)
    sell_day_idx = signal_idx + int(fill_dicts[-1]["holding_days"])

    actual_buy_cost = buy_price * (1.0 + params.buy_cost_ratio)
    actual_sell_value = weighted_sell_price * (1.0 - params.sell_cost_ratio)
    gross_return = (
        (buy_price - weighted_sell_price) / buy_price
        if direction == "short"
        else weighted_sell_price / buy_price - 1.0
    )
    net_return = (
        actual_sell_value / actual_buy_cost - 1.0
        if direction == "long"
        else (buy_price - actual_sell_value) / actual_buy_cost
    )

    holding_slice = stock_df.iloc[signal_idx + 1 : sell_day_idx + 1]
    if holding_slice.empty:
        mfe = 0.0
        mae = 0.0
    elif direction == "short":
        mfe = (buy_price - holding_slice["low"].min()) / buy_price
        mae = (buy_price - holding_slice["high"].max()) / buy_price
    else:
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
        "volume": float(signal_row["volume"])
        if pd.notna(signal_row["volume"])
        else math.nan,
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
        "exit_ma_value": float(triggered_exit_ma_value)
        if pd.notna(triggered_exit_ma_value)
        else math.nan,
        "profit_drawdown_ratio": float(triggered_profit_drawdown_ratio) * 100.0
        if pd.notna(triggered_profit_drawdown_ratio)
        else math.nan,
        "entry_factor": entry_factor,
        "entry_trigger_price": float(entry_trigger_price)
        if pd.notna(entry_trigger_price)
        else math.nan,
        "entry_fill_type": entry_fill_type,
    }, None
