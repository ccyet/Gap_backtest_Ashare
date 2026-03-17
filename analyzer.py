from __future__ import annotations

from collections import Counter
import json

import pandas as pd

from models import AnalysisParams
from rules import apply_gap_filters, simulate_trade


BASE_DETAIL_COLUMNS = [
    "date",
    "stock_code",
    "prev_close",
    "prev_high",
    "prev_low",
    "open",
    "close",
    "volume",
    "gap_pct_vs_prev_close",
    "buy_price",
    "sell_price",
    "sell_date",
    "exit_type",
    "holding_days",
    "gross_return_pct",
    "net_return_pct",
    "win_flag",
    "mfe_pct",
    "mae_pct",
    "max_profit_pct",
    "exit_ma_value",
    "profit_drawdown_ratio",
    "fills",
    "fill_count",
    "fill_detail_json",
]

DETAIL_COLUMNS = BASE_DETAIL_COLUMNS + [
    "trade_no",
    "nav_before_trade",
    "nav_after_trade",
]

DAILY_COLUMNS = [
    "date",
    "sample_count",
    "win_count",
    "lose_count",
    "win_rate_pct",
    "avg_net_return_pct",
    "median_net_return_pct",
    "avg_holding_days",
]

EQUITY_COLUMNS = [
    "date",
    "net_value",
    "drawdown_pct",
    "trade_no",
    "stock_code",
    "event",
]


def _empty_scan_stats() -> dict[str, int]:
    return {
        "signal_count": 0,
        "closed_trade_candidates": 0,
        "skipped_insufficient_future": 0,
        "skipped_unclosed_trade": 0,
        "skipped_no_exit": 0,
    }


def _empty_strategy_stats() -> dict[str, float]:
    return {
        "executed_trades": 0,
        "skipped_overlapping_position": 0,
        "win_count": 0,
        "lose_count": 0,
        "strategy_win_rate_pct": 0.0,
        "final_net_value": 1.0,
        "total_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_holding_days": 0.0,
        "avg_mfe_pct": 0.0,
        "avg_mae_pct": 0.0,
        "profit_risk_ratio": 0.0,
        "trade_return_volatility_pct": 0.0,
    }


def scan_trade_candidates(all_data: pd.DataFrame, params: AnalysisParams) -> tuple[pd.DataFrame, dict[str, int]]:
    if all_data.empty:
        return pd.DataFrame(columns=BASE_DETAIL_COLUMNS), _empty_scan_stats()

    start_ts = pd.to_datetime(params.start_date)
    end_ts = pd.to_datetime(params.end_date)

    detail_records: list[dict] = []
    stats = Counter()

    for _, stock_df in all_data.groupby("stock_code", sort=True):
        enriched = apply_gap_filters(stock_df, params)
        signal_mask = enriched["is_signal"] & enriched["date"].between(start_ts, end_ts)
        signal_indices = enriched.index[signal_mask].tolist()
        stats["signal_count"] += len(signal_indices)

        for signal_idx in signal_indices:
            direction = "long" if params.gap_direction == "up" else "short"
            trade, skip_reason = simulate_trade(enriched, signal_idx, params, direction=direction)
            if trade is None:
                if skip_reason == "insufficient_future":
                    stats["skipped_insufficient_future"] += 1
                elif skip_reason == "unclosed_trade":
                    stats["skipped_unclosed_trade"] += 1
                else:
                    stats["skipped_no_exit"] += 1
                continue

            fills = trade.get("fills", [])
            total_weight = sum(float(fill.get("weight", 0.0)) for fill in fills)
            if total_weight <= 0:
                stats["skipped_no_exit"] += 1
                continue
            trade["sell_price"] = sum(float(fill["sell_price"]) * float(fill["weight"]) for fill in fills) / total_weight
            trade["sell_date"] = fills[-1]["sell_date"]
            trade["exit_type"] = "+".join(str(fill["exit_type"]) for fill in fills)
            trade["gross_return_pct"] = (trade["sell_price"] / float(trade["buy_price"]) - 1.0) * 100.0
            trade["fill_count"] = len(fills)
            trade["fill_detail_json"] = json.dumps(fills, ensure_ascii=False)
            detail_records.append(trade)
            stats["closed_trade_candidates"] += 1

    detail_df = pd.DataFrame(detail_records, columns=BASE_DETAIL_COLUMNS)
    if detail_df.empty:
        return detail_df, {**_empty_scan_stats(), **dict(stats)}

    detail_df = detail_df.sort_values(["date", "stock_code", "sell_date"]).reset_index(drop=True)
    return detail_df, {**_empty_scan_stats(), **dict(stats)}


def build_strategy_trades(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if candidate_df.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS), _empty_strategy_stats()

    strategy_records: list[dict] = []
    stats = Counter()
    last_sell_date: pd.Timestamp | None = None
    current_nav = 1.0
    trade_no = 0

    sorted_candidates = candidate_df.sort_values(["date", "stock_code", "sell_date"]).reset_index(drop=True)

    for record in sorted_candidates.to_dict("records"):
        buy_date = pd.to_datetime(record["date"])
        sell_date = pd.to_datetime(record["sell_date"])

        if last_sell_date is not None and buy_date <= last_sell_date:
            stats["skipped_overlapping_position"] += 1
            continue

        trade_no += 1
        nav_before_trade = current_nav
        nav_after_trade = nav_before_trade * (1.0 + float(record["net_return_pct"]) / 100.0)

        strategy_record = {
            **record,
            "trade_no": trade_no,
            "nav_before_trade": nav_before_trade,
            "nav_after_trade": nav_after_trade,
        }
        strategy_records.append(strategy_record)

        current_nav = nav_after_trade
        last_sell_date = sell_date
        stats["executed_trades"] += 1

    strategy_df = pd.DataFrame(strategy_records, columns=DETAIL_COLUMNS)
    if strategy_df.empty:
        return strategy_df, {**_empty_strategy_stats(), **dict(stats)}

    stats["win_count"] = int(strategy_df["win_flag"].sum())
    stats["lose_count"] = int(len(strategy_df) - stats["win_count"])
    stats["strategy_win_rate_pct"] = float(strategy_df["win_flag"].mean() * 100.0)
    stats["final_net_value"] = float(strategy_df["nav_after_trade"].iloc[-1])
    stats["total_return_pct"] = (stats["final_net_value"] - 1.0) * 100.0
    stats["avg_holding_days"] = float(strategy_df["holding_days"].mean())
    stats["avg_mfe_pct"] = float(strategy_df["mfe_pct"].mean())
    stats["avg_mae_pct"] = float(strategy_df["mae_pct"].mean())
    if stats["avg_mae_pct"] == 0:
        stats["profit_risk_ratio"] = 0.0
    else:
        stats["profit_risk_ratio"] = stats["avg_mfe_pct"] / abs(stats["avg_mae_pct"])
    stats["trade_return_volatility_pct"] = float(strategy_df["net_return_pct"].std(ddof=0))

    return strategy_df, {**_empty_strategy_stats(), **dict(stats)}


def build_equity_curve(all_data: pd.DataFrame, strategy_df: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    start_ts = pd.to_datetime(params.start_date)
    end_ts = pd.to_datetime(params.end_date)

    market_dates: list[pd.Timestamp]
    if not all_data.empty:
        market_dates = sorted(
            timestamp
            for timestamp in pd.to_datetime(all_data["date"]).dropna().unique().tolist()
            if start_ts <= pd.Timestamp(timestamp) <= end_ts
        )
    else:
        market_dates = [start_ts]

    if not market_dates:
        market_dates = [start_ts]

    if strategy_df.empty:
        equity_df = pd.DataFrame(
            {
                "date": market_dates,
                "net_value": 1.0,
                "drawdown_pct": 0.0,
                "trade_no": pd.NA,
                "stock_code": "",
                "event": "",
            }
        )
        return equity_df[EQUITY_COLUMNS]

    # 价格查找表，用于持仓期间逐日盯市净值
    close_lookup = {}
    if not all_data.empty:
        for row in all_data[["stock_code", "date", "close"]].dropna().itertuples(index=False):
            close_lookup[(str(row.stock_code), pd.Timestamp(row.date).normalize())] = float(row.close)

    trades = strategy_df.sort_values("date").to_dict("records")
    trade_index = 0
    active_trade: dict | None = None
    active_last_close: float | None = None

    current_nav = 1.0
    curve_records: list[dict] = []

    buy_cost_ratio = params.buy_cost_ratio
    sell_cost_ratio = params.sell_cost_ratio
    direction = "long" if params.gap_direction == "up" else "short"

    for market_date in market_dates:
        current_date = pd.Timestamp(market_date).normalize()
        event_label = ""
        event_trade_no = pd.NA
        event_stock_code = ""

        if active_trade is None and trade_index < len(trades):
            next_trade = trades[trade_index]
            next_buy_date = pd.Timestamp(next_trade["date"]).normalize()
            if current_date >= next_buy_date:
                active_trade = next_trade
                active_last_close = None
                event_label = "buy"
                event_trade_no = int(active_trade["trade_no"])
                event_stock_code = str(active_trade["stock_code"])

        if active_trade is not None:
            stock_code = str(active_trade["stock_code"])
            buy_price = float(active_trade["buy_price"])
            nav_before_trade = float(active_trade["nav_before_trade"])
            sell_date = pd.Timestamp(active_trade["sell_date"]).normalize()

            day_close = close_lookup.get((stock_code, current_date), active_last_close)
            if day_close is not None:
                active_last_close = day_close

            if current_date < sell_date:
                if day_close is not None:
                    if direction == "short":
                        unrealized_net_return = (buy_price - day_close * (1.0 - sell_cost_ratio)) / (buy_price * (1.0 + buy_cost_ratio))
                    else:
                        unrealized_net_return = (day_close * (1.0 - sell_cost_ratio)) / (buy_price * (1.0 + buy_cost_ratio)) - 1.0
                    current_nav = nav_before_trade * (1.0 + unrealized_net_return)
            else:
                current_nav = float(active_trade["nav_after_trade"])
                event_label = str(active_trade["exit_type"]) if not event_label else f"{event_label}+{active_trade['exit_type']}"
                event_trade_no = int(active_trade["trade_no"])
                event_stock_code = stock_code
                active_trade = None
                active_last_close = None
                trade_index += 1

        curve_records.append(
            {
                "date": pd.Timestamp(current_date),
                "net_value": current_nav,
                "trade_no": event_trade_no,
                "stock_code": event_stock_code,
                "event": event_label,
            }
        )

    equity_df = pd.DataFrame(curve_records, columns=["date", "net_value", "trade_no", "stock_code", "event"])
    equity_df["drawdown_pct"] = (equity_df["net_value"] / equity_df["net_value"].cummax() - 1.0) * 100.0
    return equity_df[EQUITY_COLUMNS]


def analyze_all_stocks(
    all_data: pd.DataFrame,
    params: AnalysisParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    candidate_df, scan_stats = scan_trade_candidates(all_data, params)
    strategy_df, strategy_stats = build_strategy_trades(candidate_df)
    daily_df = build_daily_summary(strategy_df)
    equity_df = build_equity_curve(all_data, strategy_df, params)

    if not equity_df.empty:
        strategy_stats["max_drawdown_pct"] = abs(float(equity_df["drawdown_pct"].min()))

    combined_stats: dict[str, float] = {**scan_stats, **strategy_stats}
    return strategy_df, daily_df, equity_df, combined_stats


def build_daily_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    daily_df = (
        detail_df.groupby("date", as_index=False)
        .agg(
            sample_count=("stock_code", "size"),
            win_count=("win_flag", "sum"),
            avg_net_return_pct=("net_return_pct", "mean"),
            median_net_return_pct=("net_return_pct", "median"),
            avg_holding_days=("holding_days", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily_df["lose_count"] = daily_df["sample_count"] - daily_df["win_count"]
    daily_df["win_rate_pct"] = daily_df["win_count"] / daily_df["sample_count"] * 100.0

    ordered = daily_df[DAILY_COLUMNS].copy()
    ordered["sample_count"] = ordered["sample_count"].astype(int)
    ordered["win_count"] = ordered["win_count"].astype(int)
    ordered["lose_count"] = ordered["lose_count"].astype(int)
    return ordered
