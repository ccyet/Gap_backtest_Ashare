from __future__ import annotations

from collections import Counter

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
        "skipped_time_target_met": 0,
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
            trade, skip_reason = simulate_trade(enriched, signal_idx, params)
            if trade is None:
                if skip_reason == "insufficient_future":
                    stats["skipped_insufficient_future"] += 1
                elif skip_reason == "time_target_met":
                    stats["skipped_time_target_met"] += 1
                else:
                    stats["skipped_no_exit"] += 1
                continue

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

    if strategy_df.empty:
        market_dates = []
        if not all_data.empty:
            market_dates = sorted(
                timestamp
                for timestamp in pd.to_datetime(all_data["date"]).dropna().unique().tolist()
                if start_ts <= pd.Timestamp(timestamp) <= end_ts
            )
        if not market_dates:
            market_dates = [start_ts]

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

    last_exit_ts = pd.to_datetime(strategy_df["sell_date"]).max()
    curve_end = max(end_ts, last_exit_ts)

    market_dates = []
    if not all_data.empty:
        market_dates = sorted(
            timestamp
            for timestamp in pd.to_datetime(all_data["date"]).dropna().unique().tolist()
            if start_ts <= pd.Timestamp(timestamp) <= curve_end
        )
    if not market_dates:
        market_dates = list(pd.date_range(start=start_ts, end=curve_end, freq="D"))

    events = strategy_df.sort_values("sell_date").to_dict("records")
    event_index = 0
    current_nav = 1.0
    curve_records: list[dict] = []

    for market_date in market_dates:
        event_label = ""
        event_trade_no = pd.NA
        event_stock_code = ""

        while event_index < len(events) and pd.Timestamp(events[event_index]["sell_date"]) == pd.Timestamp(market_date):
            current_nav = float(events[event_index]["nav_after_trade"])
            event_label = str(events[event_index]["exit_type"])
            event_trade_no = int(events[event_index]["trade_no"])
            event_stock_code = str(events[event_index]["stock_code"])
            event_index += 1

        curve_records.append(
            {
                "date": pd.Timestamp(market_date),
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
