from __future__ import annotations

from datetime import timedelta
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import analyze_all_stocks, run_parameter_scan
from data_loader import (
    describe_file_source,
    describe_tables,
    list_candidate_tables,
    load_market_data,
)
from exporter import export_to_excel_bytes
from models import (
    AnalysisParams,
    GAP_ENTRY_MODES,
    ParamScanAxis,
    ParamScanConfig,
    PartialExitRule,
    SCAN_FIELD_CASTERS,
    SCAN_METRICS,
    normalize_column_overrides,
    normalize_stock_codes,
    parse_scan_values,
    validate_params,
)


st.set_page_config(layout="wide", page_title="Gap_test 回测系统")

RESULT_STATE_KEYS = [
    "detail_df",
    "daily_df",
    "equity_df",
    "stats",
    "scan_df",
    "scan_metric",
    "scan_axis_fields",
    "best_scan_overrides",
    "excel_bytes",
    "download_name",
]

DETAIL_PRICE_COLUMNS = [
    "prev_close",
    "prev_high",
    "prev_low",
    "open",
    "close",
    "buy_price",
    "sell_price",
    "exit_ma_value",
]
DETAIL_PERCENT_COLUMNS = [
    "gap_pct_vs_prev_close",
    "gross_return_pct",
    "net_return_pct",
    "mfe_pct",
    "mae_pct",
    "max_profit_pct",
    "profit_drawdown_ratio",
]
DETAIL_NAV_COLUMNS = ["nav_before_trade", "nav_after_trade"]
SUMMARY_PERCENT_COLUMNS = [
    "win_rate_pct",
    "avg_net_return_pct",
    "median_net_return_pct",
]
EQUITY_PERCENT_COLUMNS = ["drawdown_pct"]
SCAN_FIELD_LABELS = {
    "gap_pct": "跳空幅度",
    "max_gap_filter_pct": "最大高开/低开过滤",
    "time_stop_days": "最多持有天数",
    "time_stop_target_pct": "时间退出收益阈值",
    "stop_loss_pct": "全仓止损",
    "take_profit_pct": "固定止盈",
    "profit_drawdown_pct": "盈利回撤",
    "exit_ma_period": "离场均线周期",
    "buy_slippage_pct": "买入滑点",
    "sell_slippage_pct": "卖出滑点",
}
SCAN_METRIC_LABELS = {
    "signal_count": "信号数",
    "closed_trade_candidates": "候选平仓交易数",
    "executed_trades": "实际执行交易数",
    "strategy_win_rate_pct": "策略胜率",
    "total_return_pct": "总收益率",
    "max_drawdown_pct": "最大回撤",
    "final_net_value": "最终净值",
    "avg_holding_days": "平均持有天数",
    "profit_risk_ratio": "收益风险比",
    "trade_return_volatility_pct": "单笔收益波动率",
}
DETAIL_COLUMN_LABELS = {
    "date": "信号日期",
    "stock_code": "股票代码",
    "prev_close": "前收盘价",
    "prev_high": "前最高价",
    "prev_low": "前最低价",
    "open": "开盘价",
    "close": "当日收盘价",
    "volume": "成交量",
    "gap_pct_vs_prev_close": "相对昨收跳空幅度",
    "buy_date": "买入日期",
    "buy_price": "买入价",
    "sell_price": "卖出均价",
    "sell_date": "卖出日期",
    "exit_type": "退出方式",
    "holding_days": "持有天数",
    "gross_return_pct": "毛收益率",
    "net_return_pct": "净收益率",
    "win_flag": "是否盈利",
    "mfe_pct": "最大有利波动",
    "mae_pct": "最大不利波动",
    "max_profit_pct": "最大浮盈",
    "exit_ma_value": "离场均线值",
    "profit_drawdown_ratio": "利润回撤比例",
    "fill_count": "成交批次数",
    "fill_detail_json": "成交明细",
    "nav_before_trade": "交易前净值",
    "nav_after_trade": "交易后净值",
}
SCAN_COLUMN_LABELS = {
    "scan_id": "扫描编号",
    "rank": "排名",
    "signal_count": "信号数",
    "closed_trade_candidates": "候选平仓交易数",
    "executed_trades": "实际执行交易数",
    "strategy_win_rate_pct": "策略胜率",
    "total_return_pct": "总收益率",
    "max_drawdown_pct": "最大回撤",
    "final_net_value": "最终净值",
    "avg_holding_days": "平均持有天数",
    "profit_risk_ratio": "收益风险比",
    "trade_return_volatility_pct": "单笔收益波动率",
}
SUMMARY_COLUMN_LABELS = {
    "date": "开仓日期",
    "signal_count": "信号数",
    "executed_trades": "实际执行交易数",
    "win_rate_pct": "胜率",
    "avg_net_return_pct": "平均净收益率",
    "median_net_return_pct": "净收益率中位数",
    "avg_holding_days": "平均持有天数",
}
EQUITY_COLUMN_LABELS = {
    "date": "日期",
    "net_value": "净值",
    "drawdown_pct": "回撤",
}
UPDATE_LOG_COLUMN_LABELS = {
    "symbol": "股票代码",
    "adjust": "复权方式",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "rows": "更新行数",
    "updated_at": "更新时间",
    "status": "状态",
    "error_message": "错误信息",
}


def dataframe_stretch(
    data: Any,
    *,
    hide_index: bool = False,
    column_config: Any = None,
    height: int | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "hide_index": hide_index,
        "use_container_width": True,
    }
    if column_config is not None:
        kwargs["column_config"] = column_config
    if height is not None:
        kwargs["height"] = height
    st.dataframe(data, **kwargs)


def form_submit_button_stretch(label: str) -> bool:
    return st.form_submit_button(label)


def clear_result_state() -> None:
    for key in RESULT_STATE_KEYS:
        st.session_state.pop(key, None)


def build_data_format_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "列名建议": "交易日期",
                "是否必填": "必填",
                "说明": "交易日，一行代表一只股票在一个交易日的数据",
                "示例": "2026-03-13",
            },
            {
                "列名建议": "股票代码",
                "是否必填": "必填",
                "说明": "股票唯一标识",
                "示例": "000001.SZ",
            },
            {
                "列名建议": "开盘价",
                "是否必填": "必填",
                "说明": "当天开盘价",
                "示例": "10.52",
            },
            {
                "列名建议": "最高价",
                "是否必填": "必填",
                "说明": "当天最高价",
                "示例": "10.88",
            },
            {
                "列名建议": "最低价",
                "是否必填": "必填",
                "说明": "当天最低价",
                "示例": "10.31",
            },
            {
                "列名建议": "收盘价",
                "是否必填": "必填",
                "说明": "当天收盘价",
                "示例": "10.66",
            },
            {
                "列名建议": "成交量",
                "是否必填": "选填",
                "说明": "当天成交量，不填也可以分析",
                "示例": "1256300",
            },
        ]
    )


def build_sample_input_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "交易日期": "2026-03-10",
                "股票代码": "000001.SZ",
                "开盘价": 10.20,
                "最高价": 10.45,
                "最低价": 10.08,
                "收盘价": 10.32,
                "成交量": 1856200,
            },
            {
                "交易日期": "2026-03-11",
                "股票代码": "000001.SZ",
                "开盘价": 10.58,
                "最高价": 10.90,
                "最低价": 10.50,
                "收盘价": 10.84,
                "成交量": 2365400,
            },
            {
                "交易日期": "2026-03-10",
                "股票代码": "600000.SH",
                "开盘价": 8.85,
                "最高价": 8.93,
                "最低价": 8.72,
                "收盘价": 8.80,
                "成交量": 3124500,
            },
        ]
    )


def build_template_note() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "说明": [
                "请把表头放在第 1 行，不要在上方再加标题行。",
                "一行代表一只股票在一个交易日的数据，不能把多天数据横着放。",
                "至少需要交易日期、股票代码、开盘价、最高价、最低价、收盘价 6 列。",
                "支持的日期格式包括 2026-03-13、20260313、Excel 日期单元格。",
                "如果您的列名不同，可以在页面里用“字段映射”手动指定。",
            ]
        }
    )


def build_template_bytes() -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(cast(Any, buffer), engine="openpyxl") as writer:
        build_sample_input_data().to_excel(
            writer, sheet_name="行情数据模板", index=False
        )
        build_data_format_table().to_excel(writer, sheet_name="字段说明", index=False)
        build_template_note().to_excel(writer, sheet_name="填写说明", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def format_detail_for_display(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df

    display_df = detail_df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    display_df["sell_date"] = pd.to_datetime(display_df["sell_date"]).dt.strftime(
        "%Y-%m-%d"
    )

    for column in DETAIL_PRICE_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.2f}" if pd.notna(value) else ""
            )

    for column in DETAIL_PERCENT_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.2f}%" if pd.notna(value) else ""
            )

    for column in DETAIL_NAV_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else ""
            )

    if "volume" in display_df.columns:
        display_df["volume"] = display_df["volume"].map(
            lambda value: f"{value:,.0f}" if pd.notna(value) else ""
        )

    display_df = display_df.rename(
        columns={
            column: DETAIL_COLUMN_LABELS[column]
            for column in display_df.columns
            if column in DETAIL_COLUMN_LABELS
        }
    )

    return display_df


def format_summary_for_display(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df

    display_df = daily_df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    for column in SUMMARY_PERCENT_COLUMNS:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else ""
        )
    display_df["avg_holding_days"] = display_df["avg_holding_days"].map(
        lambda value: f"{value:.2f}" if pd.notna(value) else ""
    )
    display_df = display_df.rename(
        columns={
            column: SUMMARY_COLUMN_LABELS[column]
            for column in display_df.columns
            if column in SUMMARY_COLUMN_LABELS
        }
    )
    return display_df


def format_equity_for_display(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return equity_df

    display_df = equity_df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    display_df["net_value"] = display_df["net_value"].map(
        lambda value: f"{value:.4f}" if pd.notna(value) else ""
    )
    for column in EQUITY_PERCENT_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.2f}%" if pd.notna(value) else ""
            )
    display_df = display_df.rename(
        columns={
            column: EQUITY_COLUMN_LABELS[column]
            for column in display_df.columns
            if column in EQUITY_COLUMN_LABELS
        }
    )
    return display_df


def build_download_name(start_date: str, end_date: str) -> str:
    return f"gap_analysis_{start_date}_{end_date}.xlsx"


def build_scan_axis(field_name: str, raw_text: str) -> ParamScanAxis | None:
    if field_name == "":
        return None
    values = parse_scan_values(field_name, raw_text)
    if not values:
        return None
    return ParamScanAxis(field_name=field_name, values=values)


def format_scan_for_display(scan_df: pd.DataFrame) -> pd.DataFrame:
    if scan_df.empty:
        return scan_df
    display_df = scan_df.copy()
    for column in display_df.columns:
        if column.endswith("_pct") and column != "rank":
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.2f}%" if pd.notna(value) else ""
            )
        elif column == "final_net_value":
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else ""
            )
        elif column == "rank":
            display_df[column] = display_df[column].map(
                lambda value: int(value) if pd.notna(value) else ""
            )
    display_df = display_df.rename(
        columns={
            column: (
                SCAN_COLUMN_LABELS.get(column)
                or SCAN_FIELD_LABELS.get(column)
                or SCAN_METRIC_LABELS.get(column)
                or column
            )
            for column in display_df.columns
        }
    )
    return display_df


def format_update_log_for_display(preview_df: pd.DataFrame) -> pd.DataFrame:
    if preview_df.empty:
        return preview_df

    display_df = preview_df.copy()
    for column in ("start_date", "end_date"):
        if column in display_df.columns:
            display_df[column] = pd.to_datetime(
                display_df[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    if "updated_at" in display_df.columns:
        display_df["updated_at"] = pd.to_datetime(
            display_df["updated_at"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")
    if "rows" in display_df.columns:
        display_df["rows"] = display_df["rows"].map(
            lambda value: f"{int(value):,}" if pd.notna(value) else ""
        )
    display_df = display_df.rename(
        columns={
            column: UPDATE_LOG_COLUMN_LABELS[column]
            for column in display_df.columns
            if column in UPDATE_LOG_COLUMN_LABELS
        }
    )
    return display_df


def section_header(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def summarize_stock_scope(stock_scope_text: str) -> str:
    stock_codes = normalize_stock_codes(stock_scope_text)
    if not stock_codes:
        return "全市场"
    preview = "、".join(stock_codes[:3])
    if len(stock_codes) <= 3:
        return preview
    return f"{preview} 等 {len(stock_codes)} 只"


def summarize_data_source(
    data_source_label: str,
    *,
    adjust_label: str,
    local_data_root: str,
    db_path: str,
    table_name: str,
    input_file_path: str,
    uploaded_market_file: Any | None,
) -> tuple[str, str]:
    if data_source_label == "本地 Parquet（AKShare 离线）":
        return "本地 Parquet", f"目录：{local_data_root} · 复权：{adjust_label}"
    if data_source_label == "SQLite 数据库":
        db_name = Path(db_path).name if db_path.strip() else "未填写路径"
        table_text = table_name.strip() or "自动识别表"
        return "SQLite", f"数据库：{db_name} · 数据表：{table_text}"

    file_name = "待选择文件"
    if uploaded_market_file is not None:
        file_name = uploaded_market_file.name
    elif input_file_path.strip():
        file_name = Path(input_file_path.strip()).name
    return "文件导入", f"文件：{file_name} · 复权：{adjust_label}"


def build_update_log_column_config() -> dict[str, object]:
    return {
        "股票代码": st.column_config.TextColumn("股票代码", width="small"),
        "复权方式": st.column_config.TextColumn("复权方式", width="small"),
        "开始日期": st.column_config.TextColumn("开始日期", width="small"),
        "结束日期": st.column_config.TextColumn("结束日期", width="small"),
        "更新行数": st.column_config.TextColumn("更新行数", width="small"),
        "更新时间": st.column_config.TextColumn("更新时间", width="medium"),
        "状态": st.column_config.TextColumn("状态", width="small"),
        "错误信息": st.column_config.TextColumn("错误信息", width="large"),
    }


def build_summary_column_config() -> dict[str, object]:
    return {
        "开仓日期": st.column_config.TextColumn("开仓日期", width="small"),
        "信号数": st.column_config.TextColumn("信号数", width="small"),
        "实际执行交易数": st.column_config.TextColumn("实际执行交易数", width="small"),
        "胜率": st.column_config.TextColumn("胜率", width="small"),
        "平均净收益率": st.column_config.TextColumn("平均净收益率", width="small"),
        "净收益率中位数": st.column_config.TextColumn("净收益率中位数", width="small"),
        "平均持有天数": st.column_config.TextColumn("平均持有天数", width="small"),
    }


def build_equity_column_config() -> dict[str, object]:
    return {
        "日期": st.column_config.TextColumn("日期", width="small"),
        "净值": st.column_config.TextColumn("净值", width="small"),
        "回撤": st.column_config.TextColumn("回撤", width="small"),
    }


def build_detail_column_config() -> dict[str, object]:
    return {
        "信号日期": st.column_config.TextColumn("信号日期", width="small"),
        "股票代码": st.column_config.TextColumn("股票代码", width="small"),
        "相对昨收跳空幅度": st.column_config.TextColumn(
            "相对昨收跳空幅度", width="small"
        ),
        "买入日期": st.column_config.TextColumn("买入日期", width="small"),
        "卖出日期": st.column_config.TextColumn("卖出日期", width="small"),
        "退出方式": st.column_config.TextColumn("退出方式", width="small"),
        "持有天数": st.column_config.TextColumn("持有天数", width="small"),
        "净收益率": st.column_config.TextColumn("净收益率", width="small"),
        "最大有利波动": st.column_config.TextColumn("最大有利波动", width="small"),
        "最大不利波动": st.column_config.TextColumn("最大不利波动", width="small"),
        "成交明细": st.column_config.TextColumn("成交明细", width="large"),
    }


def build_scan_column_config() -> dict[str, object]:
    return {
        "扫描编号": st.column_config.TextColumn("扫描编号", width="small"),
        "排名": st.column_config.TextColumn("排名", width="small"),
        "跳空幅度": st.column_config.TextColumn("跳空幅度", width="small"),
        "最大高开/低开过滤": st.column_config.TextColumn(
            "最大高开/低开过滤", width="small"
        ),
        "最多持有天数": st.column_config.TextColumn("最多持有天数", width="small"),
        "策略胜率": st.column_config.TextColumn("策略胜率", width="small"),
        "总收益率": st.column_config.TextColumn("总收益率", width="small"),
        "最大回撤": st.column_config.TextColumn("最大回撤", width="small"),
        "最终净值": st.column_config.TextColumn("最终净值", width="small"),
        "平均持有天数": st.column_config.TextColumn("平均持有天数", width="small"),
        "收益风险比": st.column_config.TextColumn("收益风险比", width="small"),
    }


def run_local_data_update(
    symbols_text: str,
    start_date: str,
    end_date: str,
    adjust: str,
    refresh_symbols: bool,
    export_excel: bool,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "scripts/update_data.py",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
    ]
    if adjust:
        cmd.extend(["--adjust", adjust])
    if symbols_text.strip():
        cmd.extend(["--symbols", symbols_text.strip()])
    if refresh_symbols:
        cmd.append("--refresh-symbols")
    if export_excel:
        cmd.append("--export-excel")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, output.strip()


def load_update_log_preview(limit: int = 20) -> pd.DataFrame:
    log_file = Path("data/market/metadata/update_log.parquet")
    if not log_file.exists():
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "symbol",
                    "adjust",
                    "start_date",
                    "end_date",
                    "rows",
                    "updated_at",
                    "status",
                    "error_message",
                ]
            )
        )
    try:
        df = pd.read_parquet(log_file)
    except Exception:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "symbol",
                    "adjust",
                    "start_date",
                    "end_date",
                    "rows",
                    "updated_at",
                    "status",
                    "error_message",
                ]
            )
        )

    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        df = df.sort_values("updated_at", ascending=False)
    return df.head(limit).reset_index(drop=True)


st.title("Gap_test 回测系统")
st.caption("离线数据更新、策略配置与结果分析统一在一个研究工作台中完成。")

today = pd.Timestamp.today().date()
default_update_start = today - timedelta(days=30)
default_backtest_start = today - timedelta(days=365)

section_header(
    "数据准备",
    "推荐先离线更新本地 parquet，再进行回测；更新区保留在顶部，避免与回测配置混在一起。",
)

with st.expander("本地行情更新（离线下载）", expanded=False):
    up_c1, up_c2, up_c3 = st.columns(3)
    with up_c1:
        update_symbols = st.text_area(
            "股票代码（可选）", value="", help="多个代码用逗号分隔，空表示全量。"
        )
    with up_c2:
        update_start = st.date_input(
            "更新开始日期",
            value=default_update_start,
            key="offline_update_start",
        )
        update_end = st.date_input(
            "更新结束日期", value=today, key="offline_update_end"
        )
    with up_c3:
        update_adjust = st.selectbox(
            "更新复权方式", options=["qfq", "hfq"], index=0, key="offline_update_adjust"
        )
        refresh_symbol_meta = st.checkbox(
            "刷新股票列表", value=False, key="offline_update_refresh"
        )
        export_excel_after_update = st.checkbox(
            "更新后另存为 Excel", value=False, key="offline_update_export"
        )

    if st.button("开始更新本地数据", key="offline_update_submit"):
        ok, output = run_local_data_update(
            symbols_text=update_symbols,
            start_date=update_start.strftime("%Y-%m-%d"),
            end_date=update_end.strftime("%Y-%m-%d"),
            adjust=update_adjust,
            refresh_symbols=bool(refresh_symbol_meta),
            export_excel=bool(export_excel_after_update),
        )
        if ok:
            st.success("本地数据更新完成")
        else:
            st.error("本地数据更新失败")
        st.caption(f"导出目录：data/market/exports/{update_adjust}/")
        if output:
            st.code(output)

    preview = load_update_log_preview(limit=20)
    if not preview.empty:
        st.markdown("**最近更新日志**")
        dataframe_stretch(
            format_update_log_for_display(preview),
            hide_index=True,
            column_config=build_update_log_column_config(),
            height=280,
        )

st.divider()

# ===== Sidebar: 基础参数 =====
st.sidebar.header("基础参数")
st.sidebar.caption("左侧只放发起回测所需的核心输入，详细规则放在主区域。")
stock_scope_text = st.sidebar.text_area(
    "股票池", value="", help="多个代码可用逗号/空格/换行。留空表示全市场。"
)
start_date = st.sidebar.date_input("回测开始", value=default_backtest_start)
end_date = st.sidebar.date_input("回测结束", value=today)
SOURCE_LABEL_TO_TYPE = {
    "本地 Parquet（AKShare 离线）": "local_parquet",
    "Excel/CSV 文件": "file",
    "SQLite 数据库": "sqlite",
}
data_source_label = st.sidebar.selectbox(
    "数据源", options=list(SOURCE_LABEL_TO_TYPE.keys())
)
adjust_label = st.sidebar.selectbox("复权方式", options=["qfq", "hfq"], index=0)
submitted = st.sidebar.button("开始回测", type="primary")
st.sidebar.divider()
st.sidebar.caption("数据源补充信息")

# 数据源输入（仍放侧边栏，保持小白可见）
default_db_path = str(Path.cwd() / "market_data.sqlite")
db_path = default_db_path
table_name = ""
input_file_path = ""
excel_sheet_name = ""
uploaded_market_file = None

local_data_root = st.sidebar.text_input(
    "本地 Parquet 根目录", value="data/market/daily"
)
if data_source_label == "SQLite 数据库":
    db_path = st.sidebar.text_input("SQLite 路径", value=default_db_path)
    table_name = st.sidebar.text_input("表名（可选）", value="")
elif data_source_label == "Excel/CSV 文件":
    uploaded_market_file = st.sidebar.file_uploader(
        "上传行情文件", type=["xlsx", "xlsm", "csv"]
    )
    input_file_path = st.sidebar.text_input("或本地文件路径（可选）", value="")
    excel_sheet_name = st.sidebar.text_input("工作表（Excel 可选）", value="")
else:
    st.sidebar.caption("当前使用本地 parquet 数据源，回测将直接读取本地目录。")

# ===== 主界面：配置摘要 =====
section_header(
    "回测概览",
    "先确认研究范围与数据来源，再进入规则配置；摘要区域只显示最关键状态。",
)
source_summary_title, source_summary_desc = summarize_data_source(
    data_source_label,
    adjust_label=adjust_label,
    local_data_root=local_data_root,
    db_path=db_path,
    table_name=table_name,
    input_file_path=input_file_path,
    uploaded_market_file=uploaded_market_file,
)
summary_cols = st.columns(3)
summary_cols[0].metric(
    "交易方向",
    "做多"
    if st.session_state.get("direction_label", "向上跳空") == "向上跳空"
    else "做空",
    border=True,
)
summary_cols[1].metric("股票池", summarize_stock_scope(stock_scope_text), border=True)
summary_cols[2].metric("回测区间", f"{start_date} → {end_date}", border=True)
summary_cols_2 = st.columns(3)
summary_cols_2[0].metric(
    "数据源", source_summary_title, source_summary_desc, border=True
)
summary_cols_2[1].metric(
    "分批退出",
    "开启" if st.session_state.get("partial_exit_enabled", False) else "关闭",
    border=True,
)
summary_cols_2[2].metric(
    "时间退出",
    "开启" if st.session_state.get("use_time_stop", True) else "关闭",
    border=True,
)

st.divider()
section_header("策略配置", "核心参数默认展开，高级参数保持次要，避免一屏堆满控件。")

# ===== 主界面：规则配置 =====
with st.expander("⚙️ 核心交易规则配置", expanded=True):
    st.caption("优先配置开仓、止损止盈、时间退出与交易成本。")
    direction_label = st.selectbox(
        "交易方向", options=["向上跳空", "向下跳空"], key="direction_label"
    )
    gap_entry_mode = st.selectbox(
        "开仓信号模式",
        options=list(GAP_ENTRY_MODES),
        format_func=lambda value: (
            "严格突破前高/前低" if value == "strict_break" else "开盘相对昨收达到阈值"
        ),
    )
    gap_pct = st.number_input("跳空幅度（%）", min_value=0.0, value=2.0, step=0.1)
    max_gap_filter_pct = st.number_input(
        "最大高开/低开过滤（%）", min_value=0.0, value=9.9, step=0.1
    )

    use_ma_filter = st.checkbox("启用快慢线开单过滤", value=False)
    c1, c2 = st.columns(2)
    fast_ma_period = c1.number_input(
        "快线周期", min_value=1, value=5, step=1, disabled=not use_ma_filter
    )
    slow_ma_period = c2.number_input(
        "慢线周期", min_value=1, value=20, step=1, disabled=not use_ma_filter
    )

    use_time_stop = st.checkbox("启用时间退出", value=True, key="use_time_stop")
    c3, c4 = st.columns(2)
    time_stop_days = c3.number_input(
        "最多持有天数 N", min_value=1, value=5, step=1, disabled=not use_time_stop
    )
    time_stop_target_pct = c4.number_input(
        "时间退出收益阈值（%）", value=1.0, step=0.1, disabled=not use_time_stop
    )
    time_exit_mode_label = st.selectbox(
        "数据结束处理", options=["按原规则剔除未达条件信号", "第 N 天按收盘价结束交易"]
    )

    stop_loss_pct = st.number_input("全仓止损（%）", min_value=0.0, value=3.0, step=0.1)
    enable_take_profit = st.checkbox("启用固定止盈", value=True)
    take_profit_pct = st.number_input(
        "固定止盈（%）",
        min_value=0.0,
        value=5.0,
        step=0.1,
        disabled=not enable_take_profit,
    )

    enable_profit_drawdown_exit = st.checkbox("启用盈利回撤止盈（整笔）", value=False)
    profit_drawdown_pct = st.number_input(
        "盈利回撤（%）",
        min_value=0.0,
        value=40.0,
        step=1.0,
        disabled=not enable_profit_drawdown_exit,
    )

    enable_ma_exit = st.checkbox("启用均线离场（整笔）", value=False)
    exit_ma_period = st.number_input(
        "离场均线周期", min_value=1, value=10, step=1, disabled=not enable_ma_exit
    )
    ma_exit_batches = st.number_input(
        "均线离场分批数",
        min_value=2,
        max_value=3,
        value=2,
        step=1,
        disabled=not enable_ma_exit,
    )

    buy_cost_pct = st.number_input(
        "买入成本（%）", min_value=0.0, value=0.03, step=0.01, format="%.4f"
    )
    sell_cost_pct = st.number_input(
        "卖出成本（%）", min_value=0.0, value=0.13, step=0.01, format="%.4f"
    )
    buy_slippage_pct = st.number_input(
        "买入滑点（%）", min_value=0.0, value=0.0, step=0.01, format="%.4f"
    )
    sell_slippage_pct = st.number_input(
        "卖出滑点（%）", min_value=0.0, value=0.0, step=0.01, format="%.4f"
    )

with st.expander("🛠️ 分批止盈高级配置", expanded=False):
    st.caption("仅在需要拆分仓位管理时启用，按 priority 从小到大执行。")
    partial_exit_enabled = st.checkbox(
        "启用分批止盈", value=False, key="partial_exit_enabled"
    )
    partial_exit_count = st.number_input(
        "分批数量",
        min_value=2,
        max_value=3,
        value=2,
        step=1,
        disabled=not partial_exit_enabled,
    )
    partial_rule_inputs = []
    for i in range(1, int(partial_exit_count) + 1):
        with st.expander(f"第 {i} 批", expanded=(i <= 2)):
            c1, c2 = st.columns(2)
            weight_default = (
                50.0 if int(partial_exit_count) == 2 else [30.0, 30.0, 40.0][i - 1]
            )
            mode_default = (
                ["fixed_tp", "ma_exit"][i - 1]
                if int(partial_exit_count) == 2
                else ["fixed_tp", "fixed_tp", "ma_exit"][i - 1]
            )
            weight_pct = c1.number_input(
                f"第{i}批 仓位比例%",
                min_value=0.0,
                max_value=100.0,
                value=weight_default,
                step=1.0,
                disabled=not partial_exit_enabled,
                key=f"p_weight_{i}",
            )
            priority = c1.number_input(
                f"第{i}批 priority",
                min_value=1,
                max_value=10,
                value=i,
                step=1,
                disabled=not partial_exit_enabled,
                key=f"p_priority_{i}",
            )
            mode = c2.selectbox(
                f"第{i}批 退出方式",
                options=["fixed_tp", "ma_exit", "profit_drawdown"],
                index=["fixed_tp", "ma_exit", "profit_drawdown"].index(mode_default),
                disabled=not partial_exit_enabled,
                key=f"p_mode_{i}",
            )
            tp = st.number_input(
                f"第{i}批 目标收益%",
                value=5.0,
                step=0.1,
                disabled=(not partial_exit_enabled) or mode != "fixed_tp",
                key=f"p_tp_{i}",
            )
            ma = st.number_input(
                f"第{i}批 均线周期",
                min_value=1,
                value=10,
                step=1,
                disabled=(not partial_exit_enabled) or mode != "ma_exit",
                key=f"p_ma_{i}",
            )
            dd = st.number_input(
                f"第{i}批 回撤比例%",
                min_value=0.0,
                value=20.0,
                step=0.1,
                disabled=(not partial_exit_enabled) or mode != "profit_drawdown",
                key=f"p_dd_{i}",
            )
            mpa = st.number_input(
                f"第{i}批 最小浮盈激活%",
                min_value=0.0,
                value=5.0,
                step=0.1,
                disabled=(not partial_exit_enabled) or mode != "profit_drawdown",
                key=f"p_mpa_{i}",
            )
            partial_rule_inputs.append(
                {
                    "enabled": bool(partial_exit_enabled),
                    "weight_pct": float(weight_pct),
                    "mode": mode,
                    "priority": int(priority),
                    "target_profit_pct": float(tp) if mode == "fixed_tp" else None,
                    "ma_period": int(ma) if mode == "ma_exit" else None,
                    "drawdown_pct": float(dd) if mode == "profit_drawdown" else None,
                    "min_profit_to_activate_drawdown": float(mpa)
                    if mode == "profit_drawdown"
                    else None,
                }
            )

with st.expander("🔎 参数敏感性扫描", expanded=False):
    st.caption("适合做参数边界探索；建议先用少量组合验证，再扩大扫描范围。")
    scan_enabled = st.checkbox("启用参数扫描", value=False)
    scan_metric = st.selectbox(
        "扫描排序指标",
        options=list(SCAN_METRICS),
        format_func=lambda value: str(SCAN_METRIC_LABELS.get(value, value) or value),
        disabled=not scan_enabled,
    )
    scan_field_options = [""] + list(SCAN_FIELD_CASTERS.keys())
    scan_axis_1 = st.selectbox(
        "扫描维度 1",
        options=scan_field_options,
        format_func=lambda value: str(
            "请选择字段"
            if value == ""
            else (SCAN_FIELD_LABELS.get(value, value) or value)
        ),
        disabled=not scan_enabled,
    )
    scan_axis_1_values = st.text_input(
        "维度 1 取值",
        value="",
        disabled=not scan_enabled,
        help="使用逗号分隔，例如 2,3,4",
    )
    scan_axis_2 = st.selectbox(
        "扫描维度 2（可选）",
        options=scan_field_options,
        format_func=lambda value: str(
            "不启用第二维"
            if value == ""
            else (SCAN_FIELD_LABELS.get(value, value) or value)
        ),
        disabled=not scan_enabled,
    )
    scan_axis_2_values = st.text_input(
        "维度 2 取值",
        value="",
        disabled=not scan_enabled,
        help="留空表示只扫描一维",
    )
    scan_max_combinations = st.number_input(
        "最大组合数",
        min_value=1,
        max_value=100,
        value=25,
        step=1,
        disabled=not scan_enabled,
    )

# 字段映射
with st.expander("字段映射（可选）", expanded=False):
    st.caption("仅在导入文件列名不标准时填写，常规本地 parquet 回测无需改动。")
    mc1, mc2, mc3 = st.columns(3)
    date_column = mc1.text_input("日期列名", value="")
    stock_code_column = mc1.text_input("股票代码列名", value="")
    open_column = mc1.text_input("开盘价列名", value="")
    high_column = mc2.text_input("最高价列名", value="")
    low_column = mc2.text_input("最低价列名", value="")
    close_column = mc2.text_input("收盘价列名", value="")
    volume_column = mc3.text_input("成交量列名", value="")

if submitted:
    clear_result_state()
    source_type = SOURCE_LABEL_TO_TYPE[data_source_label]
    uploaded_file_bytes = (
        uploaded_market_file.getvalue() if uploaded_market_file is not None else None
    )
    uploaded_file_name = (
        uploaded_market_file.name if uploaded_market_file is not None else None
    )

    column_overrides = normalize_column_overrides(
        {
            "date": date_column,
            "stock_code": stock_code_column,
            "open": open_column,
            "high": high_column,
            "low": low_column,
            "close": close_column,
            "volume": volume_column,
        }
    )
    partial_rules = tuple(PartialExitRule(**rule) for rule in partial_rule_inputs)
    scan_axes = tuple(
        axis
        for axis in (
            build_scan_axis(str(scan_axis_1), scan_axis_1_values),
            build_scan_axis(str(scan_axis_2), scan_axis_2_values),
        )
        if axis is not None
    )
    scan_config = ParamScanConfig(
        enabled=bool(scan_enabled),
        axes=scan_axes,
        metric=str(scan_metric),
        max_combinations=int(scan_max_combinations),
    )
    params = AnalysisParams(
        data_source_type=source_type,
        db_path=db_path.strip(),
        table_name=table_name.strip() or None,
        column_overrides=column_overrides,
        excel_sheet_name=excel_sheet_name.strip() or None,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        stock_codes=normalize_stock_codes(stock_scope_text),
        gap_direction="up" if direction_label == "向上跳空" else "down",
        gap_entry_mode=str(gap_entry_mode),
        gap_pct=float(gap_pct),
        max_gap_filter_pct=float(max_gap_filter_pct),
        use_ma_filter=bool(use_ma_filter),
        fast_ma_period=int(fast_ma_period),
        slow_ma_period=int(slow_ma_period),
        time_stop_days=int(time_stop_days),
        time_stop_target_pct=float(time_stop_target_pct),
        stop_loss_pct=float(stop_loss_pct),
        take_profit_pct=float(take_profit_pct),
        enable_take_profit=bool(enable_take_profit),
        enable_profit_drawdown_exit=bool(enable_profit_drawdown_exit),
        profit_drawdown_pct=float(profit_drawdown_pct),
        enable_ma_exit=bool(enable_ma_exit),
        exit_ma_period=int(exit_ma_period),
        ma_exit_batches=int(ma_exit_batches),
        partial_exit_enabled=bool(partial_exit_enabled),
        partial_exit_count=int(partial_exit_count),
        partial_exit_rules=partial_rules,
        buy_cost_pct=float(buy_cost_pct),
        sell_cost_pct=float(sell_cost_pct),
        buy_slippage_pct=float(buy_slippage_pct),
        sell_slippage_pct=float(sell_slippage_pct),
        time_exit_mode="strict"
        if time_exit_mode_label == "按原规则剔除未达条件信号"
        else "force_close",
        local_data_root=str(local_data_root),
        adjust=str(adjust_label),
        scan_config=scan_config,
    )

    errors, warnings = validate_params(params)
    for warning in warnings:
        st.warning(warning)

    if params.data_source_type == "file":
        input_file_path = input_file_path.strip()
        if uploaded_file_bytes is not None and input_file_path:
            st.warning("同时提供了上传文件和本地文件路径，当前会优先使用上传文件。")
        if uploaded_file_bytes is None and not input_file_path:
            errors.append("请选择 Excel/CSV 文件，或者填写本地文件路径。")
        if (
            uploaded_file_bytes is None
            and input_file_path
            and not Path(input_file_path).exists()
        ):
            errors.append(f"找不到文件：{input_file_path}")

    if errors:
        st.error("参数校验失败")
        for error in errors:
            st.error(error)
    else:
        try:
            with st.spinner("正在运行回测，请稍候..."):
                all_data = load_market_data(
                    source_type=params.data_source_type,
                    start_date=params.start_date,
                    end_date=params.end_date,
                    stock_codes=params.stock_codes,
                    table_name=params.table_name,
                    column_overrides=params.column_overrides,
                    lookback_days=params.required_lookback_days,
                    lookahead_days=params.required_lookahead_days,
                    db_path=params.db_path,
                    file_path=input_file_path or None,
                    file_bytes=uploaded_file_bytes,
                    file_name=uploaded_file_name,
                    sheet_name=params.excel_sheet_name,
                    local_data_root=params.local_data_root,
                    adjust=params.adjust,
                )
                scan_df = pd.DataFrame()
                best_scan_overrides: dict[str, int | float] = {}
                if params.scan_config.enabled:
                    (
                        scan_df,
                        detail_df,
                        daily_df,
                        equity_df,
                        stats,
                        best_scan_overrides,
                    ) = run_parameter_scan(all_data, params)
                else:
                    detail_df, daily_df, equity_df, stats = analyze_all_stocks(
                        all_data, params
                    )
                excel_bytes = export_to_excel_bytes(
                    detail_df, daily_df, equity_df, scan_df=scan_df
                )
            st.success("回测完成")
            st.session_state["detail_df"] = detail_df
            st.session_state["daily_df"] = daily_df
            st.session_state["equity_df"] = equity_df
            st.session_state["stats"] = stats
            st.session_state["scan_df"] = scan_df
            st.session_state["scan_metric"] = params.scan_config.metric
            st.session_state["scan_axis_fields"] = [
                axis.field_name for axis in params.scan_config.axes
            ]
            st.session_state["best_scan_overrides"] = best_scan_overrides
            st.session_state["excel_bytes"] = excel_bytes
            st.session_state["download_name"] = build_download_name(
                params.start_date, params.end_date
            )
        except Exception as exc:
            st.error(f"回测失败：{exc}")

detail_df = st.session_state.get("detail_df", pd.DataFrame())
daily_df = st.session_state.get("daily_df", pd.DataFrame())
equity_df = st.session_state.get("equity_df", pd.DataFrame())
stats = st.session_state.get("stats", {})
scan_df = st.session_state.get("scan_df", pd.DataFrame())
scan_metric = str(st.session_state.get("scan_metric", "total_return_pct"))
scan_axis_fields = list(st.session_state.get("scan_axis_fields", []))
best_scan_overrides = dict(st.session_state.get("best_scan_overrides", {}))

if isinstance(detail_df, pd.DataFrame) and "excel_bytes" in st.session_state:
    tab_names = ["📊 绩效总览", "📈 资金曲线", "📝 交易明细"]
    if isinstance(scan_df, pd.DataFrame) and not scan_df.empty:
        tab_names.append("🔎 参数扫描")
    tabs = st.tabs(tab_names)
    tab_summary, tab_curve, tab_details = tabs[:3]
    with tab_summary:
        section_header("绩效总览", "先看关键结果，再决定是否继续展开明细或参数扫描。")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总收益率", f"{float(stats.get('total_return_pct', 0.0)):.2f}%")
        c2.metric("胜率", f"{float(stats.get('strategy_win_rate_pct', 0.0)):.2f}%")
        c3.metric("最大回撤", f"{float(stats.get('max_drawdown_pct', 0.0)):.2f}%")
        c4.metric("交易笔数", f"{int(stats.get('executed_trades', len(detail_df)))}")
        if best_scan_overrides:
            summary_text = ", ".join(
                f"{SCAN_FIELD_LABELS.get(field_name, field_name) or field_name}={value}"
                for field_name, value in best_scan_overrides.items()
            )
            st.caption(f"最佳参数组合：{summary_text}")
        if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
            st.markdown("**按开仓日汇总**")
            dataframe_stretch(
                format_summary_for_display(daily_df),
                hide_index=True,
                column_config=build_summary_column_config(),
                height=260,
            )
        st.download_button(
            "导出 Excel",
            data=st.session_state["excel_bytes"],
            file_name=st.session_state["download_name"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with tab_curve:
        if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
            section_header("净值与回撤", "默认展示净值曲线，并保留表格视图便于核对。")
            chart_df = equity_df.copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            fig = px.line(chart_df, x="date", y="net_value", title="资金曲线")
            st.plotly_chart(fig, use_container_width=True)
            dataframe_stretch(
                format_equity_for_display(equity_df),
                hide_index=True,
                column_config=build_equity_column_config(),
                height=260,
            )
        else:
            st.info("暂无资金曲线数据")

    with tab_details:
        if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
            section_header("交易明细", "表头已转中文，优先展示交易关键信息与成交明细。")
            dataframe_stretch(
                format_detail_for_display(detail_df),
                hide_index=True,
                column_config=build_detail_column_config(),
                height=420,
            )
            csv_bytes = detail_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "导出 CSV",
                data=csv_bytes,
                file_name="trade_details.csv",
                mime="text/csv",
            )
        else:
            st.info("暂无交易明细")

    if len(tabs) == 4:
        with tabs[3]:
            section_header(
                "参数扫描结果", "先看排名表，再看热力图/折线图判断参数敏感性。"
            )
            dataframe_stretch(
                format_scan_for_display(scan_df),
                hide_index=True,
                column_config=build_scan_column_config(),
                height=360,
            )
            if len(scan_axis_fields) == 2:
                pivot = scan_df.pivot(
                    index=scan_axis_fields[1],
                    columns=scan_axis_fields[0],
                    values=scan_metric,
                )
                fig = px.imshow(
                    pivot,
                    text_auto=True if pivot.size <= 36 else False,
                    aspect="auto",
                    labels={
                        "x": SCAN_FIELD_LABELS.get(
                            scan_axis_fields[0], scan_axis_fields[0]
                        )
                        or scan_axis_fields[0],
                        "y": SCAN_FIELD_LABELS.get(
                            scan_axis_fields[1], scan_axis_fields[1]
                        )
                        or scan_axis_fields[1],
                        "color": SCAN_METRIC_LABELS.get(scan_metric, scan_metric)
                        or scan_metric,
                    },
                )
                fig.update_xaxes(side="top")
                st.plotly_chart(fig, use_container_width=True)
            elif len(scan_axis_fields) == 1:
                chart_df = scan_df.sort_values(scan_axis_fields[0]).copy()
                fig = px.line(
                    chart_df,
                    x=scan_axis_fields[0],
                    y=scan_metric,
                    markers=True,
                    labels={
                        scan_axis_fields[0]: SCAN_FIELD_LABELS.get(
                            scan_axis_fields[0], scan_axis_fields[0]
                        )
                        or scan_axis_fields[0],
                        scan_metric: SCAN_METRIC_LABELS.get(scan_metric, scan_metric)
                        or scan_metric,
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
