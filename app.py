from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import analyze_all_stocks
from data_loader import describe_file_source, describe_tables, list_candidate_tables, load_market_data
from exporter import export_to_excel_bytes
from models import AnalysisParams, PartialExitRule, normalize_column_overrides, normalize_stock_codes, validate_params


st.set_page_config(layout="wide", page_title="Gap_test 回测系统")

RESULT_STATE_KEYS = [
    "detail_df",
    "daily_df",
    "equity_df",
    "stats",
    "excel_bytes",
    "download_name",
]

DETAIL_PRICE_COLUMNS = ["prev_close", "prev_high", "prev_low", "open", "close", "buy_price", "sell_price", "exit_ma_value"]
DETAIL_PERCENT_COLUMNS = ["gap_pct_vs_prev_close", "gross_return_pct", "net_return_pct", "mfe_pct", "mae_pct", "max_profit_pct", "profit_drawdown_ratio"]
DETAIL_NAV_COLUMNS = ["nav_before_trade", "nav_after_trade"]
SUMMARY_PERCENT_COLUMNS = ["win_rate_pct", "avg_net_return_pct", "median_net_return_pct"]
EQUITY_PERCENT_COLUMNS = ["drawdown_pct"]


def dataframe_stretch(data: object, *, hide_index: bool = False) -> None:
    st.dataframe(data, hide_index=hide_index)


def form_submit_button_stretch(label: str) -> bool:
    return st.form_submit_button(label)


def clear_result_state() -> None:
    for key in RESULT_STATE_KEYS:
        st.session_state.pop(key, None)


def build_data_format_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"列名建议": "交易日期", "是否必填": "必填", "说明": "交易日，一行代表一只股票在一个交易日的数据", "示例": "2026-03-13"},
            {"列名建议": "股票代码", "是否必填": "必填", "说明": "股票唯一标识", "示例": "000001.SZ"},
            {"列名建议": "开盘价", "是否必填": "必填", "说明": "当天开盘价", "示例": "10.52"},
            {"列名建议": "最高价", "是否必填": "必填", "说明": "当天最高价", "示例": "10.88"},
            {"列名建议": "最低价", "是否必填": "必填", "说明": "当天最低价", "示例": "10.31"},
            {"列名建议": "收盘价", "是否必填": "必填", "说明": "当天收盘价", "示例": "10.66"},
            {"列名建议": "成交量", "是否必填": "选填", "说明": "当天成交量，不填也可以分析", "示例": "1256300"},
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
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        build_sample_input_data().to_excel(writer, sheet_name="行情数据模板", index=False)
        build_data_format_table().to_excel(writer, sheet_name="字段说明", index=False)
        build_template_note().to_excel(writer, sheet_name="填写说明", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def format_detail_for_display(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df

    display_df = detail_df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    display_df["sell_date"] = pd.to_datetime(display_df["sell_date"]).dt.strftime("%Y-%m-%d")

    for column in DETAIL_PRICE_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")

    for column in DETAIL_PERCENT_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.2f}%" if pd.notna(value) else "")

    for column in DETAIL_NAV_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")

    if "volume" in display_df.columns:
        display_df["volume"] = display_df["volume"].map(lambda value: f"{value:,.0f}" if pd.notna(value) else "")

    return display_df


def format_summary_for_display(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df

    display_df = daily_df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    for column in SUMMARY_PERCENT_COLUMNS:
        display_df[column] = display_df[column].map(lambda value: f"{value:.2f}%" if pd.notna(value) else "")
    display_df["avg_holding_days"] = display_df["avg_holding_days"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    return display_df


def format_equity_for_display(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return equity_df

    display_df = equity_df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    display_df["net_value"] = display_df["net_value"].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    for column in EQUITY_PERCENT_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{value:.2f}%" if pd.notna(value) else "")
    return display_df


def build_download_name(start_date: str, end_date: str) -> str:
    return f"gap_analysis_{start_date}_{end_date}.xlsx"


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
        return pd.DataFrame(columns=["symbol", "adjust", "start_date", "end_date", "rows", "updated_at", "status", "error_message"])
    try:
        df = pd.read_parquet(log_file)
    except Exception:
        return pd.DataFrame(columns=["symbol", "adjust", "start_date", "end_date", "rows", "updated_at", "status", "error_message"])

    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        df = df.sort_values("updated_at", ascending=False)
    return df.head(limit).reset_index(drop=True)


st.title("Gap_test 回测系统")

# ===== Sidebar: 基础参数 =====
st.sidebar.header("基础参数")
stock_scope_text = st.sidebar.text_area("股票池", value="", help="多个代码可用逗号/空格/换行。留空表示全市场。")
start_date = st.sidebar.date_input("回测开始", value=(pd.Timestamp.today().date() - pd.Timedelta(days=365)))
end_date = st.sidebar.date_input("回测结束", value=pd.Timestamp.today().date())
data_source_label = st.sidebar.selectbox("数据源", options=["Excel/CSV 文件", "SQLite 数据库"])
adjust_label = st.sidebar.selectbox("复权方式", options=["qfq", "hfq"], index=0)
submitted = st.sidebar.button("开始回测", type="primary")

# 数据源输入（仍放侧边栏，保持小白可见）
default_db_path = str(Path.cwd() / "market_data.sqlite")
db_path = default_db_path
table_name = ""
input_file_path = ""
excel_sheet_name = ""
uploaded_market_file = None

if data_source_label == "SQLite 数据库":
    db_path = st.sidebar.text_input("SQLite 路径", value=default_db_path)
    table_name = st.sidebar.text_input("表名（可选）", value="")
else:
    uploaded_market_file = st.sidebar.file_uploader("上传行情文件", type=["xlsx", "xlsm", "csv"])
    input_file_path = st.sidebar.text_input("或本地文件路径（可选）", value="")
    excel_sheet_name = st.sidebar.text_input("工作表（Excel 可选）", value="")

# ===== 主界面：配置摘要 =====
st.subheader("当前配置摘要")
summary_cols = st.columns(3)
summary_cols[0].info(f"方向: {'做多' if st.session_state.get('direction_label', '向上跳空') == '向上跳空' else '做空'}")
summary_cols[1].info(f"股票池: {stock_scope_text.strip() or '全市场'}")
summary_cols[2].info(f"区间: {start_date} ~ {end_date}")
summary_cols_2 = st.columns(3)
summary_cols_2[0].info(f"复权: {adjust_label}")
summary_cols_2[1].info(f"分批退出: {'开启' if st.session_state.get('partial_exit_enabled', False) else '关闭'}")
summary_cols_2[2].info(f"时间退出: {'开启' if st.session_state.get('use_time_stop', True) else '开启'}")

# ===== 主界面：规则配置 =====
with st.expander("⚙️ 核心交易规则配置", expanded=True):
    direction_label = st.selectbox("交易方向", options=["向上跳空", "向下跳空"], key="direction_label")
    gap_pct = st.number_input("跳空幅度（%）", min_value=0.0, value=2.0, step=0.1)
    max_gap_filter_pct = st.number_input("最大高开/低开过滤（%）", min_value=0.0, value=9.9, step=0.1)

    use_ma_filter = st.checkbox("启用快慢线开单过滤", value=False)
    c1, c2 = st.columns(2)
    fast_ma_period = c1.number_input("快线周期", min_value=1, value=5, step=1, disabled=not use_ma_filter)
    slow_ma_period = c2.number_input("慢线周期", min_value=1, value=20, step=1, disabled=not use_ma_filter)

    use_time_stop = st.checkbox("启用时间退出", value=True, key="use_time_stop")
    c3, c4 = st.columns(2)
    time_stop_days = c3.number_input("最多持有天数 N", min_value=1, value=5, step=1, disabled=not use_time_stop)
    time_stop_target_pct = c4.number_input("时间退出收益阈值（%）", value=1.0, step=0.1, disabled=not use_time_stop)
    time_exit_mode_label = st.selectbox("数据结束处理", options=["按原规则剔除未达条件信号", "第 N 天按收盘价结束交易"])

    stop_loss_pct = st.number_input("全仓止损（%）", min_value=0.0, value=3.0, step=0.1)
    enable_take_profit = st.checkbox("启用固定止盈", value=True)
    take_profit_pct = st.number_input("固定止盈（%）", min_value=0.0, value=5.0, step=0.1, disabled=not enable_take_profit)

    enable_profit_drawdown_exit = st.checkbox("启用盈利回撤止盈（整笔）", value=False)
    profit_drawdown_pct = st.number_input("盈利回撤（%）", min_value=0.0, value=40.0, step=1.0, disabled=not enable_profit_drawdown_exit)

    enable_ma_exit = st.checkbox("启用均线离场（整笔）", value=False)
    exit_ma_period = st.number_input("离场均线周期", min_value=1, value=10, step=1, disabled=not enable_ma_exit)
    ma_exit_batches = st.number_input("均线离场分批数", min_value=2, max_value=3, value=2, step=1, disabled=not enable_ma_exit)

    buy_cost_pct = st.number_input("买入成本（%）", min_value=0.0, value=0.03, step=0.01, format="%.4f")
    sell_cost_pct = st.number_input("卖出成本（%）", min_value=0.0, value=0.13, step=0.01, format="%.4f")

with st.expander("🛠️ 分批止盈高级配置", expanded=False):
    partial_exit_enabled = st.checkbox("启用分批止盈", value=False, key="partial_exit_enabled")
    partial_exit_count = st.number_input("分批数量", min_value=2, max_value=3, value=2, step=1, disabled=not partial_exit_enabled)
    partial_rule_inputs = []
    for i in range(1, int(partial_exit_count) + 1):
        with st.expander(f"第 {i} 批", expanded=(i <= 2)):
            c1, c2 = st.columns(2)
            weight_default = 50.0 if int(partial_exit_count) == 2 else [30.0, 30.0, 40.0][i - 1]
            mode_default = ["fixed_tp", "ma_exit"][i - 1] if int(partial_exit_count) == 2 else ["fixed_tp", "fixed_tp", "ma_exit"][i - 1]
            weight_pct = c1.number_input(f"第{i}批 仓位比例%", min_value=0.0, max_value=100.0, value=weight_default, step=1.0, disabled=not partial_exit_enabled, key=f"p_weight_{i}")
            priority = c1.number_input(f"第{i}批 priority", min_value=1, max_value=10, value=i, step=1, disabled=not partial_exit_enabled, key=f"p_priority_{i}")
            mode = c2.selectbox(f"第{i}批 退出方式", options=["fixed_tp", "ma_exit", "profit_drawdown"], index=["fixed_tp", "ma_exit", "profit_drawdown"].index(mode_default), disabled=not partial_exit_enabled, key=f"p_mode_{i}")
            tp = st.number_input(f"第{i}批 目标收益%", value=5.0, step=0.1, disabled=(not partial_exit_enabled) or mode != "fixed_tp", key=f"p_tp_{i}")
            ma = st.number_input(f"第{i}批 均线周期", min_value=1, value=10, step=1, disabled=(not partial_exit_enabled) or mode != "ma_exit", key=f"p_ma_{i}")
            dd = st.number_input(f"第{i}批 回撤比例%", min_value=0.0, value=20.0, step=0.1, disabled=(not partial_exit_enabled) or mode != "profit_drawdown", key=f"p_dd_{i}")
            mpa = st.number_input(f"第{i}批 最小浮盈激活%", min_value=0.0, value=5.0, step=0.1, disabled=(not partial_exit_enabled) or mode != "profit_drawdown", key=f"p_mpa_{i}")
            partial_rule_inputs.append({
                "enabled": bool(partial_exit_enabled),
                "weight_pct": float(weight_pct),
                "mode": mode,
                "priority": int(priority),
                "target_profit_pct": float(tp) if mode == "fixed_tp" else None,
                "ma_period": int(ma) if mode == "ma_exit" else None,
                "drawdown_pct": float(dd) if mode == "profit_drawdown" else None,
                "min_profit_to_activate_drawdown": float(mpa) if mode == "profit_drawdown" else None,
            })

# 字段映射
with st.expander("字段映射（可选）", expanded=False):
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
    source_type = "file" if data_source_label == "Excel/CSV 文件" else "sqlite"
    uploaded_file_bytes = uploaded_market_file.getvalue() if uploaded_market_file is not None else None
    uploaded_file_name = uploaded_market_file.name if uploaded_market_file is not None else None

    column_overrides = normalize_column_overrides({
        "date": date_column,
        "stock_code": stock_code_column,
        "open": open_column,
        "high": high_column,
        "low": low_column,
        "close": close_column,
        "volume": volume_column,
    })
    partial_rules = tuple(PartialExitRule(**rule) for rule in partial_rule_inputs)
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
        time_exit_mode="strict" if time_exit_mode_label == "按原规则剔除未达条件信号" else "force_close",
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
        if uploaded_file_bytes is None and input_file_path and not Path(input_file_path).exists():
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
                )
                detail_df, daily_df, equity_df, stats = analyze_all_stocks(all_data, params)
                excel_bytes = export_to_excel_bytes(detail_df, daily_df, equity_df)
            st.success("回测完成")
            st.session_state["detail_df"] = detail_df
            st.session_state["daily_df"] = daily_df
            st.session_state["equity_df"] = equity_df
            st.session_state["stats"] = stats
            st.session_state["excel_bytes"] = excel_bytes
            st.session_state["download_name"] = build_download_name(params.start_date, params.end_date)
        except Exception as exc:
            st.error(f"回测失败：{exc}")

detail_df = st.session_state.get("detail_df", pd.DataFrame())
daily_df = st.session_state.get("daily_df", pd.DataFrame())
equity_df = st.session_state.get("equity_df", pd.DataFrame())
stats = st.session_state.get("stats", {})

if isinstance(detail_df, pd.DataFrame) and "excel_bytes" in st.session_state:
    tab_summary, tab_curve, tab_details = st.tabs(["📊 绩效总览", "📈 资金曲线", "📝 交易明细"])
    with tab_summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总收益率", f"{float(stats.get('total_return_pct', 0.0)):.2f}%")
        c2.metric("胜率", f"{float(stats.get('strategy_win_rate_pct', 0.0)):.2f}%")
        c3.metric("最大回撤", f"{float(stats.get('max_drawdown_pct', 0.0)):.2f}%")
        c4.metric("交易笔数", f"{int(stats.get('executed_trades', len(detail_df)))}")
        st.download_button(
            "导出 Excel",
            data=st.session_state["excel_bytes"],
            file_name=st.session_state["download_name"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with tab_curve:
        if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
            chart_df = equity_df.copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            fig = px.line(chart_df, x="date", y="net_value", title="资金曲线")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无资金曲线数据")

    with tab_details:
        if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
            st.dataframe(format_detail_for_display(detail_df), hide_index=True)
            csv_bytes = detail_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("导出 CSV", data=csv_bytes, file_name="trade_details.csv", mime="text/csv")
        else:
            st.info("暂无交易明细")
