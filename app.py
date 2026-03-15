from __future__ import annotations

from io import BytesIO
import inspect
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import analyze_all_stocks
from data_loader import describe_file_source, describe_tables, list_candidate_tables, load_market_data
from exporter import export_to_excel_bytes
from models import AnalysisParams, normalize_column_overrides, normalize_stock_codes, validate_params


st.set_page_config(page_title="跳空统计分析工具", layout="wide")


DETAIL_PERCENT_COLUMNS = (
    "gap_pct_vs_prev_close",
    "gross_return_pct",
    "net_return_pct",
    "mfe_pct",
    "mae_pct",
    "max_profit_pct",
    "profit_drawdown_ratio",
)
DETAIL_PRICE_COLUMNS = ("prev_close", "prev_high", "prev_low", "open", "close", "buy_price", "sell_price", "exit_ma_value")
DETAIL_NAV_COLUMNS = ("nav_before_trade", "nav_after_trade")
SUMMARY_PERCENT_COLUMNS = ("win_rate_pct", "avg_net_return_pct", "median_net_return_pct")
EQUITY_PERCENT_COLUMNS = ("drawdown_pct",)
RESULT_STATE_KEYS = ("detail_df", "daily_df", "equity_df", "stats", "excel_bytes", "download_name")


def dataframe_stretch(data: object, *, hide_index: bool = False) -> None:
    params = inspect.signature(st.dataframe).parameters
    kwargs: dict[str, object] = {"hide_index": hide_index}
    if "use_container_width" in params:
        kwargs["use_container_width"] = True
    st.dataframe(data, **kwargs)


def form_submit_button_stretch(label: str) -> bool:
    params = inspect.signature(st.form_submit_button).parameters
    kwargs: dict[str, object] = {}
    if "use_container_width" in params:
        kwargs["use_container_width"] = True
    return st.form_submit_button(label, **kwargs)


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


st.title("跳空统计分析工具")
st.caption("用于统计指定时间范围内，满足跳空条件的股票，在统一买卖规则下的表现。")
st.info("当前按单账户、单持仓回测：一笔买入对应一笔卖出，卖出前不会再次买入。净值曲线按已平仓结果累计，持仓期间净值保持不变。")

with st.expander("数据格式说明（建议先看）", expanded=False):
    st.markdown(
        """
        支持上传 `.xlsx`、`.xlsm`、`.csv` 文件，或读取 SQLite 数据库。

        文件整理建议：
        - 一行代表一只股票在一个交易日的数据。
        - 表头放在第 1 行，不要合并单元格，不要在表头上方再写标题。
        - 至少提供“交易日期、股票代码、开盘价、最高价、最低价、收盘价”6列。
        - `成交量` 可以留空，不影响核心分析。
        - 如果列名不是下面示例，也可以在页面里通过“字段映射”手动指定。
        """
    )
    dataframe_stretch(build_data_format_table(), hide_index=True)
    st.caption("推荐日期格式：`2026-03-13`、`20260313`、Excel 日期单元格。")
    st.caption("系统也会尽量自动识别常见中文列名，如“交易日期、股票代码、开盘价、最高价、最低价、收盘价、成交量”。")
    st.markdown("示例数据：")
    dataframe_stretch(build_sample_input_data(), hide_index=True)
    st.download_button(
        "下载 Excel 模板",
        data=build_template_bytes(),
        file_name="gap_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

default_db_path = str(Path.cwd() / "market_data.sqlite")
today = pd.Timestamp.today().date()
default_start = today - pd.Timedelta(days=365)
candidate_tables: list[str] = []
table_overview_df = pd.DataFrame()
file_preview: dict[str, object] | None = None

with st.container(border=True):
    st.subheader("参数设置")
    data_source_label = st.radio("行情数据来源", options=["Excel/CSV 文件", "SQLite 数据库"], horizontal=True)
    st.caption("快慢线只作为开单过滤条件，不参与跳空信号本身的判断，也不参与卖出规则。")

    with st.form("analysis_form"):
        base_left, base_mid, base_right = st.columns(3)
        db_path = default_db_path
        table_name = ""
        input_file_path = ""
        excel_sheet_name = ""
        uploaded_market_file = None

        with base_left:
            if data_source_label == "SQLite 数据库":
                db_path = st.text_input("SQLite 数据库文件路径", value=default_db_path)
                table_name = st.text_input("行情表名（留空自动识别）", value="")
            else:
                uploaded_market_file = st.file_uploader(
                    "上传 Excel/CSV 行情文件",
                    type=["xlsx", "xlsm", "csv"],
                    help="优先推荐直接上传 Excel 文件，适合普通用户快速使用。",
                )
                input_file_path = st.text_input("或填写本地文件路径（选填）", value="")
                excel_sheet_name = st.text_input("工作表名称（Excel 选填）", value="")
            start_date = st.date_input("开始日期", value=default_start)
            end_date = st.date_input("结束日期", value=today)
            stock_scope_text = st.text_area("股票代码范围", value="", help="多个股票代码可用逗号、空格或换行分隔。留空表示全市场。")

        with base_mid:
            direction_label = st.selectbox("跳空方向", options=["向上跳空", "向下跳空"])
            gap_pct = st.number_input("跳空幅度（%）", min_value=0.0, value=2.0, step=0.1)
            max_gap_filter_pct = st.number_input("最大高开/低开过滤（%）", min_value=0.0, value=9.9, step=0.1)
            use_ma_filter = st.checkbox(
                "启用快慢线开单过滤",
                value=False,
                help="勾选后，只有满足快慢线方向过滤的跳空信号，才允许开单。",
            )
            fast_ma_period = st.number_input(
                "快线周期",
                min_value=1,
                value=5,
                step=1,
                disabled=not use_ma_filter,
                help="使用信号日前已经形成的收盘均线，不使用信号当天收盘价。",
            )
            slow_ma_period = st.number_input(
                "慢线周期",
                min_value=1,
                value=20,
                step=1,
                disabled=not use_ma_filter,
                help="使用信号日前已经形成的收盘均线，不使用信号当天收盘价。",
            )

        with base_right:
            time_stop_days = st.number_input("最多持有几天", min_value=1, value=5, step=1)
            time_stop_target_pct = st.number_input("到期最低目标涨幅（%）", value=1.0, step=0.1)
            time_exit_mode_label = st.selectbox(
                "到第 N 天后怎么处理",
                options=["按原规则剔除未达条件信号", "第 N 天按收盘价结束交易"],
            )
            stop_loss_pct = st.number_input("单次亏损超过多少止损（%）", min_value=0.0, value=3.0, step=0.1)
            enable_take_profit = st.checkbox("启用固定止盈", value=True)
            take_profit_pct = st.number_input(
                "单笔盈利超过多少止盈（%）",
                min_value=0.0,
                value=5.0,
                step=0.1,
                disabled=not enable_take_profit,
            )

            st.markdown("**盈利回撤止盈**")
            enable_profit_drawdown_exit = st.checkbox("启用盈利回撤止盈", value=False)
            profit_drawdown_pct = st.number_input(
                "盈利后，如果从最高利润回落超过多少就卖出（%）",
                min_value=0.0,
                value=40.0,
                step=1.0,
                disabled=not enable_profit_drawdown_exit,
                help="例如一度赚了 10%，后来利润回落超过 40%，就卖出。",
            )

            st.markdown("**均线离场**")
            enable_ma_exit = st.checkbox("启用均线离场", value=False)
            exit_ma_period = st.number_input(
                "跌破哪条均线后卖出",
                min_value=1,
                value=10,
                step=1,
                disabled=not enable_ma_exit,
                help="用于让盈利单继续持有，等趋势转弱再退出。",
            )
            ma_exit_batches = st.number_input(
                "均线离场分几批卖出",
                min_value=2,
                max_value=3,
                value=2,
                step=1,
                disabled=not enable_ma_exit,
                help="预设 2 批，最多 3 批。触发均线离场时按批次逐步卖出。",
            )

            buy_cost_pct = st.number_input("买入成本（%）", min_value=0.0, value=0.03, step=0.01, format="%.4f")
            sell_cost_pct = st.number_input("卖出成本（%）", min_value=0.0, value=0.13, step=0.01, format="%.4f")

        with st.expander("字段映射（选填，数据库字段名不标准时再填写）"):
            map_col_1, map_col_2, map_col_3 = st.columns(3)
            with map_col_1:
                date_column = st.text_input("日期列名", value="")
                stock_code_column = st.text_input("股票代码列名", value="")
                open_column = st.text_input("开盘价列名", value="")
            with map_col_2:
                high_column = st.text_input("最高价列名", value="")
                low_column = st.text_input("最低价列名", value="")
                close_column = st.text_input("收盘价列名", value="")
            with map_col_3:
                volume_column = st.text_input("成交量列名", value="")

        submitted = form_submit_button_stretch("开始统计")

    if use_ma_filter:
        if direction_label == "向上跳空":
            st.info("当前快慢线过滤规则：只有在跳空上涨当天，开盘价高于快线且高于慢线，才允许开单；否则该信号直接过滤。")
        else:
            st.info("当前快慢线过滤规则：只有在跳空下跌当天，开盘价低于快线且低于慢线，才允许开单；否则该信号直接过滤。")

    current_column_overrides = normalize_column_overrides(
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

    if data_source_label == "SQLite 数据库" and db_path.strip() and Path(db_path).exists():
        try:
            tables = list_candidate_tables(db_path)
            candidate_tables = tables
            table_overview_df = pd.DataFrame(describe_tables(db_path))
            if tables:
                st.caption("可自动识别的行情表：" + "、".join(tables))
            else:
                st.caption("当前数据库未识别到标准行情表，请检查字段名。")
        except Exception as exc:
            st.caption(f"无法读取数据库表信息：{exc}")
    elif data_source_label == "Excel/CSV 文件":
        try:
            preview_path = input_file_path.strip() or None
            preview_bytes = uploaded_market_file.getvalue() if uploaded_market_file is not None else None
            preview_name = uploaded_market_file.name if uploaded_market_file is not None else None
            if preview_bytes is not None or preview_path:
                file_preview = describe_file_source(
                    file_path=preview_path,
                    file_bytes=preview_bytes,
                    file_name=preview_name,
                    sheet_name=excel_sheet_name.strip() or None,
                    column_overrides=current_column_overrides,
                )
        except Exception as exc:
            st.caption(f"无法读取文件结构：{exc}")

if not table_overview_df.empty:
    with st.expander("数据库表结构预览"):
        dataframe_stretch(table_overview_df, hide_index=True)

if file_preview:
    with st.expander("文件结构预览"):
        preview_rows = [
            {
                "文件名": file_preview.get("file_name", ""),
                "文件类型": file_preview.get("file_type", ""),
                "使用的工作表": file_preview.get("selected_sheet", "无"),
                "列数": file_preview.get("column_count", 0),
                "自动识别字段": "是" if file_preview.get("auto_detected") else "否",
                "字段预览": file_preview.get("columns_preview", ""),
            }
        ]
        dataframe_stretch(pd.DataFrame(preview_rows), hide_index=True)
        if file_preview.get("sheet_names"):
            st.caption("可用工作表：" + "、".join(file_preview["sheet_names"]))
        detected_fields = str(file_preview.get("detected_fields", "")).strip()
        if detected_fields:
            st.caption("识别结果：" + detected_fields)

if submitted:
    clear_result_state()
    source_type = "file" if data_source_label == "Excel/CSV 文件" else "sqlite"
    uploaded_file_bytes = uploaded_market_file.getvalue() if uploaded_market_file is not None else None
    uploaded_file_name = uploaded_market_file.name if uploaded_market_file is not None else None
    input_file_path = input_file_path.strip()
    excel_sheet_name = excel_sheet_name.strip()

    column_overrides = current_column_overrides
    params = AnalysisParams(
        data_source_type=source_type,
        db_path=db_path.strip(),
        table_name=table_name.strip() or None,
        column_overrides=column_overrides,
        excel_sheet_name=excel_sheet_name or None,
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
        buy_cost_pct=float(buy_cost_pct),
        sell_cost_pct=float(sell_cost_pct),
        time_exit_mode="strict" if time_exit_mode_label == "按原规则剔除未达条件信号" else "force_close",
    )

    errors, warnings = validate_params(params)
    for warning in warnings:
        st.warning(warning)

    if params.data_source_type == "sqlite" and column_overrides and not params.table_name and len(candidate_tables) > 1:
        st.warning("您填写了字段映射，但未指定表名。当前会自动选择第一个匹配的行情表，建议同时填写表名。")

    if params.data_source_type == "file":
        if uploaded_file_bytes is not None and input_file_path:
            st.warning("同时提供了上传文件和本地文件路径，当前会优先使用上传文件。")
        if uploaded_file_bytes is None and not input_file_path:
            errors.append("请选择 Excel/CSV 文件，或者填写本地文件路径。")
        if uploaded_file_bytes is None and input_file_path and not Path(input_file_path).exists():
            errors.append(f"找不到文件：{input_file_path}")

    if errors:
        for error in errors:
            st.error(error)
    else:
        try:
            with st.spinner("正在读取数据并计算结果，请稍候..."):
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

            st.session_state["detail_df"] = detail_df
            st.session_state["daily_df"] = daily_df
            st.session_state["equity_df"] = equity_df
            st.session_state["stats"] = stats
            st.session_state["excel_bytes"] = excel_bytes
            st.session_state["download_name"] = build_download_name(params.start_date, params.end_date)
        except Exception as exc:
            st.error(f"统计失败：{exc}")


detail_df = st.session_state.get("detail_df", pd.DataFrame())
daily_df = st.session_state.get("daily_df", pd.DataFrame())
equity_df = st.session_state.get("equity_df", pd.DataFrame())
stats = st.session_state.get("stats", {})

if isinstance(detail_df, pd.DataFrame) and "excel_bytes" in st.session_state:
    if detail_df.empty:
        st.info("本次分析没有生成完整交易结果。请检查参数设置，或确认数据库里有足够的未来交易日数据。")
    else:
        metric_1, metric_2, metric_3, metric_4, metric_5, metric_6 = st.columns(6)
        metric_1.metric("闭环交易数", f"{int(stats.get('executed_trades', len(detail_df)))}")
        metric_2.metric("最终净值", f"{float(stats.get('final_net_value', 1.0)):.4f}")
        metric_3.metric("累计收益", f"{float(stats.get('total_return_pct', 0.0)):.2f}%")
        metric_4.metric("闭环胜率", f"{float(stats.get('strategy_win_rate_pct', 0.0)):.2f}%")
        metric_5.metric("收益波动率", f"{float(stats.get('trade_return_volatility_pct', 0.0)):.2f}%")
        metric_6.metric("最大回撤", f"{float(stats.get('max_drawdown_pct', 0.0)):.2f}%")

    if stats:
        st.caption(
            "共发现 "
            f"{stats.get('signal_count', 0)} 个信号，"
            f"其中 {stats.get('closed_trade_candidates', 0)} 个信号能形成完整闭环，"
            f"最终按单账户顺序执行 {stats.get('executed_trades', 0)} 笔交易，"
            f"跳过 {stats.get('skipped_overlapping_position', 0)} 个持仓重叠信号，"
            f"剔除 {stats.get('skipped_insufficient_future', 0)} 个未来数据不足样本，"
            f"剔除 {stats.get('skipped_time_target_met', 0)} 个到期后仍高于目标且按原规则不卖出的样本，"
            f"剔除 {stats.get('skipped_no_exit', 0)} 个未触发卖出规则样本。"
        )

    st.download_button(
        "导出 Excel",
        data=st.session_state["excel_bytes"],
        file_name=st.session_state["download_name"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
        st.subheader("策略净值曲线")
        chart_df = equity_df.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        st.line_chart(chart_df.set_index("date")[["net_value"]], height=320)

        st.subheader("回撤曲线")
        dd_df = equity_df.copy()
        dd_df["date"] = pd.to_datetime(dd_df["date"])
        st.line_chart(dd_df.set_index("date")[["drawdown_pct"]], height=220)

        dataframe_stretch(format_equity_for_display(equity_df), hide_index=True)

    st.subheader("每日统计结果（按实际执行交易的买入日汇总）")
    dataframe_stretch(format_summary_for_display(daily_df), hide_index=True)

    st.subheader("交易明细结果（按账户顺序执行）")
    dataframe_stretch(format_detail_for_display(detail_df), hide_index=True)
