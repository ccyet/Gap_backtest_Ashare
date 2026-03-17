from __future__ import annotations

from io import BytesIO
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ("date", "stock_code", "open", "high", "low", "close")
OPTIONAL_COLUMNS = ("volume",)
SUPPORTED_FILE_SUFFIXES = {".xlsx", ".xlsm", ".csv"}

COLUMN_ALIASES = {
    "date": ("date", "trade_date", "trading_date", "trade_dt", "日期", "交易日期", "交易日"),
    "stock_code": (
        "stock_code",
        "ts_code",
        "symbol",
        "ticker",
        "code",
        "security_code",
        "symbol_id",
        "股票代码",
        "证券代码",
        "代码",
    ),
    "open": ("open", "open_price", "opn", "开盘", "开盘价"),
    "high": ("high", "high_price", "hi", "最高", "最高价"),
    "low": ("low", "low_price", "lo", "最低", "最低价"),
    "close": ("close", "close_price", "settle", "settlement", "cls", "收盘", "收盘价"),
    "volume": ("volume", "vol", "volx", "turnover_volume", "成交量", "成交股数"),
}


def quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def parse_trade_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.normalize()

    raw = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    numeric_values = pd.to_numeric(series, errors="coerce")

    compact_mask = raw.str.fullmatch(r"\d{8}")
    if compact_mask.any():
        parsed.loc[compact_mask] = pd.to_datetime(raw.loc[compact_mask], format="%Y%m%d", errors="coerce")

    excel_serial_mask = parsed.isna() & numeric_values.notna() & numeric_values.between(20_000, 80_000)
    if excel_serial_mask.any():
        parsed.loc[excel_serial_mask] = pd.to_datetime(
            numeric_values.loc[excel_serial_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    unix_seconds_mask = parsed.isna() & numeric_values.notna() & numeric_values.abs().between(1_000_000_000, 9_999_999_999)
    if unix_seconds_mask.any():
        parsed.loc[unix_seconds_mask] = pd.to_datetime(
            numeric_values.loc[unix_seconds_mask],
            unit="s",
            errors="coerce",
        )

    unix_milliseconds_mask = parsed.isna() & numeric_values.notna() & numeric_values.abs().between(
        1_000_000_000_000, 9_999_999_999_999
    )
    if unix_milliseconds_mask.any():
        parsed.loc[unix_milliseconds_mask] = pd.to_datetime(
            numeric_values.loc[unix_milliseconds_mask],
            unit="ms",
            errors="coerce",
        )

    remaining_mask = parsed.isna()
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(raw.loc[remaining_mask], errors="coerce")

    return parsed.dt.normalize()


def _resolve_existing_columns(
    existing: dict[str, str],
    column_overrides: dict[str, str] | None = None,
) -> dict[str, str] | None:
    resolved: dict[str, str] = {}
    overrides = column_overrides or {}

    for canonical, requested_name in overrides.items():
        actual_name = existing.get(requested_name.lower())
        if actual_name is None:
            return None
        resolved[canonical] = actual_name

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in resolved:
            continue
        for alias in aliases:
            if alias.lower() in existing:
                resolved[canonical] = existing[alias.lower()]
                break

    if all(column in resolved for column in REQUIRED_COLUMNS):
        return resolved
    return None


def _calculate_query_window(
    start_date: str,
    end_date: str,
    lookback_days: int,
    lookahead_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    signal_start = pd.to_datetime(start_date)
    signal_end = pd.to_datetime(end_date)
    extra_history = max(lookback_days * 3, lookback_days + 10)
    extra_future = max(lookahead_days * 3, lookahead_days + 10)
    query_start = signal_start - pd.Timedelta(days=extra_history)
    query_end = signal_end + pd.Timedelta(days=extra_future)
    return query_start, query_end


def _normalize_loaded_data(
    df: pd.DataFrame,
    column_map: dict[str, str],
    start_date: str,
    end_date: str,
    stock_codes: tuple[str, ...] | None,
    lookback_days: int,
    lookahead_days: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    renamed = df.rename(columns={actual: canonical for canonical, actual in column_map.items()}).copy()
    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA

    normalized = renamed[list(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)].copy()
    normalized["date"] = parse_trade_dates(normalized["date"])
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized["stock_code"] = normalized["stock_code"].astype(str).str.strip().str.upper()
    normalized = normalized.dropna(subset=["date", "stock_code", "open", "high", "low", "close"])

    query_start, query_end = _calculate_query_window(start_date, end_date, lookback_days, lookahead_days)
    mask = (normalized["date"] >= query_start.normalize()) & (normalized["date"] <= query_end.normalize())
    normalized = normalized.loc[mask]

    normalized_codes = tuple(code.strip().upper() for code in (stock_codes or ()) if code.strip())
    if normalized_codes:
        normalized = normalized.loc[normalized["stock_code"].isin(normalized_codes)]

    return normalized.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _inspect_tables(conn: sqlite3.Connection) -> list[str]:
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """
    return [row[0] for row in conn.execute(query).fetchall()]


def _resolve_columns(
    conn: sqlite3.Connection,
    table_name: str,
    column_overrides: dict[str, str] | None = None,
) -> dict[str, str] | None:
    pragma_query = f"PRAGMA table_info({quote_ident(table_name)})"
    rows = conn.execute(pragma_query).fetchall()
    if not rows:
        return None

    existing = {str(row[1]).lower(): str(row[1]) for row in rows}
    return _resolve_existing_columns(existing, column_overrides=column_overrides)


def _resolve_frame_columns(
    df: pd.DataFrame,
    column_overrides: dict[str, str] | None = None,
) -> dict[str, str] | None:
    existing = {str(column).lower(): str(column) for column in df.columns}
    return _resolve_existing_columns(existing, column_overrides=column_overrides)


def list_candidate_tables(db_path: str) -> list[str]:
    path = Path(db_path)
    if not path.exists():
        return []

    with sqlite3.connect(path) as conn:
        tables = _inspect_tables(conn)
        return [table for table in tables if _resolve_columns(conn, table) is not None]


def describe_tables(db_path: str) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []

    overviews: list[dict[str, Any]] = []
    with sqlite3.connect(path) as conn:
        for table in _inspect_tables(conn):
            rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            columns = [str(row[1]) for row in rows]
            mapping = _resolve_columns(conn, table)
            detected_fields = ", ".join(
                f"{canonical}->{actual}" for canonical, actual in sorted((mapping or {}).items())
            )
            overviews.append(
                {
                    "table_name": table,
                    "column_count": len(columns),
                    "columns_preview": ", ".join(columns[:8]) + (" ..." if len(columns) > 8 else ""),
                    "auto_detected": "是" if mapping else "否",
                    "detected_fields": detected_fields,
                }
            )
    return overviews


def _normalize_file_suffix(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError("目前仅支持 .xlsx、.xlsm 和 .csv 文件。")
    return suffix


def _build_file_source(
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
) -> tuple[str | BytesIO, str]:
    if file_bytes is not None:
        display_name = file_name or "uploaded_file"
        _normalize_file_suffix(display_name)
        return BytesIO(file_bytes), display_name

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到文件：{file_path}")
        _normalize_file_suffix(path.name)
        return str(path), path.name

    raise ValueError("请提供文件路径或上传文件内容。")


def _read_csv_source(source: str | BytesIO) -> pd.DataFrame:
    encodings = ("utf-8-sig", "gb18030", "utf-8")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            if isinstance(source, BytesIO):
                source.seek(0)
                return pd.read_csv(source, encoding=encoding)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    raise ValueError("CSV 文件读取失败。")


def list_file_sheets(
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
) -> list[str]:
    source, display_name = _build_file_source(file_path=file_path, file_bytes=file_bytes, file_name=file_name)
    suffix = _normalize_file_suffix(display_name)
    if suffix == ".csv":
        return []

    with pd.ExcelFile(source) as workbook:
        return list(workbook.sheet_names)


def describe_file_source(
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
    sheet_name: str | None = None,
    column_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    source, display_name = _build_file_source(file_path=file_path, file_bytes=file_bytes, file_name=file_name)
    suffix = _normalize_file_suffix(display_name)

    if suffix == ".csv":
        preview_df = _read_csv_source(source)
        sheets: list[str] = []
        resolved_sheet = None
    else:
        with pd.ExcelFile(source) as workbook:
            sheets = list(workbook.sheet_names)
            resolved_sheet = sheet_name.strip() if sheet_name and sheet_name.strip() else sheets[0]
            if resolved_sheet not in sheets:
                raise ValueError(f"工作表 {resolved_sheet} 不存在。")
            preview_df = pd.read_excel(workbook, sheet_name=resolved_sheet)

    mapping = _resolve_frame_columns(preview_df, column_overrides=column_overrides)
    preview_columns = [str(column) for column in preview_df.columns]
    return {
        "file_name": display_name,
        "file_type": suffix,
        "sheet_names": sheets,
        "selected_sheet": resolved_sheet,
        "column_count": len(preview_columns),
        "columns_preview": ", ".join(preview_columns[:10]) + (" ..." if len(preview_columns) > 10 else ""),
        "auto_detected": bool(mapping),
        "detected_fields": ", ".join(
            f"{canonical}->{actual}" for canonical, actual in sorted((mapping or {}).items())
        ),
    }


def load_file_data(
    start_date: str,
    end_date: str,
    stock_codes: tuple[str, ...] | None = None,
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
    sheet_name: str | None = None,
    column_overrides: dict[str, str] | None = None,
    lookback_days: int = 0,
    lookahead_days: int = 0,
) -> pd.DataFrame:
    source, display_name = _build_file_source(file_path=file_path, file_bytes=file_bytes, file_name=file_name)
    suffix = _normalize_file_suffix(display_name)

    if suffix == ".csv":
        raw_df = _read_csv_source(source)
    else:
        with pd.ExcelFile(source) as workbook:
            target_sheet = sheet_name.strip() if sheet_name and sheet_name.strip() else workbook.sheet_names[0]
            if target_sheet not in workbook.sheet_names:
                raise ValueError(f"工作表 {target_sheet} 不存在。")
            raw_df = pd.read_excel(workbook, sheet_name=target_sheet)

    column_map = _resolve_frame_columns(raw_df, column_overrides=column_overrides)
    if column_map is None:
        raise ValueError("文件中没有找到可用的行情字段。请按页面里的数据格式说明整理表头，或填写字段映射。")

    return _normalize_loaded_data(
        raw_df,
        column_map=column_map,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        lookback_days=lookback_days,
        lookahead_days=lookahead_days,
    )


def _pick_table(
    conn: sqlite3.Connection,
    requested_table: str | None,
    column_overrides: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    if requested_table:
        mapping = _resolve_columns(conn, requested_table, column_overrides=column_overrides)
        if mapping is None:
            raise ValueError(f"表 {requested_table} 不存在，或者字段映射与实际字段不匹配。")
        return requested_table, mapping

    candidates = []
    for table in _inspect_tables(conn):
        mapping = _resolve_columns(conn, table, column_overrides=column_overrides)
        if mapping is not None:
            candidates.append((table, mapping))

    if not candidates:
        raise ValueError("数据库里没有找到可用的行情表。请检查表名或字段映射。")

    preferred_names = ("daily_prices", "stock_daily", "prices", "daily", "quotes")
    for preferred in preferred_names:
        for table, mapping in candidates:
            if table.lower() == preferred:
                return table, mapping

    return candidates[0]


def _inspect_date_style(conn: sqlite3.Connection, table_name: str, date_column: str) -> str | None:
    query = f"""
        SELECT {quote_ident(date_column)}
        FROM {quote_ident(table_name)}
        WHERE {quote_ident(date_column)} IS NOT NULL
        LIMIT 1
    """
    row = conn.execute(query).fetchone()
    if row is None:
        return None

    value = str(row[0]).strip()
    if re.fullmatch(r"\d{8}", value):
        return "compact"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}([ T].*)?", value):
        return "iso"
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}([ T].*)?", value):
        return "slash"
    return "normalized_text"


def _build_date_filter(date_column: str, date_style: str | None) -> tuple[str, str, str]:
    quoted = quote_ident(date_column)
    if date_style == "compact":
        return f"CAST({quoted} AS TEXT)", "%Y%m%d", "%Y%m%d"
    if date_style == "slash":
        return f"REPLACE(SUBSTR(CAST({quoted} AS TEXT), 1, 10), '/', '-')", "%Y-%m-%d", "%Y-%m-%d"
    return f"REPLACE(SUBSTR(CAST({quoted} AS TEXT), 1, 10), '/', '-')", "%Y-%m-%d", "%Y-%m-%d"


def load_stock_data(
    db_path: str,
    start_date: str,
    end_date: str,
    stock_codes: tuple[str, ...] | None = None,
    table_name: str | None = None,
    column_overrides: dict[str, str] | None = None,
    lookback_days: int = 0,
    lookahead_days: int = 0,
) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据库文件：{db_path}")

    query_start, query_end = _calculate_query_window(start_date, end_date, lookback_days, lookahead_days)

    with sqlite3.connect(path) as conn:
        resolved_table, column_map = _pick_table(conn, table_name, column_overrides=column_overrides)
        date_style = _inspect_date_style(conn, resolved_table, column_map["date"])
        date_expression, sql_start_format, sql_end_format = _build_date_filter(column_map["date"], date_style)

        select_parts = []
        for canonical in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            if canonical in column_map:
                select_parts.append(
                    f"{quote_ident(column_map[canonical])} AS {quote_ident(canonical)}"
                )
            else:
                select_parts.append(f"NULL AS {quote_ident(canonical)}")

        query = [
            f"SELECT {', '.join(select_parts)}",
            f"FROM {quote_ident(resolved_table)}",
        ]
        filters: list[str] = []
        params: list[object] = []

        filters.append(f"{date_expression} BETWEEN ? AND ?")
        params.extend([query_start.strftime(sql_start_format), query_end.strftime(sql_end_format)])

        normalized_codes = tuple(code.strip().upper() for code in (stock_codes or ()) if code.strip())
        if normalized_codes:
            placeholders = ", ".join("?" for _ in normalized_codes)
            filters.append(f"UPPER({quote_ident(column_map['stock_code'])}) IN ({placeholders})")
            params.extend(normalized_codes)

        if filters:
            query.append("WHERE " + " AND ".join(filters))

        query.append(f"ORDER BY {quote_ident(column_map['stock_code'])}, {quote_ident(column_map['date'])}")
        sql = "\n".join(query)

        df = pd.read_sql_query(sql, conn, params=params)

    return _normalize_loaded_data(
        df,
        column_map={column: column for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS},
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        lookback_days=lookback_days,
        lookahead_days=lookahead_days,
    )




def _list_local_symbols(local_data_root: Path, adjust: str) -> list[str]:
    metadata_path = local_data_root.parent / "metadata" / "symbols.parquet"
    if metadata_path.exists():
        try:
            meta = pd.read_parquet(metadata_path)
            if "symbol" in meta.columns:
                values = meta["symbol"].astype(str).str.strip().str.upper().tolist()
                return sorted([item for item in values if item])
        except Exception:
            pass

    adjust_dir = local_data_root / adjust
    if not adjust_dir.exists():
        return []
    return sorted(path.stem.upper() for path in adjust_dir.glob("*.parquet"))


def load_local_parquet_data(
    start_date: str,
    end_date: str,
    stock_codes: tuple[str, ...] | None = None,
    lookback_days: int = 0,
    lookahead_days: int = 0,
    local_data_root: str = "data/market/daily",
    adjust: str = "qfq",
) -> pd.DataFrame:
    root = Path(local_data_root)
    adjust_dir = root / adjust
    if not adjust_dir.exists():
        raise FileNotFoundError(f"本地数据目录不存在：{adjust_dir}")

    normalized_codes = tuple(code.strip().upper() for code in (stock_codes or ()) if code.strip())
    symbols = list(normalized_codes) if normalized_codes else _list_local_symbols(root, adjust)
    if not symbols:
        raise ValueError("本地 parquet 未找到可用股票数据。")

    frames: list[pd.DataFrame] = []
    missing_symbols: list[str] = []
    for symbol in symbols:
        file_path = adjust_dir / f"{symbol}.parquet"
        if not file_path.exists():
            missing_symbols.append(symbol)
            continue

        frame = pd.read_parquet(file_path)
        if "stock_code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "stock_code"})
        for required in REQUIRED_COLUMNS:
            if required not in frame.columns:
                raise ValueError(f"{file_path.name} 缺少关键列：{required}")

        frame = frame.copy()
        frame["stock_code"] = frame["stock_code"].astype(str).str.upper()
        frames.append(frame)

    if normalized_codes and missing_symbols:
        raise FileNotFoundError(f"以下股票缺少本地 parquet 文件：{', '.join(missing_symbols)}")

    if not frames:
        raise ValueError("本地 parquet 读取结果为空，请检查股票池或日期区间。")

    merged = pd.concat(frames, ignore_index=True)
    normalized = _normalize_loaded_data(
        merged,
        column_map={column: column for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS},
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        lookback_days=lookback_days,
        lookahead_days=lookahead_days,
    )
    if normalized.empty:
        raise ValueError("本地 parquet 过滤后无可用数据，请检查回测区间或股票池。")
    return normalized

def load_market_data(
    source_type: str,
    start_date: str,
    end_date: str,
    stock_codes: tuple[str, ...] | None = None,
    table_name: str | None = None,
    column_overrides: dict[str, str] | None = None,
    lookback_days: int = 0,
    lookahead_days: int = 0,
    db_path: str | None = None,
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
    sheet_name: str | None = None,
    local_data_root: str = "data/market/daily",
    adjust: str = "qfq",
) -> pd.DataFrame:
    if source_type == "local_parquet":
        return load_local_parquet_data(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
            local_data_root=local_data_root,
            adjust=adjust,
        )

    if source_type == "sqlite":
        return load_stock_data(
            db_path=db_path or "",
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            table_name=table_name,
            column_overrides=column_overrides,
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
        )

    if source_type == "file":
        return load_file_data(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            file_path=file_path,
            file_bytes=file_bytes,
            file_name=file_name,
            sheet_name=sheet_name,
            column_overrides=column_overrides,
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
        )

    raise ValueError("不支持的数据来源类型。")
