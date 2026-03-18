from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter


def _format_sheet(writer: pd.ExcelWriter, sheet_name: str) -> None:
    sheet = writer.sheets[sheet_name]
    sheet.freeze_panes = "A2"

    for cell in sheet[1]:
        cell.font = Font(bold=True)

    for column_index, column_cells in enumerate(sheet.columns, start=1):
        values = [
            str(cell.value) if cell.value is not None else "" for cell in column_cells
        ]
        max_len = min(max(len(value) for value in values) + 2, 24)
        sheet.column_dimensions[get_column_letter(column_index)].width = max_len


def _write_sheets(
    writer: pd.ExcelWriter,
    detail_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    scan_df: pd.DataFrame | None = None,
) -> None:
    detail_df.to_excel(writer, sheet_name="个股明细", index=False)
    daily_df.to_excel(writer, sheet_name="每日统计", index=False)
    equity_df.to_excel(writer, sheet_name="策略净值", index=False)
    if scan_df is not None and not scan_df.empty:
        scan_df.to_excel(writer, sheet_name="参数扫描", index=False)

    sheet_names = ["个股明细", "每日统计", "策略净值"]
    if scan_df is not None and not scan_df.empty:
        sheet_names.append("参数扫描")
    for sheet_name in sheet_names:
        _format_sheet(writer, sheet_name)


def export_to_excel(
    detail_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    output_path: str | Path,
    scan_df: pd.DataFrame | None = None,
) -> str:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _write_sheets(writer, detail_df, daily_df, equity_df, scan_df=scan_df)
    return str(output_path)


def export_to_excel_bytes(
    detail_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    scan_df: pd.DataFrame | None = None,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _write_sheets(writer, detail_df, daily_df, equity_df, scan_df=scan_df)
    buffer.seek(0)
    return buffer.getvalue()
