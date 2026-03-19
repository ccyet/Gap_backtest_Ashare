from __future__ import annotations

from io import BytesIO

import pandas as pd

from exporter import export_to_excel_bytes


def test_export_to_excel_bytes_includes_scan_sheet_when_provided() -> None:
    detail_df = pd.DataFrame({"date": ["2024-01-01"], "stock_code": ["000001.SZ"]})
    daily_df = pd.DataFrame({"date": ["2024-01-01"], "sample_count": [1]})
    equity_df = pd.DataFrame({"date": ["2024-01-01"], "net_value": [1.0]})
    scan_df = pd.DataFrame({"scan_id": [1], "rank": [1], "total_return_pct": [5.0]})

    payload = export_to_excel_bytes(detail_df, daily_df, equity_df, scan_df=scan_df)

    workbook = pd.ExcelFile(BytesIO(payload))
    assert "参数扫描" in workbook.sheet_names
