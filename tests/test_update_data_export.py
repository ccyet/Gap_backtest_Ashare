from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.update_data as upd


def test_update_one_symbol_export_excel(tmp_path: Path, monkeypatch):
    import pytest

    pytest.importorskip("pyarrow")
    monkeypatch.setattr(upd, "ROOT", tmp_path)
    monkeypatch.setattr(upd, "METADATA_DIR", tmp_path / "data" / "market" / "metadata")
    monkeypatch.setattr(upd, "UPDATE_LOG_PATH", tmp_path / "data" / "market" / "metadata" / "update_log.parquet")

    fake_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
            "amount": [100.0, 200.0],
        }
    )

    monkeypatch.setattr(upd.AkshareProvider, "fetch_daily_bars", staticmethod(lambda **kwargs: fake_df))
    monkeypatch.setattr(upd.AkshareProvider, "to_standard_symbol", staticmethod(lambda x: "000001.SZ"))

    local_root = tmp_path / "data" / "market" / "daily"
    upd.update_one_symbol("000001.SZ", "2024-01-01", "2024-01-02", "qfq", local_root, export_excel=True)

    excel_path = tmp_path / "data" / "market" / "exports" / "qfq" / "000001.SZ.xlsx"
    assert excel_path.exists()
