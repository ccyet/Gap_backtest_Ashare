from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.providers.akshare_provider import AkshareProvider
from data.services import local_data_service


def test_symbol_normalization_to_standard_format():
    assert AkshareProvider.to_standard_symbol("000001") == "000001.SZ"
    assert AkshareProvider.to_standard_symbol("600519") == "600519.SH"
    assert AkshareProvider.to_standard_symbol("sz000001") == "000001.SZ"


def test_load_daily_bars_filters_and_sorts(monkeypatch, tmp_path: Path):
    import pytest

    pytest.importorskip("pyarrow")

    daily_root = tmp_path / "daily"
    cfg = {"data_source": "local_parquet", "local_data_root": str(daily_root), "default_adjust": "qfq"}

    monkeypatch.setattr(local_data_service, "ROOT", tmp_path)
    monkeypatch.setattr(local_data_service, "_read_config", lambda: cfg)

    symbol_path = daily_root / "qfq" / "000001.SZ.parquet"
    symbol_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [3.0, 1.0, 2.0],
            "high": [3.0, 1.0, 2.0],
            "low": [3.0, 1.0, 2.0],
            "close": [3.0, 1.0, 2.0],
            "volume": [30.0, 10.0, 20.0],
            "amount": [30.0, 10.0, 20.0],
        }
    ).to_parquet(symbol_path, index=False)

    result = local_data_service.load_daily_bars("000001.SZ", "2024-01-02", "2024-01-03", adjust="qfq")
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]


def test_akshare_symbol_format_for_hist_api():
    assert AkshareProvider.to_akshare_symbol("000001.SZ") == "000001"
    assert AkshareProvider.to_akshare_symbol("600519.SH") == "600519"
