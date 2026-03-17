from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from data.providers.akshare_provider import AkshareProvider


def test_fetch_daily_bars_fallback_sina_to_tencent(monkeypatch):
    calls: list[str] = []

    def sina_fail(**kwargs):
        calls.append("sina")
        raise RuntimeError("sina down")

    def tx_ok(**kwargs):
        calls.append("tencent")
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [1.0, 2.0],
                "最高": [1.2, 2.2],
                "最低": [0.9, 1.9],
                "收盘": [1.1, 2.1],
                "成交量": [100.0, 200.0],
                "成交额": [1000.0, 2000.0],
            }
        )

    def em_should_not_run(**kwargs):
        calls.append("eastmoney")
        raise AssertionError("eastmoney should not be called when tencent succeeds")

    fake_ak = SimpleNamespace(
        stock_zh_a_daily=sina_fail,
        stock_zh_a_hist_tx=tx_ok,
        stock_zh_a_hist=em_should_not_run,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_ak)

    result = AkshareProvider.fetch_daily_bars("000001.SZ", "2024-01-01", "2024-01-31", adjust="qfq")
    assert not result.empty
    assert calls == ["sina", "tencent"]
    assert result["symbol"].nunique() == 1
    assert result["symbol"].iloc[0] == "000001.SZ"


def test_fetch_daily_bars_raises_after_all_fail(monkeypatch):
    def always_fail(**kwargs):
        raise RuntimeError("network error")

    fake_ak = SimpleNamespace(
        stock_zh_a_daily=always_fail,
        stock_zh_a_hist_tx=always_fail,
        stock_zh_a_hist=always_fail,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_ak)

    try:
        AkshareProvider.fetch_daily_bars("000001.SZ", "2024-01-01", "2024-01-31", adjust="qfq")
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "新浪->腾讯->东财" in str(exc)

    assert raised
