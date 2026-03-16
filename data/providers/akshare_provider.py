from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AkshareProvider:
    """akshare 调用封装层（仅此模块允许直接依赖 akshare）。"""

    @staticmethod
    def to_standard_symbol(raw_symbol: str) -> str:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol 不能为空")

        # 已是标准格式：000001.SZ / 600519.SH
        if "." in symbol:
            code, market = symbol.split(".", 1)
            if len(code) == 6 and market in {"SZ", "SH", "BJ"}:
                return f"{code}.{market}"

        # 兼容 AKShare 常见格式：sz000001 / sh600519
        if symbol.startswith("SZ") and len(symbol) == 8 and symbol[2:].isdigit():
            return f"{symbol[2:]}.SZ"
        if symbol.startswith("SH") and len(symbol) == 8 and symbol[2:].isdigit():
            return f"{symbol[2:]}.SH"
        if symbol.startswith("BJ") and len(symbol) == 8 and symbol[2:].isdigit():
            return f"{symbol[2:]}.BJ"

        # 仅 6 位数字时，按主板常见规则兜底映射
        if len(symbol) == 6 and symbol.isdigit():
            if symbol.startswith(("5", "6", "9")):
                return f"{symbol}.SH"
            if symbol.startswith(("0", "1", "2", "3")):
                return f"{symbol}.SZ"
            if symbol.startswith(("4", "8")):
                return f"{symbol}.BJ"

        raise ValueError(f"无法识别 symbol 格式: {raw_symbol}")

    @staticmethod
    def to_akshare_symbol(standard_symbol: str) -> str:
        # stock_zh_a_hist 的 symbol 参数是 6 位数字代码（例如 000001 / 600519）
        code, _market = AkshareProvider.to_standard_symbol(standard_symbol).split(".", 1)
        return code

    @staticmethod
    def fetch_symbol_list() -> pd.DataFrame:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        if df.empty:
            return pd.DataFrame(columns=["symbol", "name"])

        renamed = df.rename(columns={"code": "raw_code", "name": "name"}).copy()
        renamed["symbol"] = renamed["raw_code"].map(AkshareProvider.to_standard_symbol)
        return renamed[["symbol", "name"]].drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    @staticmethod
    def fetch_daily_bars(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        import akshare as ak

        ak_symbol = AkshareProvider.to_akshare_symbol(symbol)
        raw_df = ak.stock_zh_a_hist(
            symbol=ak_symbol,
            period="daily",
            start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
            adjust=adjust,
        )
        if raw_df.empty:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])

        # AKShare中文列名映射
        mapping = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
        out = raw_df.rename(columns=mapping).copy()
        out["symbol"] = AkshareProvider.to_standard_symbol(symbol)

        for required in ["date", "open", "high", "low", "close"]:
            if required not in out.columns:
                out[required] = pd.NA

        if "volume" not in out.columns:
            out["volume"] = pd.NA
        if "amount" not in out.columns:
            out["amount"] = pd.NA

        return out[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
