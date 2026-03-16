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

        if "." in symbol:
            code, market = symbol.split(".", 1)
            if len(code) == 6 and market in {"SZ", "SH", "BJ"}:
                return f"{code}.{market}"

        if symbol.startswith("SZ") and len(symbol) == 8 and symbol[2:].isdigit():
            return f"{symbol[2:]}.SZ"
        if symbol.startswith("SH") and len(symbol) == 8 and symbol[2:].isdigit():
            return f"{symbol[2:]}.SH"
        if symbol.startswith("BJ") and len(symbol) == 8 and symbol[2:].isdigit():
            return f"{symbol[2:]}.BJ"

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
        # 东财源 stock_zh_a_hist 使用 6 位代码
        code, _market = AkshareProvider.to_standard_symbol(standard_symbol).split(".", 1)
        return code

    @staticmethod
    def to_akshare_prefixed_symbol(standard_symbol: str) -> str:
        # 新浪/腾讯常用 sz000001 / sh600519
        code, market = AkshareProvider.to_standard_symbol(standard_symbol).split(".", 1)
        return f"{market.lower()}{code}"

    @staticmethod
    def _empty_bars() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])

    @staticmethod
    def _normalize_bars(raw_df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if raw_df.empty:
            return AkshareProvider._empty_bars()

        mapping = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }

        out = raw_df.rename(columns=mapping).copy()
        if "date" not in out.columns:
            out = out.reset_index()
            out = out.rename(columns={out.columns[0]: "date"})

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        out = out[out["date"].between(start_ts, end_ts)]

        for required in ["open", "high", "low", "close", "volume", "amount"]:
            if required not in out.columns:
                out[required] = pd.NA

        out["symbol"] = AkshareProvider.to_standard_symbol(symbol)
        return out[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)

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
    def _fetch_from_sina(ak: object, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        prefixed = AkshareProvider.to_akshare_prefixed_symbol(symbol)
        if not hasattr(ak, "stock_zh_a_daily"):
            raise AttributeError("akshare 不存在 stock_zh_a_daily")
        func = getattr(ak, "stock_zh_a_daily")
        return func(
            symbol=prefixed,
            start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
            adjust=adjust,
        )

    @staticmethod
    def _fetch_from_tencent(ak: object, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        prefixed = AkshareProvider.to_akshare_prefixed_symbol(symbol)
        for name in ["stock_zh_a_hist_tx", "stock_zh_a_daily_tx"]:
            if not hasattr(ak, name):
                continue
            func = getattr(ak, name)
            # 兼容不同版本签名
            for kwargs in (
                {
                    "symbol": prefixed,
                    "start_date": pd.to_datetime(start_date).strftime("%Y%m%d"),
                    "end_date": pd.to_datetime(end_date).strftime("%Y%m%d"),
                    "adjust": adjust,
                },
                {
                    "symbol": prefixed,
                    "start_date": pd.to_datetime(start_date).strftime("%Y%m%d"),
                    "end_date": pd.to_datetime(end_date).strftime("%Y%m%d"),
                },
            ):
                try:
                    return func(**kwargs)
                except TypeError:
                    continue
        raise AttributeError("akshare 不存在可用腾讯源接口")

    @staticmethod
    def _fetch_from_eastmoney(ak: object, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        if not hasattr(ak, "stock_zh_a_hist"):
            raise AttributeError("akshare 不存在 stock_zh_a_hist")

        return ak.stock_zh_a_hist(
            symbol=AkshareProvider.to_akshare_symbol(symbol),
            period="daily",
            start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
            adjust=adjust,
        )

    @staticmethod
    def fetch_daily_bars(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        import akshare as ak

        errors: list[str] = []
        sources = [
            ("sina", AkshareProvider._fetch_from_sina),
            ("tencent", AkshareProvider._fetch_from_tencent),
            ("eastmoney", AkshareProvider._fetch_from_eastmoney),
        ]

        for source_name, fetcher in sources:
            try:
                raw_df = fetcher(ak, symbol, start_date, end_date, adjust)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source_name}: {exc}")
                continue

            normalized = AkshareProvider._normalize_bars(raw_df, symbol, start_date, end_date)
            if not normalized.empty:
                return normalized

        if errors:
            raise RuntimeError("日线下载失败，已尝试新浪->腾讯->东财: " + " | ".join(errors))
        return AkshareProvider._empty_bars()
