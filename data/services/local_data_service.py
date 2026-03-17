from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "data_source.yaml"


def _read_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return {"data_source": "local_parquet", "local_data_root": "data/market/daily", "default_adjust": "qfq"}
    loaded: dict[str, str] = {}
    for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or ":" not in text:
            continue
        key, value = text.split(":", 1)
        loaded[key.strip()] = value.strip().strip("\"'" )
    return {
        "data_source": str(loaded.get("data_source", "local_parquet")),
        "local_data_root": str(loaded.get("local_data_root", "data/market/daily")),
        "default_adjust": str(loaded.get("default_adjust", "qfq")),
    }


def _daily_file(symbol: str, adjust: str) -> Path:
    config = _read_config()
    local_root = ROOT / config["local_data_root"]
    return local_root / adjust / f"{symbol}.parquet"


def _normalize_daily_df(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    result = result[result["date"].between(start_ts, end_ts)]
    return result.reset_index(drop=True)


def load_daily_bars(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    file_path = _daily_file(symbol=symbol, adjust=adjust)
    if not file_path.exists():
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
    df = pd.read_parquet(file_path)
    return _normalize_daily_df(df, start_date, end_date)


def load_many_daily_bars(
    symbols: list[str],
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        data[symbol] = load_daily_bars(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)
    return data


def load_symbol_list() -> pd.DataFrame:
    metadata_file = ROOT / "data" / "market" / "metadata" / "symbols.parquet"
    if not metadata_file.exists():
        return pd.DataFrame(columns=["symbol", "name"])
    df = pd.read_parquet(metadata_file)
    if "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol", "name"])
    keep_columns = [column for column in ["symbol", "name"] if column in df.columns]
    return df[keep_columns].drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)
