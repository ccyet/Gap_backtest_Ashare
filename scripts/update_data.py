from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.providers.akshare_provider import AkshareProvider
CONFIG_PATH = ROOT / "config" / "data_source.yaml"
METADATA_DIR = ROOT / "data" / "market" / "metadata"
UPDATE_LOG_PATH = METADATA_DIR / "update_log.parquet"
SYMBOLS_PATH = METADATA_DIR / "symbols.parquet"


def read_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or ":" not in text:
            continue
        key, value = text.split(":", 1)
        cfg[key.strip()] = value.strip().strip("\"'" )
    return {
        "data_source": str(cfg.get("data_source", "local_parquet")),
        "local_data_root": str(cfg.get("local_data_root", "data/market/daily")),
        "default_adjust": str(cfg.get("default_adjust", "qfq")),
    }


def _append_update_log(row: dict[str, object]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if UPDATE_LOG_PATH.exists():
        log_df = pd.read_parquet(UPDATE_LOG_PATH)
    else:
        log_df = pd.DataFrame(
            columns=["symbol", "adjust", "start_date", "end_date", "rows", "updated_at", "status", "error_message"]
        )

    updated = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    updated["updated_at"] = pd.to_datetime(updated["updated_at"], errors="coerce")
    updated.to_parquet(UPDATE_LOG_PATH, index=False)


def _clean_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = AkshareProvider.to_standard_symbol(symbol)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_columns:
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)

    out = out.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    out = out[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
    return out


def _merge_incremental(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged




def _export_symbol_to_excel(symbol: str, adjust: str, local_root: Path) -> str | None:
    parquet_path = local_root / adjust / f"{symbol}.parquet"
    if not parquet_path.exists():
        return None

    export_dir = ROOT / "data" / "market" / "exports" / adjust
    export_dir.mkdir(parents=True, exist_ok=True)
    excel_path = export_dir / f"{symbol}.xlsx"

    df = pd.read_parquet(parquet_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.to_excel(excel_path, index=False)
    return str(excel_path)

def update_one_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    local_root: Path,
    export_excel: bool,
) -> None:
    standardized_symbol = AkshareProvider.to_standard_symbol(symbol)
    symbol_path = local_root / adjust / f"{standardized_symbol}.parquet"
    symbol_path.parent.mkdir(parents=True, exist_ok=True)

    status = "success"
    error_message = ""
    rows = 0

    try:
        raw = AkshareProvider.fetch_daily_bars(
            symbol=standardized_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        cleaned = _clean_bars(raw, standardized_symbol)

        if symbol_path.exists():
            existing = pd.read_parquet(symbol_path)
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
            merged = _merge_incremental(existing, cleaned)
        else:
            merged = cleaned.sort_values("date").reset_index(drop=True)

        merged.to_parquet(symbol_path, index=False)
        rows = len(merged)
        if export_excel:
            _export_symbol_to_excel(standardized_symbol, adjust, local_root)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error_message = str(exc)

    _append_update_log(
        {
            "symbol": standardized_symbol,
            "adjust": adjust,
            "start_date": start_date,
            "end_date": end_date,
            "rows": int(rows),
            "updated_at": pd.Timestamp.utcnow(),
            "status": status,
            "error_message": error_message,
        }
    )


def update_symbol_metadata() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols = AkshareProvider.fetch_symbol_list().copy()
    if "symbol" in symbols.columns:
        symbols["symbol"] = symbols["symbol"].map(AkshareProvider.to_standard_symbol)
    symbols.to_parquet(SYMBOLS_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线更新 AKShare A 股日线到本地 parquet")
    parser.add_argument("--symbols", type=str, default="", help="逗号分隔标准代码，如 000001.SZ,600519.SH；留空则按 symbols.parquet 全量更新")
    parser.add_argument("--start-date", type=str, default="20100101", help="起始日期，支持 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y%m%d"), help="结束日期")
    parser.add_argument("--adjust", type=str, default="", choices=["", "qfq", "hfq"], help="复权类型，默认读取配置")
    parser.add_argument("--refresh-symbols", action="store_true", help="先刷新股票列表 metadata")
    parser.add_argument("--export-excel", action="store_true", help="更新完成后把对应 symbol 另存为 Excel")
    return parser.parse_args()


def _normalize_date(date_text: str) -> str:
    return pd.to_datetime(date_text).strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    config = read_config()
    if config["data_source"] != "local_parquet":
        raise ValueError("当前仅支持 local_parquet 数据源")

    local_root = ROOT / config["local_data_root"]
    adjust = args.adjust or config["default_adjust"]
    start_date = _normalize_date(args.start_date)
    end_date = _normalize_date(args.end_date)

    if args.refresh_symbols or not SYMBOLS_PATH.exists():
        update_symbol_metadata()

    if args.symbols.strip():
        symbols = [AkshareProvider.to_standard_symbol(item.strip()) for item in args.symbols.split(",") if item.strip()]
    else:
        symbol_df = pd.read_parquet(SYMBOLS_PATH) if SYMBOLS_PATH.exists() else pd.DataFrame(columns=["symbol"])
        symbols = [AkshareProvider.to_standard_symbol(symbol) for symbol in symbol_df.get("symbol", pd.Series(dtype=str)).tolist()]

    for symbol in symbols:
        update_one_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            local_root=local_root,
            export_excel=bool(args.export_excel),
        )


if __name__ == "__main__":
    main()
