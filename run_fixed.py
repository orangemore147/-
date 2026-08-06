from __future__ import annotations

import io
import zipfile
from typing import List

import pandas as pd

import backtest as b


def load_symbol_fixed(symbol: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    for ym in b.month_range(b.START_MONTH, b.END_MONTH):
        fname = f"{symbol}-{b.INTERVAL}-{ym}.zip"
        url = f"{b.BASE_URL}/{symbol}/{b.INTERVAL}/{fname}"
        raw = b.download_zip(url, b.CACHE / fname)
        if raw is None:
            print(f"SKIP missing {fname}")
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(member) as fh:
                    df = pd.read_csv(fh, header=None, names=cols, dtype=str)
        except Exception as exc:
            print(f"SKIP invalid {fname}: {exc}")
            continue

        # Binance archive formats are not fully consistent: some CSV files
        # include a literal header row. Coercion removes that row safely.
        numeric_cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
        parts.append(df)

    if not parts:
        raise RuntimeError(f"No data downloaded for {symbol}")

    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = (
        df[["time", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]]
        .dropna()
        .drop_duplicates("time")
        .set_index("time")
        .sort_index()
    )
    return df


b.load_symbol = load_symbol_fixed

if __name__ == "__main__":
    b.main()
