from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import wave7_derivatives_edge as w7

BASE = 'https://data.binance.vision/data/futures/um/daily/metrics'
CACHE = Path('.cache_wave7_metrics_daily')
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
START = pd.Timestamp('2023-01-01', tz='UTC')
END = pd.Timestamp('2026-07-31', tz='UTC')
WANTED = [
    'sum_open_interest',
    'sum_open_interest_value',
    'count_toptrader_long_short_ratio',
    'sum_toptrader_long_short_ratio',
    'count_long_short_ratio',
    'sum_taker_long_short_vol_ratio',
]


def download_one(symbol: str, day: pd.Timestamp) -> Path | None:
    ds = day.strftime('%Y-%m-%d')
    fname = f'{symbol}-metrics-{ds}.zip'
    path = CACHE / symbol / fname
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f'{BASE}/{symbol}/{fname}'
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0 wave7-oi-research/1.0'})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return path
    except requests.RequestException:
        return None


def parse_one(path: Path) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(path) as zf:
            member = next(x for x in zf.namelist() if x.lower().endswith('.csv'))
            with zf.open(member) as fh:
                frame = pd.read_csv(fh)
    except Exception:
        return None

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    tcol = next((c for c in frame.columns if 'time' in c), None)
    if tcol is None:
        return None
    frame['time'] = w7.parse_time(frame[tcol])
    for c in WANTED:
        frame[c] = pd.to_numeric(frame[c], errors='coerce') if c in frame.columns else np.nan
    frame = frame.dropna(subset=['time']).set_index('time').sort_index()
    return frame[WANTED]


def load_metrics_daily(symbol: str) -> pd.DataFrame:
    days = list(pd.date_range(START, END, freq='D', tz='UTC'))
    print(f'Downloading daily OI metrics for {symbol}: {len(days)} days')
    with ThreadPoolExecutor(max_workers=24) as pool:
        paths = list(pool.map(lambda d: download_one(symbol, d), days))
    valid_paths = [p for p in paths if p is not None]
    print(f'{symbol}: downloaded/found {len(valid_paths)} daily metric files')
    parts = []
    for path in valid_paths:
        part = parse_one(path)
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError(f'No daily metrics for {symbol}')
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    agg = {
        'sum_open_interest': 'last',
        'sum_open_interest_value': 'last',
        'count_toptrader_long_short_ratio': 'last',
        'sum_toptrader_long_short_ratio': 'last',
        'count_long_short_ratio': 'last',
        'sum_taker_long_short_vol_ratio': 'mean',
    }
    result = out.resample('4h', origin='start_day').agg(agg)
    print(f'{symbol}: metric range {result.index.min()} -> {result.index.max()}, rows={len(result):,}')
    return result


w7.SYMBOLS = SYMBOLS
w7.START_MONTH = '2023-01'
w7.END_MONTH = '2026-07'
w7.OUT = Path('results_wave7_daily')
w7.CACHE = CACHE
w7.load_metrics = load_metrics_daily

if __name__ == '__main__':
    w7.main()
