from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import wave7_derivatives_edge as core

CACHE = Path('.cache_wave7_metrics_daily')
BASE = 'https://data.binance.vision/data/futures/um/daily/metrics'
SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT']
START = pd.Timestamp('2023-01-01')
END = pd.Timestamp('2026-07-31')


def fetch_day(symbol: str, day: pd.Timestamp):
    ds = day.strftime('%Y-%m-%d')
    fname = f'{symbol}-metrics-{ds}.zip'
    path = CACHE / symbol / fname
    if path.exists() and path.stat().st_size > 0:
        raw = path.read_bytes()
    else:
        url = f'{BASE}/{symbol}/{fname}'
        raw = None
        headers = {'User-Agent': 'Mozilla/5.0 wave7-research/1.0'}
        for attempt in range(4):
            try:
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                raw = r.content
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                break
            except requests.RequestException:
                if attempt == 3:
                    return None
                time.sleep(0.5 * (2 ** attempt))
        if raw is None:
            return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            member = next(x for x in zf.namelist() if x.lower().endswith('.csv'))
            with zf.open(member) as fh:
                f = pd.read_csv(fh)
    except Exception:
        return None
    f.columns = [str(c).strip().lower() for c in f.columns]
    tcol = next((c for c in f.columns if 'time' in c), None)
    if tcol is None:
        return None
    f['time'] = core.parse_time(f[tcol])
    wanted = [
        'sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio',
        'sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio'
    ]
    for c in wanted:
        f[c] = pd.to_numeric(f[c], errors='coerce') if c in f.columns else np.nan
    return f.dropna(subset=['time']).set_index('time')[wanted]


def load_metrics(symbol: str) -> pd.DataFrame:
    days = list(pd.date_range(START, END, freq='D'))
    parts = []
    with ThreadPoolExecutor(max_workers=18) as ex:
        futures = {ex.submit(fetch_day, symbol, d): d for d in days}
        done = 0
        for fut in as_completed(futures):
            x = fut.result()
            if x is not None and not x.empty:
                parts.append(x)
            done += 1
            if done % 250 == 0:
                print(symbol, 'metrics days processed', done, '/', len(days))
    if not parts:
        raise RuntimeError(f'No daily metrics for {symbol}')
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    agg = {
        'sum_open_interest':'last','sum_open_interest_value':'last',
        'count_toptrader_long_short_ratio':'last','sum_toptrader_long_short_ratio':'last',
        'count_long_short_ratio':'last','sum_taker_long_short_vol_ratio':'mean'
    }
    return out.resample('4h', origin='start_day').agg(agg)


core.CACHE = CACHE
core.SYMBOLS = SYMBOLS
core.load_metrics = load_metrics
core.main()
