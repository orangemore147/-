from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

CACHE = Path('.cache_wave6_stock_adjusted')
STOCKS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'INTC',
    'ORCL', 'IBM', 'CSCO', 'PEP', 'MCD', 'GE', 'MA', 'BABA', 'LLY',
    'UNH', 'ASML',
]


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_stock(symbol: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f'{symbol}.csv'
    if path.exists():
        frame = pd.read_csv(path, parse_dates=['time']).set_index('time')
        frame.index = pd.DatetimeIndex(frame.index).tz_convert('UTC')
        return frame

    start = int(pd.Timestamp('2014-01-01', tz='UTC').timestamp())
    end = int(pd.Timestamp('2026-08-08', tz='UTC').timestamp())
    url = (
        f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
        f'?period1={start}&period2={end}&interval=1d&events=div%2Csplits'
    )
    last_error = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=45, headers={'User-Agent': 'Mozilla/5.0 wave6'})
            response.raise_for_status()
            payload = response.json()['chart']['result'][0]
            index = pd.to_datetime(payload['timestamp'], unit='s', utc=True).normalize()
            quote = payload['indicators']['quote'][0]
            raw_close = pd.Series(quote['close'], index=index, dtype=float)
            adjusted_values = payload['indicators'].get('adjclose', [{}])[0].get('adjclose')
            adjusted = (
                pd.Series(adjusted_values, index=index, dtype=float)
                if adjusted_values is not None
                else raw_close.copy()
            )
            ratio = adjusted / raw_close.replace(0, np.nan)
            frame = pd.DataFrame(
                {
                    'open': pd.Series(quote['open'], index=index, dtype=float) * ratio,
                    'high': pd.Series(quote['high'], index=index, dtype=float) * ratio,
                    'low': pd.Series(quote['low'], index=index, dtype=float) * ratio,
                    'close': adjusted,
                    'volume': pd.Series(quote['volume'], index=index, dtype=float),
                }
            ).dropna(subset=['open', 'high', 'low', 'close'])
            frame.index.name = 'time'
            frame.reset_index().to_csv(path, index=False)
            return frame
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f'{symbol}: {last_error}')


def stock_data() -> Dict[str, pd.DataFrame]:
    output: Dict[str, pd.DataFrame] = {}
    for symbol in STOCKS + ['SPY']:
        df = load_stock(symbol)
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['rsi3'] = rsi(df['close'], 3)
        df['ret5'] = df['close'].pct_change(5)
        df['ret126'] = df['close'].pct_change(126)
        df['vol21'] = df['close'].pct_change().rolling(21).std()
        for lookback in (20, 55):
            df[f'prior_high_{lookback}'] = df['high'].rolling(lookback).max().shift(1)
        output[symbol] = df
    return output
