from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path('results_wave9')
SYMBOLS = [
    'BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT',
    'ADAUSDT','DOGEUSDT','LINKUSDT','LTCUSDT','AVAXUSDT',
]
TRAIN_END = pd.Timestamp('2024-12-31 23:59:59', tz='UTC')
TEST_START = pd.Timestamp('2025-01-01', tz='UTC')
END = pd.Timestamp('2026-07-31 23:59:59', tz='UTC')
ONE_WAY_COST = 0.0008
DAYS_YEAR = 365.25


@dataclass(frozen=True)
class Config:
    family: str
    lookback: int
    side_n: int
    rebalance: int

    @property
    def name(self) -> str:
        return f'{self.family}_lb{self.lookback}_n{self.side_n}_r{self.rebalance}'


def load_data():
    w3.START_MONTH = '2023-01'
    w3.END_MONTH = '2026-07'
    w3.CACHE = Path('.cache_wave3')
    closes = {}
    lows = {}
    funding = {}
    for s in SYMBOLS:
        print('Loading', s)
        k = w3.load_kline(s, 'perp')
        f = w3.load_funding(s)
        closes[s] = k['close'].resample('1D').last()
        lows[s] = k['low'].resample('1D').min()
        funding[s] = f.resample('1D').sum()
    idx = closes['BTCUSDT'].index
    for s in SYMBOLS[1:]:
        idx = idx.intersection(closes[s].index)
    idx = idx.sort_values()
    close = pd.DataFrame({s: closes[s].reindex(idx) for s in SYMBOLS}, index=idx)
    low = pd.DataFrame({s: lows[s].reindex(idx) for s in SYMBOLS}, index=idx)
    fund = pd.DataFrame({s: funding[s].reindex(idx) for s in SYMBOLS}, index=idx).fillna(0.0)
    return close, low, fund


def make_score(close: pd.DataFrame, low: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.family == 'conventional_reversal':
        # Lower past return = stronger long candidate; higher = short candidate.
        return close.pct_change(cfg.lookback)
    if cfg.family == 'past_trough_reversal':
        # PTL-inspired anchor: distance from the lowest price observed in formation window.
        trough = low.rolling(cfg.lookback, min_periods=cfg.lookback).min()
        return close / trough - 1.0
    if cfg.family == 'trough_adjusted_reversal':
        # Require both weak total return and closeness to the past trough.
        total = close.pct_change(cfg.lookback)
        trough = low.rolling(cfg.lookback, min_periods=cfg.lookback).min()
        ptl = close / trough - 1.0
        return 0.5 * total.rank(axis=1, pct=True) + 0.5 * ptl.rank(axis=1, pct=True)
    raise ValueError(cfg.family)


def weights_from_score(score: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    raw = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for i in range(0, len(score), cfg.rebalance):
        row = score.iloc[i].dropna().sort_values()
        if len(row) < 2 * cfg.side_n:
            raw.iloc[i] = 0.0
            continue
        raw.iloc[i] = 0.0
        long_names = row.head(cfg.side_n).index
        short_names = row.tail(cfg.side_n).index
        raw.loc[score.index[i], long_names] = 0.5 / cfg.side_n
        raw.loc[score.index[i], short_names] = -0.5 / cfg.side_n
    # Signal is known only after today's close; trade from next day's return.
    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def strategy_returns(close: pd.DataFrame, funding: pd.DataFrame, w: pd.DataFrame, cost_mult: float = 1.0):
    r = close.pct_change().fillna(0.0)
    gross = (w * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    # Positive funding: longs pay, shorts receive. Negative funding reverses the sign.
    funding_cost = (w * funding).sum(axis=1)
    return gross - turnover * ONE_WAY_COST * cost_mult - funding_cost, turnover


def metrics(r: pd.Series, turnover: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    r = r.loc[start:end].dropna()
    turnover = turnover.reindex(r.index).fillna(0.0)
    if len(r) < 180:
        return {}
    eq = (1 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).total_seconds() / (365.25 * 86400), 0.01)
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * math.sqrt(DAYS_YEAR)
    sharpe = r.mean() * DAYS_YEAR / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    monthly = (1 + r).resample('ME').prod() - 1
    return {
        'cagr': float(cagr),
        'total_return': float(eq.iloc[-1] - 1),
        'sharpe': float(sharpe),
        'max_drawdown': float(dd.min()),
        'avg_month': float(monthly.mean()),
        'worst_month': float(monthly.min()),
        'positive_month_pct': float((monthly > 0).mean()),
        'month_ge_10_pct': float((monthly >= 0.10).mean()),
        'annual_turnover': float(turnover.sum() / years),
    }


def train_score(m):
    if not m or not np.isfinite(m['sharpe']):
        return -1e9
    if m['cagr'] < 0.08 or m['max_drawdown'] < -0.35:
        return -1e9
    return 1.8 * m['sharpe'] + 0.8 * m['cagr'] + 0.5 * m['max_drawdown'] + 0.3 * m['positive_month_pct']


def main():
    OUT.mkdir(exist_ok=True)
    close, low, funding = load_data()
    train_start = close.index.min() + pd.Timedelta(days=190)
    cfgs = [Config(f, lb, n, reb) for f in ['conventional_reversal','past_trough_reversal','trough_adjusted_reversal'] for lb in [30,60,90,180] for n in [1,2,3] for reb in [1,3,7]]
    rows: List[dict] = []
    cache: Dict[str, tuple[pd.Series,pd.Series,pd.Series,pd.Series]] = {}
    for cfg in cfgs:
        print('Testing', cfg.name)
        score = make_score(close, low, cfg)
        w = weights_from_score(score, cfg)
        base, turn = strategy_returns(close, funding, w, 1.0)
        stress, stress_turn = strategy_returns(close, funding, w, 1.5)
        cache[cfg.name] = (base, turn, stress, stress_turn)
        for split, st, en in [('train', train_start, TRAIN_END), ('test', TEST_START, END)]:
            m = metrics(base, turn, st, en)
            sm = metrics(stress, stress_turn, st, en)
            rows.append({'name': cfg.name, 'family': cfg.family, 'lookback': cfg.lookback, 'side_n': cfg.side_n, 'rebalance': cfg.rebalance, 'split': split, **m, **{f'stress_{k}': v for k,v in sm.items()}})
    allr = pd.DataFrame(rows)
    allr.to_csv(OUT / 'all_results.csv', index=False)
    selected = []
    for family in allr['family'].unique():
        tr = allr[(allr.family == family) & (allr.split == 'train')].copy()
        tr['score'] = tr.apply(lambda x: train_score(x.to_dict()), axis=1)
        best = tr.sort_values('score', ascending=False).iloc[0]
        if best['score'] <= -1e8:
            continue
        test = allr[(allr.name == best['name']) & (allr.split == 'test')].iloc[0].copy()
        test['passes_8pct'] = bool(
            test['cagr'] >= 0.08 and test['sharpe'] >= 1.0 and test['max_drawdown'] >= -0.25
            and test['positive_month_pct'] >= 0.55 and test['stress_cagr'] >= 0.08
            and test['stress_sharpe'] >= 0.75
        )
        selected.append(test)
    sel = pd.DataFrame(selected)
    sel.to_csv(OUT / 'selected_oos.csv', index=False)
    cols = ['name','family','cagr','sharpe','max_drawdown','avg_month','worst_month','positive_month_pct','month_ge_10_pct','annual_turnover','stress_cagr','stress_sharpe','passes_8pct']
    lines = [
        '# Wave 9 — cross-sectional crypto reversal', '',
        '- Binance USD-M perpetual daily returns; funding included.',
        '- Train selection: 2023–2024. Strict OOS: 2025–2026-07.',
        '- One-way cost 0.08%; stress case uses 1.5x execution cost.',
        '- 1x gross market-neutral exposure: 50% long / 50% short.',
        '- Any strategy with OOS CAGR below 8% is rejected.', '',
        '## Train-selected families — OOS', '',
        sel[cols].to_markdown(index=False, floatfmt='.4f') if not sel.empty else 'No train-eligible family.', '',
        f'8% pass count: **{int(sel["passes_8pct"].sum()) if not sel.empty else 0}**.', '',
        'This is an independent replication-inspired test, not a claim that the paper’s exact proprietary data construction is reproduced.',
    ]
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
