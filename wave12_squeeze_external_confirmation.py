from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path('results_wave12')
NEW_SYMBOLS = [
    'BCHUSDT','DOTUSDT','NEARUSDT','AAVEUSDT','ETCUSDT',
    'FILUSDT','ATOMUSDT','UNIUSDT','XLMUSDT','TRXUSDT',
]
START_MONTH = '2023-01'
END_MONTH = '2026-07'
TEST_START = pd.Timestamp('2025-01-01', tz='UTC')
TEST_END = pd.Timestamp('2026-07-31 23:59:59', tz='UTC')
ONE_WAY_COST = 0.0008
# Parameters are frozen from prior Wave 6 candidate. No re-optimization here.
LOOKBACK = 18
COMPRESSION_RANK = 0.25
TP = 0.030
SL = 0.018
MAX_HOLD_BARS = 6
BARS_YEAR = 6 * 365.25  # 4h bars/year


def prepare_symbol(symbol: str) -> tuple[str, pd.DataFrame] | None:
    try:
        k = w3.load_kline(symbol, 'perp')
        f = w3.load_funding(symbol)
    except Exception as exc:
        print('EXCLUDE', symbol, exc)
        return None
    bars = k.resample('4h', origin='start_day').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum'
    }).dropna()
    bars['funding'] = f.resample('4h', origin='start_day').sum().reindex(bars.index).fillna(0.0)
    prev = bars['close'].shift(1)
    tr = pd.concat([
        bars['high'] - bars['low'],
        (bars['high'] - prev).abs(),
        (bars['low'] - prev).abs(),
    ], axis=1).max(axis=1)
    bars['atr_pct'] = tr.rolling(21).mean() / bars['close']
    bars['atr_rank126'] = bars['atr_pct'].rolling(126).rank(pct=True)
    bars['prior_low'] = bars['low'].rolling(LOOKBACK).min().shift(1)
    return symbol, bars


def load_data() -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    w3.START_MONTH = START_MONTH
    w3.END_MONTH = END_MONTH
    w3.CACHE = Path('.cache_wave12')
    # BTC only supplies the market-regime filter.
    btc_k = w3.load_kline('BTCUSDT', 'perp')
    btc = btc_k.resample('4h', origin='start_day').agg({'close':'last'}).dropna()
    btc['ema200'] = btc['close'].ewm(span=200, adjust=False).mean()

    data: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(prepare_symbol, s): s for s in NEW_SYMBOLS}
        for fut in as_completed(futs):
            result = fut.result()
            if result is not None:
                s, bars = result
                data[s] = bars
                print('Loaded', s, len(bars))
    if len(data) < 5:
        raise RuntimeError(f'Only {len(data)} new symbols available; confirmation universe too small')
    return data, btc


def build_timeline(data: Dict[str, pd.DataFrame], btc: pd.DataFrame):
    start = max([x.index.min() for x in data.values()] + [btc.index.min()])
    end = min([x.index.max() for x in data.values()] + [btc.index.max()])
    idx = btc.loc[start:end].index
    return idx


def aligned(data, field, idx):
    return pd.DataFrame({s: data[s][field].reindex(idx) for s in data}, index=idx)


def simulate(data: Dict[str, pd.DataFrame], btc: pd.DataFrame) -> pd.DataFrame:
    symbols = list(data)
    idx = build_timeline(data, btc)
    open_ = aligned(data, 'open', idx).to_numpy(float)
    high = aligned(data, 'high', idx).to_numpy(float)
    low = aligned(data, 'low', idx).to_numpy(float)
    close = aligned(data, 'close', idx)
    funding = aligned(data, 'funding', idx).fillna(0.0).to_numpy(float)
    atr_pct = aligned(data, 'atr_pct', idx).clip(lower=1e-8)
    rank = aligned(data, 'atr_rank126', idx)
    prior_low = aligned(data, 'prior_low', idx)
    btc2 = btc.reindex(idx)
    bear = (btc2['close'] < btc2['ema200'])

    eligible = close.lt(prior_low).mul(bear, axis=0) & rank.le(COMPRESSION_RANK)
    score = ((COMPRESSION_RANK - rank + 0.01) / atr_pct).where(eligible).to_numpy(float)

    records: List[dict] = []
    active = -1
    entry = stop = target = funding_cost = 0.0
    entry_i = bars_held = 0
    for i in range(1, len(idx)):
        if active >= 0:
            # Short receives positive funding, pays negative funding.
            funding_cost += -float(funding[i, active])
            stop_hit = high[i, active] >= stop
            target_hit = low[i, active] <= target
            if stop_hit:
                exit_price, reason = stop, 'stop'
            elif target_hit:
                exit_price, reason = target, 'target'
            else:
                bars_held += 1
                if bars_held < MAX_HOLD_BARS or np.isnan(close.iloc[i, active]):
                    continue
                exit_price, reason = float(close.iloc[i, active]), 'time'
            gross = -(exit_price / entry - 1.0)
            net = gross - 2 * ONE_WAY_COST - funding_cost
            records.append({
                'symbol': symbols[active], 'entry_time': idx[entry_i], 'exit_time': idx[i],
                'entry_price': entry, 'exit_price': exit_price, 'gross_return': gross,
                'funding_cost': funding_cost, 'net_return': net, 'reason': reason,
            })
            active = -1
            continue

        prev = score[i - 1]
        if np.all(np.isnan(prev)):
            continue
        j = int(np.nanargmax(prev))
        if np.isnan(open_[i, j]) or open_[i, j] <= 0:
            continue
        active = j
        entry_i = i
        bars_held = 0
        funding_cost = 0.0
        entry = float(open_[i, j])
        stop = entry * (1 + SL)
        target = entry * (1 - TP)
    return pd.DataFrame(records)


def bootstrap5(values: np.ndarray, seed: int = 20260807) -> float:
    if len(values) < 20:
        return float('nan')
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05))


def metrics(trades: pd.DataFrame, extra_roundtrip_cost: float = 0.0) -> dict:
    t = trades[trades['entry_time'] >= TEST_START].copy()
    if t.empty:
        return {}
    r = t['net_return'].to_numpy(float) - extra_roundtrip_cost
    eq = np.cumprod(1 + r)
    years = max((TEST_END - TEST_START).total_seconds() / (365.25 * 86400), 0.01)
    cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1.0
    peak = np.maximum.accumulate(eq)
    dd = np.min(eq / peak - 1)
    wins = r[r > 0]
    losses = r[r < 0]
    pf = wins.sum() / -losses.sum() if len(losses) else float('inf')
    years_s = pd.Series(t['entry_time'].dt.year.to_numpy(), index=np.arange(len(t)))
    yearly = pd.DataFrame({'y': years_s, 'r': r}).groupby('y')['r'].apply(lambda z: np.prod(1 + z) - 1)
    return {
        'trades': int(len(r)),
        'cagr': float(cagr),
        'total_return': float(eq[-1] - 1),
        'avg_trade': float(np.mean(r)),
        'win_rate': float(np.mean(r > 0)),
        'profit_factor': float(pf),
        'max_drawdown': float(dd),
        'median_win': float(np.median(wins)) if len(wins) else 0.0,
        'target_hit_rate': float(np.mean(t['reason'].to_numpy() == 'target')),
        'bootstrap_mean_5pct': bootstrap5(r),
        'positive_year_pct': float(np.mean(yearly.to_numpy() > 0)),
    }


def main():
    OUT.mkdir(exist_ok=True)
    data, btc = load_data()
    trades = simulate(data, btc)
    if not trades.empty:
        trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
        trades['exit_time'] = pd.to_datetime(trades['exit_time'], utc=True)
    trades.to_csv(OUT / 'trades.csv', index=False)
    base = metrics(trades, 0.0)
    # Extra 0.08% on entry and exit = 1.5x original total execution cost.
    stress = metrics(trades, 2 * ONE_WAY_COST * 0.5)
    row = {'universe': ','.join(data.keys()), **base, **{f'stress_{k}': v for k, v in stress.items()}}
    passed = bool(
        base and base['trades'] >= 50 and base['cagr'] >= 0.08 and base['profit_factor'] >= 1.25
        and base['max_drawdown'] >= -0.25 and base['bootstrap_mean_5pct'] > 0
        and base['positive_year_pct'] >= 1.0 and stress['cagr'] >= 0.08 and stress['profit_factor'] >= 1.10
    )
    row['passes_external_confirmation'] = passed
    summary = pd.DataFrame([row])
    summary.to_csv(OUT / 'summary.csv', index=False)
    cols = ['trades','cagr','total_return','avg_trade','win_rate','profit_factor','max_drawdown','median_win','target_hit_rate','bootstrap_mean_5pct','positive_year_pct','stress_cagr','stress_profit_factor','stress_max_drawdown','passes_external_confirmation']
    lines = [
        '# Wave 12 — frozen squeeze-short external-universe confirmation', '',
        '- Rule was frozen before this test: 4h downside break of prior 18 bars after ATR-rank compression <=25%, BTC below EMA200.',
        '- Short only; TP 3.0%, SL 1.8%, maximum hold 24h; next-bar-open entry; stop-first on ambiguous bars.',
        '- New universe was not used in Wave 6 parameter selection: ' + ', '.join(data.keys()),
        '- Strict evaluation window: 2025–2026-07; Binance USD-M funding included.',
        '- Base one-way execution cost 0.08%; stress uses 1.5x total execution cost.',
        '- Confirmation requires CAGR>=8%, PF>=1.25, DD<=25%, bootstrap 5% mean>0, both calendar years positive, and stress CAGR>=8%.', '',
        summary[cols].to_markdown(index=False, floatfmt='.4f'), '',
        f'EXTERNAL CONFIRMATION PASS: **{passed}**.',
    ]
    report = '\n'.join(lines)
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
