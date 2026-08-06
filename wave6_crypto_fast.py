from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path('results_wave6_crypto_fast')
TRAIN_END = pd.Timestamp('2024-12-31 23:59:59', tz='UTC')
TEST_START = pd.Timestamp('2025-01-01', tz='UTC')
ONE_WAY_COST = 0.00060 + 0.00020
SYMBOLS = w3.SYMBOLS


@dataclass(frozen=True)
class Config:
    signal: str
    side: int
    tp: float
    sl: float
    hold: int

    @property
    def name(self) -> str:
        direction = 'L' if self.side == 1 else 'S'
        return f'{self.signal}_{direction}_tp{self.tp:.3f}_sl{self.sl:.3f}_h{self.hold}'


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_data() -> Dict[str, pd.DataFrame]:
    w3.START_MONTH = '2023-01'
    w3.END_MONTH = '2026-07'
    w3.CACHE = Path('.cache_wave3')
    output: Dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        print('Loading', symbol)
        bars = w3.load_kline(symbol, 'perp')
        funding = w3.load_funding(symbol)
        bars = bars.resample('4h', origin='start_day').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'quote_volume': 'sum'}
        ).dropna()
        bars['funding_event'] = funding.resample('4h', origin='start_day').sum().reindex(bars.index).fillna(0.0)
        bars['ema50'] = bars['close'].ewm(span=50, adjust=False).mean()
        bars['ema200'] = bars['close'].ewm(span=200, adjust=False).mean()
        bars['rsi7'] = rsi(bars['close'], 7)
        bars['ret42'] = bars['close'].pct_change(42)
        bars['vol42'] = bars['close'].pct_change().rolling(42).std().clip(lower=1e-8)
        bars['volume_ratio'] = bars['quote_volume'] / bars['quote_volume'].rolling(42).median()
        tr = pd.concat([
            bars['high'] - bars['low'],
            (bars['high'] - bars['close'].shift()).abs(),
            (bars['low'] - bars['close'].shift()).abs(),
        ], axis=1).max(axis=1)
        bars['atr_pct'] = tr.rolling(21).mean() / bars['close']
        bars['atr_rank126'] = bars['atr_pct'].rolling(126).rank(pct=True)
        for lookback in (6, 18):
            bars[f'prior_high_{lookback}'] = bars['high'].rolling(lookback).max().shift(1)
            bars[f'prior_low_{lookback}'] = bars['low'].rolling(lookback).min().shift(1)
        output[symbol] = bars
    start = max(frame.index.min() for frame in output.values())
    end = min(frame.index.max() for frame in output.values())
    return {symbol: frame.loc[start:end].copy() for symbol, frame in output.items()}


def aligned(data: Dict[str, pd.DataFrame], field: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({symbol: data[symbol][field].reindex(index) for symbol in SYMBOLS}, index=index)


def signals(data: Dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> Dict[tuple[str, int], np.ndarray]:
    close = aligned(data, 'close', index)
    ema200 = aligned(data, 'ema200', index)
    rsi7 = aligned(data, 'rsi7', index)
    ret42 = aligned(data, 'ret42', index)
    vol42 = aligned(data, 'vol42', index)
    volume_ratio = aligned(data, 'volume_ratio', index)
    atr_pct = aligned(data, 'atr_pct', index).clip(lower=1e-8)
    atr_rank = aligned(data, 'atr_rank126', index)
    btc = data['BTCUSDT'].reindex(index)
    bull = (btc['close'] > btc['ema200']).to_numpy()[:, None]
    bear = (btc['close'] < btc['ema200']).to_numpy()[:, None]
    output: Dict[tuple[str, int], np.ndarray] = {}

    for side in (1, -1):
        regime = bull if side == 1 else bear
        trend = (close > ema200).to_numpy() if side == 1 else (close < ema200).to_numpy()
        directional_return = ret42 if side == 1 else -ret42
        for lookback in (6, 18):
            level = aligned(data, f'prior_high_{lookback}' if side == 1 else f'prior_low_{lookback}', index)
            direction = (close > level).to_numpy() if side == 1 else (close < level).to_numpy()
            for volume in (1.0, 1.5):
                eligible = regime & trend & direction & (volume_ratio >= volume).to_numpy()
                output[(f'breakout{lookback}_v{volume:g}', side)] = (directional_return / vol42).where(eligible).to_numpy()

        for threshold in (25, 35):
            if side == 1:
                eligible = regime & trend & (rsi7 <= threshold).to_numpy() & (ret42 > 0).to_numpy()
                score = ((threshold - rsi7 + 1) * ret42 / vol42).where(eligible)
            else:
                eligible = regime & trend & (rsi7 >= 100 - threshold).to_numpy() & (ret42 < 0).to_numpy()
                score = ((rsi7 - 100 + threshold + 1) * -ret42 / vol42).where(eligible)
            output[(f'pullback_rsi{threshold}', side)] = score.to_numpy()

        for lookback in (6, 18):
            level = aligned(data, f'prior_high_{lookback}' if side == 1 else f'prior_low_{lookback}', index)
            direction = (close > level).to_numpy() if side == 1 else (close < level).to_numpy()
            for rank in (0.15, 0.25):
                eligible = regime & direction & (atr_rank <= rank).to_numpy()
                output[(f'squeeze{lookback}_r{rank:g}', side)] = (((rank - atr_rank + 0.01) / atr_pct).where(eligible)).to_numpy()
    return output


def simulate(
    cfg: Config,
    score: np.ndarray,
    index: pd.DatetimeIndex,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    funding: np.ndarray,
) -> pd.DataFrame:
    records: List[dict] = []
    active = -1
    entry = stop = target = funding_cost = 0.0
    entry_i = bars = 0
    n, _ = score.shape
    for i in range(1, n):
        if active >= 0:
            funding_cost += cfg.side * float(funding[i, active])
            stop_hit = low[i, active] <= stop if cfg.side == 1 else high[i, active] >= stop
            target_hit = high[i, active] >= target if cfg.side == 1 else low[i, active] <= target
            if stop_hit:
                exit_price, reason = stop, 'stop'
            elif target_hit:
                exit_price, reason = target, 'target'
            else:
                bars += 1
                if bars < cfg.hold or np.isnan(close[i, active]):
                    continue
                exit_price, reason = close[i, active], 'time'
            gross = cfg.side * (exit_price / entry - 1.0)
            records.append({
                'symbol': SYMBOLS[active], 'side': 'long' if cfg.side == 1 else 'short',
                'entry_time': index[entry_i], 'exit_time': index[i], 'entry_price': entry,
                'exit_price': exit_price, 'bars': bars, 'gross_return': gross,
                'funding_cost': funding_cost, 'net_return': gross - 2 * ONE_WAY_COST - funding_cost,
                'reason': reason,
            })
            active = -1
            continue

        previous = score[i - 1]
        if np.all(np.isnan(previous)):
            continue
        candidate = int(np.nanargmax(previous))
        if np.isnan(open_[i, candidate]) or open_[i, candidate] <= 0:
            continue
        active = candidate
        entry_i, bars, funding_cost = i, 0, 0.0
        entry = float(open_[i, candidate])
        stop = entry * (1 - cfg.sl) if cfg.side == 1 else entry * (1 + cfg.sl)
        target = entry * (1 + cfg.tp) if cfg.side == 1 else entry * (1 - cfg.tp)
    return pd.DataFrame(records)


def bootstrap_lower(values: pd.Series, seed: int = 42) -> float:
    if len(values) < 10:
        return float('nan')
    source = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = rng.choice(source, size=(3000, len(source)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05))


def metrics(trades: pd.DataFrame, extra_cost: float = 0.0) -> dict:
    if trades.empty:
        return {'trades': 0, 'total_return': 0.0, 'avg_trade': 0.0, 'win_rate': 0.0, 'avg_win': 0.0, 'median_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0, 'max_drawdown': 0.0, 'target_hit_rate': 0.0, 'winner_ge_2_pct': 0.0, 'bootstrap_mean_5pct': float('nan'), 'positive_year_pct': 0.0}
    returns = trades['net_return'].astype(float) - extra_cost
    wins, losses = returns[returns > 0], returns[returns < 0]
    equity = (1 + returns).cumprod()
    yearly = pd.DataFrame({'year': trades['entry_time'].dt.year, 'return': returns}).groupby('year')['return'].apply(lambda x: (1 + x).prod() - 1)
    return {
        'trades': int(len(trades)), 'total_return': float(equity.iloc[-1] - 1), 'avg_trade': float(returns.mean()),
        'win_rate': float((returns > 0).mean()), 'avg_win': float(wins.mean()) if len(wins) else 0.0,
        'median_win': float(wins.median()) if len(wins) else 0.0, 'avg_loss': float(losses.mean()) if len(losses) else 0.0,
        'profit_factor': float(wins.sum() / -losses.sum()) if len(losses) else float('inf'),
        'max_drawdown': float((equity / equity.cummax() - 1).min()),
        'target_hit_rate': float((trades['reason'] == 'target').mean()),
        'winner_ge_2_pct': float((wins >= 0.02).mean()) if len(wins) else 0.0,
        'bootstrap_mean_5pct': bootstrap_lower(returns), 'positive_year_pct': float((yearly > 0).mean()),
    }


def train_score(row: pd.Series) -> float:
    if row['trades'] < 50 or row['total_return'] <= 0 or row['profit_factor'] < 1.05 or row['max_drawdown'] < -0.50:
        return -1e9
    return 800 * row['avg_trade'] + 1.5 * math.log(row['profit_factor']) + row['target_hit_rate'] + row['winner_ge_2_pct'] + row['max_drawdown']


def passes(row: pd.Series) -> bool:
    return bool(
        row['trades'] >= 60 and row['total_return'] > 0 and row['avg_trade'] > 0
        and row['profit_factor'] >= 1.25 and row['max_drawdown'] >= -0.25
        and row['median_win'] >= 0.02 and row['winner_ge_2_pct'] >= 0.80
        and row['bootstrap_mean_5pct'] > 0 and row['positive_year_pct'] >= 0.50
        and row['stress_profit_factor'] >= 1.05 and row['stress_avg_trade'] > 0
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_data()
    index = data['BTCUSDT'].index.sort_values()
    signal_map = signals(data, index)
    open_ = aligned(data, 'open', index).to_numpy(float)
    high = aligned(data, 'high', index).to_numpy(float)
    low = aligned(data, 'low', index).to_numpy(float)
    close = aligned(data, 'close', index).to_numpy(float)
    funding = aligned(data, 'funding_event', index).fillna(0.0).to_numpy(float)

    rows: List[dict] = []
    for (signal_name, side), matrix in signal_map.items():
        for tp in (0.022, 0.030):
            for sl in (0.012, 0.018):
                for hold in (6, 12, 24):
                    cfg = Config(signal_name, side, tp, sl, hold)
                    trades = simulate(cfg, matrix, index, open_, high, low, close, funding)
                    if not trades.empty:
                        trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
                        trades['exit_time'] = pd.to_datetime(trades['exit_time'], utc=True)
                    train = trades[trades['entry_time'] <= TRAIN_END] if not trades.empty else trades
                    test = trades[trades['entry_time'] >= TEST_START] if not trades.empty else trades
                    for split, subset in [('train', train), ('test', test)]:
                        base, stress = metrics(subset), metrics(subset, ONE_WAY_COST)
                        rows.append({'name': cfg.name, 'signal': signal_name, 'side': side, 'tp': tp, 'sl': sl, 'hold': hold, 'split': split, **base, **{'stress_' + key: value for key, value in stress.items()}})

    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'all_results.csv', index=False)
    selected_rows = []
    families = results[['signal', 'side']].drop_duplicates().itertuples(index=False)
    for signal_name, side in families:
        train = results[(results['signal'] == signal_name) & (results['side'] == side) & (results['split'] == 'train')].copy()
        train['selection_score'] = train.apply(train_score, axis=1)
        best = train.sort_values('selection_score', ascending=False).iloc[0]
        if best['selection_score'] <= -1e8:
            continue
        row = results[(results['name'] == best['name']) & (results['split'] == 'test')].iloc[0].copy()
        row['passes_strict'] = passes(row)
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUT / 'selected_oos.csv', index=False)
    passing = selected[selected['passes_strict'] == True] if not selected.empty else selected
    columns = ['name', 'signal', 'side', 'trades', 'total_return', 'avg_trade', 'win_rate', 'avg_win', 'median_win', 'avg_loss', 'profit_factor', 'max_drawdown', 'target_hit_rate', 'winner_ge_2_pct', 'positive_year_pct', 'bootstrap_mean_5pct', 'stress_profit_factor', 'stress_avg_trade', 'passes_strict']
    report = [
        '# Fast crypto trade-level 2% target audit', '',
        '- Binance USD-M perpetual 4h; train 2023–2024; strict OOS 2025–2026-07.',
        '- Binance historical funding included.',
        f'- One-way execution cost {ONE_WAY_COST:.3%}; stress adds the same amount again.',
        '- Entry next 4h open; same-bar stop before target; one position at a time.', '',
        selected[columns].sort_values(['passes_strict', 'profit_factor'], ascending=[False, False]).to_markdown(index=False, floatfmt='.4f') if not selected.empty else 'No selectable strategy.', '',
        f'Strict pass count: **{len(passing)}**.',
    ]
    (OUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    (OUT / 'decision.json').write_text(json.dumps({'strict_pass_count': int(len(passing)), 'passing': passing.to_dict(orient='records')}, indent=2, default=str), encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
