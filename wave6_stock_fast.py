from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from wave6_adjusted_stock_data import STOCKS, stock_data

OUT = Path('results_wave6_stock_fast')
TRAIN_END = pd.Timestamp('2022-12-31', tz='UTC')
TEST_START = pd.Timestamp('2023-01-01', tz='UTC')
ONE_WAY_COST = 0.00042 + 0.00030


@dataclass(frozen=True)
class Config:
    signal: str
    tp: float
    sl: float
    hold: int

    @property
    def name(self) -> str:
        return f'{self.signal}_tp{self.tp:.3f}_sl{self.sl:.3f}_h{self.hold}'


def aligned(data: Dict[str, pd.DataFrame], field: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({symbol: data[symbol][field].reindex(index) for symbol in STOCKS}, index=index)


def signal_matrices(data: Dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    close = aligned(data, 'close', index)
    sma50 = aligned(data, 'sma50', index)
    sma200 = aligned(data, 'sma200', index)
    ema20 = aligned(data, 'ema20', index)
    ema100 = aligned(data, 'ema100', index)
    rsi3 = aligned(data, 'rsi3', index)
    ret5 = aligned(data, 'ret5', index)
    ret126 = aligned(data, 'ret126', index)
    vol21 = aligned(data, 'vol21', index).clip(lower=1e-8)
    spy = data['SPY'].reindex(index)
    regime = (spy['close'] > spy['sma200']).to_numpy()[:, None]

    output: Dict[str, np.ndarray] = {}
    for lookback in (20, 55):
        level = aligned(data, f'prior_high_{lookback}', index)
        eligible = regime & (close > level).to_numpy() & (close > sma200).to_numpy() & (ret126 > 0).to_numpy()
        score = (ret126 / vol21).where(eligible).to_numpy()
        output[f'breakout{lookback}'] = score

    for threshold in (10, 20):
        eligible = regime & (close > sma200).to_numpy() & (sma50 > sma200).to_numpy() & (rsi3 <= threshold).to_numpy() & (ret126 > 0).to_numpy()
        score = (((threshold - rsi3 + 1) * ret126) / vol21).where(eligible).to_numpy()
        output[f'pullback_rsi{threshold}'] = score

    for dip in (0.01, 0.02):
        eligible = regime & (ema20 > ema100).to_numpy() & (ema100 > sma200).to_numpy() & (ret5 <= -dip).to_numpy() & (ret126 > 0).to_numpy()
        score = ((-ret5 + ret126) / vol21).where(eligible).to_numpy()
        output[f'trendpullback{int(dip*100)}'] = score
    return output


def simulate(
    config: Config,
    score: np.ndarray,
    index: pd.DatetimeIndex,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> pd.DataFrame:
    records: List[dict] = []
    active_symbol = -1
    entry = stop = target = 0.0
    entry_i = bars = 0
    n, m = score.shape

    for i in range(1, n):
        if active_symbol >= 0:
            if not np.isnan(low[i, active_symbol]) and low[i, active_symbol] <= stop:
                exit_price, reason = stop, 'stop'
            elif not np.isnan(high[i, active_symbol]) and high[i, active_symbol] >= target:
                exit_price, reason = target, 'target'
            else:
                bars += 1
                if bars < config.hold or np.isnan(close[i, active_symbol]):
                    continue
                exit_price, reason = close[i, active_symbol], 'time'
            gross = exit_price / entry - 1.0
            records.append({
                'symbol': STOCKS[active_symbol], 'entry_time': index[entry_i], 'exit_time': index[i],
                'entry_price': entry, 'exit_price': exit_price, 'bars': bars,
                'gross_return': gross, 'net_return': gross - 2 * ONE_WAY_COST, 'reason': reason,
            })
            active_symbol = -1
            continue

        previous_scores = score[i - 1]
        if np.all(np.isnan(previous_scores)):
            continue
        candidate = int(np.nanargmax(previous_scores))
        if np.isnan(open_[i, candidate]) or open_[i, candidate] <= 0:
            continue
        active_symbol = candidate
        entry_i = i
        bars = 0
        entry = float(open_[i, candidate])
        stop = entry * (1.0 - config.sl)
        target = entry * (1.0 + config.tp)

    if active_symbol >= 0 and not np.isnan(close[-1, active_symbol]):
        gross = close[-1, active_symbol] / entry - 1.0
        records.append({
            'symbol': STOCKS[active_symbol], 'entry_time': index[entry_i], 'exit_time': index[-1],
            'entry_price': entry, 'exit_price': close[-1, active_symbol], 'bars': bars,
            'gross_return': gross, 'net_return': gross - 2 * ONE_WAY_COST, 'reason': 'end',
        })
    return pd.DataFrame(records)


def bootstrap_lower(values: pd.Series, seed: int = 42) -> float:
    if len(values) < 10:
        return float('nan')
    source = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = rng.choice(source, size=(5000, len(source)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05))


def metrics(trades: pd.DataFrame, extra_cost: float = 0.0) -> dict:
    if trades.empty:
        return {'trades': 0, 'total_return': 0.0, 'avg_trade': 0.0, 'win_rate': 0.0, 'avg_win': 0.0, 'median_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0, 'max_drawdown': 0.0, 'target_hit_rate': 0.0, 'winner_ge_2_pct': 0.0, 'bootstrap_mean_5pct': float('nan'), 'positive_year_pct': 0.0}
    returns = trades['net_return'].astype(float) - extra_cost
    wins, losses = returns[returns > 0], returns[returns < 0]
    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    yearly = pd.DataFrame({'year': trades['entry_time'].dt.year, 'return': returns}).groupby('year')['return'].apply(lambda x: (1 + x).prod() - 1)
    return {
        'trades': int(len(trades)), 'total_return': float(equity.iloc[-1] - 1),
        'avg_trade': float(returns.mean()), 'win_rate': float((returns > 0).mean()),
        'avg_win': float(wins.mean()) if len(wins) else 0.0,
        'median_win': float(wins.median()) if len(wins) else 0.0,
        'avg_loss': float(losses.mean()) if len(losses) else 0.0,
        'profit_factor': float(wins.sum() / -losses.sum()) if len(losses) else float('inf'),
        'max_drawdown': float(dd.min()), 'target_hit_rate': float((trades['reason'] == 'target').mean()),
        'winner_ge_2_pct': float((wins >= 0.02).mean()) if len(wins) else 0.0,
        'bootstrap_mean_5pct': bootstrap_lower(returns),
        'positive_year_pct': float((yearly > 0).mean()),
    }


def score(row: pd.Series) -> float:
    if row['trades'] < 40 or row['total_return'] <= 0 or row['profit_factor'] < 1.05 or row['max_drawdown'] < -0.45:
        return -1e9
    return 800 * row['avg_trade'] + 1.5 * math.log(row['profit_factor']) + row['target_hit_rate'] + row['winner_ge_2_pct'] + row['max_drawdown']


def passes(row: pd.Series) -> bool:
    return bool(
        row['trades'] >= 35 and row['total_return'] > 0 and row['avg_trade'] > 0
        and row['profit_factor'] >= 1.25 and row['max_drawdown'] >= -0.25
        and row['median_win'] >= 0.02 and row['winner_ge_2_pct'] >= 0.80
        and row['bootstrap_mean_5pct'] > 0 and row['positive_year_pct'] >= 0.67
        and row['stress_profit_factor'] >= 1.05 and row['stress_avg_trade'] > 0
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = stock_data()
    index = data['SPY'].index.sort_values()
    matrices = signal_matrices(data, index)
    open_ = aligned(data, 'open', index).to_numpy(float)
    high = aligned(data, 'high', index).to_numpy(float)
    low = aligned(data, 'low', index).to_numpy(float)
    close = aligned(data, 'close', index).to_numpy(float)

    rows: List[dict] = []
    for signal_name, matrix in matrices.items():
        for tp in (0.022, 0.030):
            for sl in (0.012, 0.018):
                for hold in (5, 10, 20):
                    cfg = Config(signal_name, tp, sl, hold)
                    trades = simulate(cfg, matrix, index, open_, high, low, close)
                    if not trades.empty:
                        trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
                        trades['exit_time'] = pd.to_datetime(trades['exit_time'], utc=True)
                    train = trades[trades['entry_time'] <= TRAIN_END] if not trades.empty else trades
                    test = trades[trades['entry_time'] >= TEST_START] if not trades.empty else trades
                    for split, subset in [('train', train), ('test', test)]:
                        base = metrics(subset)
                        stress = metrics(subset, ONE_WAY_COST)
                        rows.append({'name': cfg.name, 'signal': signal_name, 'tp': tp, 'sl': sl, 'hold': hold, 'split': split, **base, **{'stress_' + key: value for key, value in stress.items()}})

    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'all_results.csv', index=False)
    selected_rows = []
    for signal_name in matrices:
        train = results[(results['signal'] == signal_name) & (results['split'] == 'train')].copy()
        train['selection_score'] = train.apply(score, axis=1)
        best = train.sort_values('selection_score', ascending=False).iloc[0]
        if best['selection_score'] <= -1e8:
            continue
        test_row = results[(results['name'] == best['name']) & (results['split'] == 'test')].iloc[0].copy()
        test_row['passes_strict'] = passes(test_row)
        selected_rows.append(test_row)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUT / 'selected_oos.csv', index=False)
    passing = selected[selected['passes_strict'] == True] if not selected.empty else selected

    columns = ['name', 'signal', 'trades', 'total_return', 'avg_trade', 'win_rate', 'avg_win', 'median_win', 'avg_loss', 'profit_factor', 'max_drawdown', 'target_hit_rate', 'winner_ge_2_pct', 'positive_year_pct', 'bootstrap_mean_5pct', 'stress_profit_factor', 'stress_avg_trade', 'passes_strict']
    report = [
        '# Fast stock trade-level 2% target audit', '',
        '- Adjusted OHLC; train 2014–2022; strict OOS 2023–2026-08.',
        f'- One-way execution cost {ONE_WAY_COST:.3%}; stress test adds the same cost again.',
        '- Entry next open; same-day stop assumed before target; one position at a time.',
        '- A pass means historical positive expectancy with net winner target, never every future trade winning.', '',
        selected[columns].sort_values(['passes_strict', 'profit_factor'], ascending=[False, False]).to_markdown(index=False, floatfmt='.4f') if not selected.empty else 'No selectable strategy.', '',
        f'Strict pass count: **{len(passing)}**.',
    ]
    (OUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    (OUT / 'decision.json').write_text(json.dumps({'strict_pass_count': int(len(passing)), 'passing': passing.to_dict(orient='records')}, indent=2, default=str), encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
