from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import wave3_funding_carry as w3

OUT = Path('results_wave10')
SYMBOL = 'BTCUSDT'
START_MONTH = '2020-01'
END_MONTH = '2026-07'
ONE_WAY_COST = 0.0008
ROUNDTRIP_REFERENCE = 2 * ONE_WAY_COST
HOURS_YEAR = 24 * 365.25
TRAIN_END = pd.Timestamp('2023-12-31 23:00:00', tz='UTC')
VALID_START = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
VALID_END = pd.Timestamp('2024-12-31 23:00:00', tz='UTC')
OOS_START = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
OOS_END = pd.Timestamp('2026-07-31 23:00:00', tz='UTC')
HORIZON = 6


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_data() -> pd.DataFrame:
    w3.START_MONTH = START_MONTH
    w3.END_MONTH = END_MONTH
    w3.CACHE = Path('.cache_wave10')
    k = w3.load_kline(SYMBOL, 'perp')
    f = w3.load_funding(SYMBOL)
    df = k[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].copy()
    df['funding'] = f.reindex(df.index).fillna(0.0)
    return df.loc[:OOS_END].copy()


def feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    c = df['close']
    ret1 = c.pct_change()
    x = pd.DataFrame(index=df.index)
    for h in [1, 3, 6, 12, 24, 72, 168]:
        x[f'ret_{h}'] = c.pct_change(h)
    for h in [6, 24, 72, 168]:
        x[f'vol_{h}'] = ret1.rolling(h).std()
    for span in [24, 72, 168, 336]:
        ema = c.ewm(span=span, adjust=False).mean()
        x[f'ema_gap_{span}'] = c / ema - 1
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - c.shift()).abs(),
        (df['low'] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    x['atr_24'] = tr.rolling(24).mean() / c
    x['rsi_7'] = rsi(c, 7) / 100.0
    x['rsi_14'] = rsi(c, 14) / 100.0
    qv = df['quote_volume'].replace(0, np.nan)
    x['volume_ratio_24'] = qv / qv.rolling(24).median()
    x['volume_z_168'] = (qv - qv.rolling(168).mean()) / qv.rolling(168).std().replace(0, np.nan)
    x['range_pos_24'] = (c - df['low'].rolling(24).min()) / (df['high'].rolling(24).max() - df['low'].rolling(24).min()).replace(0, np.nan)
    x['funding_24'] = df['funding'].rolling(24).sum()
    x['funding_168'] = df['funding'].rolling(168).sum()
    hour = df.index.hour.to_numpy()
    dow = df.index.dayofweek.to_numpy()
    x['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    x['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    x['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    x['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    # Predict next six-hour close-to-close return. Features use information available at t close only.
    y = c.shift(-HORIZON) / c - 1
    return x.replace([np.inf, -np.inf], np.nan), y


def model_grid() -> List[Dict[str, float | int]]:
    return [
        {'max_depth': 2, 'learning_rate': 0.03, 'n_estimators': 300},
        {'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 300},
        {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 300},
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300},
        {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 250},
    ]


def fit_predict(train_x, train_y, test_x, params) -> np.ndarray:
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=20260807,
        n_jobs=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=2.0,
        min_child_weight=20,
        **params,
    )
    model.fit(train_x, train_y, verbose=False)
    return model.predict(test_x)


def positions_from_prediction(pred: pd.Series, threshold: float, mode: str) -> pd.Series:
    if mode == 'long_only':
        pos = (pred > threshold).astype(float)
    elif mode == 'long_short':
        pos = pd.Series(np.where(pred > threshold, 1.0, np.where(pred < -threshold, -1.0, 0.0)), index=pred.index)
    else:
        raise ValueError(mode)
    # Prediction made after t close; position becomes active on next hourly return.
    return pos.shift(1).fillna(0.0)


def pnl(df: pd.DataFrame, position: pd.Series, cost_mult: float = 1.0) -> pd.Series:
    ret = df['close'].pct_change().reindex(position.index).fillna(0.0)
    fund = df['funding'].reindex(position.index).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    return position * ret - turnover * ONE_WAY_COST * cost_mult - position * fund


def metrics(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if len(r) < 24 * 90:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / HOURS_YEAR
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * math.sqrt(HOURS_YEAR)
    sharpe = r.mean() * HOURS_YEAR / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    monthly = (1 + r).resample('ME').prod() - 1
    yearly = (1 + r).resample('YE').prod() - 1
    return {
        'cagr': float(cagr),
        'total_return': float(eq.iloc[-1] - 1),
        'sharpe': float(sharpe),
        'max_drawdown': float(dd.min()),
        'avg_month': float(monthly.mean()),
        'worst_month': float(monthly.min()),
        'positive_month_pct': float((monthly > 0).mean()),
        'month_ge_10_pct': float((monthly >= 0.10).mean()),
        'positive_year_pct': float((yearly > 0).mean()),
    }


def validate_config(df, x, y, params, threshold_mult, mode):
    mask_train = (x.index <= TRAIN_END) & y.notna() & x.notna().all(axis=1)
    mask_valid = (x.index >= VALID_START) & (x.index <= VALID_END) & x.notna().all(axis=1)
    pred = fit_predict(x.loc[mask_train], y.loc[mask_train], x.loc[mask_valid], params)
    ps = pd.Series(pred, index=x.loc[mask_valid].index)
    threshold = ROUNDTRIP_REFERENCE * threshold_mult
    pos = positions_from_prediction(ps, threshold, mode)
    base = pnl(df, pos, 1.0)
    stress = pnl(df, pos, 1.5)
    return metrics(base), metrics(stress)


def select_config(df, x, y):
    rows = []
    for params in model_grid():
        for tm in [0.75, 1.0, 1.5, 2.0, 3.0]:
            for mode in ['long_only', 'long_short']:
                print('Validate', params, 'threshold', tm, mode)
                m, sm = validate_config(df, x, y, params, tm, mode)
                if not m:
                    continue
                score = -1e9
                if m['cagr'] >= 0.08 and m['max_drawdown'] >= -0.35 and sm.get('cagr', -1) > 0:
                    score = 2.0 * m['sharpe'] + 0.7 * m['cagr'] + 0.5 * m['max_drawdown'] + 0.3 * m['positive_month_pct']
                rows.append({'params': params, 'threshold_mult': tm, 'mode': mode, 'score': score, **m, **{f'stress_{k}': v for k, v in sm.items()}})
    table = pd.DataFrame(rows).sort_values('score', ascending=False)
    table.to_csv(OUT / 'validation_grid.csv', index=False)
    if table.empty or table.iloc[0]['score'] <= -1e8:
        return None, table
    return table.iloc[0].to_dict(), table


def oos_walk_forward(df, x, y, selected):
    params = selected['params']
    if isinstance(params, str):
        import ast
        params = ast.literal_eval(params)
    tm = float(selected['threshold_mult'])
    mode = str(selected['mode'])
    preds = []
    months = pd.period_range('2025-01', '2026-07', freq='M')
    valid_features = x.notna().all(axis=1)
    valid_target = y.notna()
    for p in months:
        start = pd.Timestamp(p.start_time, tz='UTC')
        end = pd.Timestamp(p.end_time, tz='UTC').floor('h')
        test_mask = (x.index >= start) & (x.index <= end) & valid_features
        train_cutoff = start - pd.Timedelta(hours=HORIZON + 1)
        train_mask = (x.index < train_cutoff) & valid_features & valid_target
        # Expanding window, but limit to latest ~4 years to allow regime adaptation.
        train_start = max(x.index.min(), start - pd.Timedelta(days=1460))
        train_mask &= x.index >= train_start
        if test_mask.sum() == 0:
            continue
        print('OOS month', p, 'train', int(train_mask.sum()), 'test', int(test_mask.sum()))
        pr = fit_predict(x.loc[train_mask], y.loc[train_mask], x.loc[test_mask], params)
        preds.append(pd.Series(pr, index=x.loc[test_mask].index))
    pred = pd.concat(preds).sort_index()
    pos = positions_from_prediction(pred, ROUNDTRIP_REFERENCE * tm, mode)
    base = pnl(df, pos, 1.0)
    stress = pnl(df, pos, 1.5)
    return pred, pos, base, stress


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    x, y = feature_frame(df)
    selected, grid = select_config(df, x, y)
    if selected is None:
        report = '# Wave 10 — cost-aware XGBoost BTC perpetual\n\nNo validation configuration reached the 8% research floor. OOS was not opened for model selection.\n'
        (OUT / 'report.md').write_text(report, encoding='utf-8')
        print(report)
        return
    pred, pos, base, stress = oos_walk_forward(df, x, y, selected)
    m = metrics(base.loc[OOS_START:OOS_END])
    sm = metrics(stress.loc[OOS_START:OOS_END])
    turnover = float(pos.diff().abs().sum() / ((pos.index[-1] - pos.index[0]).total_seconds() / (365.25 * 86400)))
    active = float((pos != 0).mean())
    passes = bool(
        m['cagr'] >= 0.08 and m['sharpe'] >= 1.0 and m['max_drawdown'] >= -0.30
        and m['positive_month_pct'] >= 0.55 and sm['cagr'] >= 0.08 and sm['sharpe'] >= 0.75
    )
    pd.DataFrame({'prediction': pred, 'position': pos, 'return': base, 'stress_return': stress}).to_csv(OUT / 'oos_series.csv')
    summary = pd.DataFrame([{'mode': selected['mode'], 'threshold_mult': selected['threshold_mult'], 'params': selected['params'], 'annual_turnover': turnover, 'active_pct': active, **m, **{f'stress_{k}': v for k, v in sm.items()}, 'passes_8pct': passes}])
    summary.to_csv(OUT / 'oos_summary.csv', index=False)
    cols = ['mode','threshold_mult','cagr','sharpe','max_drawdown','avg_month','worst_month','positive_month_pct','month_ge_10_pct','annual_turnover','active_pct','stress_cagr','stress_sharpe','stress_max_drawdown','passes_8pct']
    lines = [
        '# Wave 10 — cost-aware XGBoost BTC perpetual', '',
        '- Binance BTCUSDT USD-M perpetual 1h data.',
        '- Features are lag-safe technical/volume/funding variables; target is next 6h return.',
        '- Hyperparameters and execution threshold selected on 2024 validation only after fitting on data through 2023.',
        '- Strict OOS: monthly expanding/rolling retraining during 2025–2026-07; no OOS return used for configuration selection.',
        '- Base execution cost: 0.08% per unit of one-way turnover; historical funding included.',
        '- Stress: 1.5x execution cost.',
        '- Research floor: OOS CAGR >=8%; Sharpe>=1; DD<=30%; positive months>=55%; stress CAGR>=8%.', '',
        '## Selected model — strict OOS', '',
        summary[cols].to_markdown(index=False, floatfmt='.4f'), '',
        f'PASS: **{passes}**.', '',
        'This is an independent perpetual-futures replication inspired by Bysik & Ślepaczuk (2026), not an exact reproduction of their dataset or model pipeline.',
    ]
    report = '\n'.join(lines)
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
