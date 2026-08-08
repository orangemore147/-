from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import continuous_research_v2 as v2
import wave3_funding_carry as w3

OUT = Path('results_robust_audit')
STATE = OUT / 'state.json'
STATUS = OUT / 'status.json'
REPORT = OUT / 'latest_report.md'
ALERT = OUT / 'ALERT.json'
AUDIT = OUT / 'audit_results.csv'

DEV_START = pd.Timestamp('2023-01-01', tz='UTC')
DEV_END = v2.DEV_VAL_END
HOLD_START = v2.HOLDOUT_START
HOLD_END = v2.HOLDOUT_END
PRIMARY = ['BTCUSDT', 'SOLUSDT']
EXTERNAL = ['ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT']
ONE_WAY_COST = v2.ONE_WAY_COST


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'audited_keys': [], 'strict_passes': []}


def cfg_from_row(r: pd.Series, symbol: str | None = None) -> v2.Config:
    return v2.Config(
        family=str(r['family']),
        symbol=str(symbol if symbol is not None else r['symbol']),
        side=str(r['side']),
        lookback=int(r['lookback']),
        hold=int(r['hold']),
        threshold=float(r['threshold']),
        aux=float(r['aux']),
        stop=float(r['stop']),
        target=float(r['target']),
    )


def strict_simulate(b: pd.DataFrame, funding_events: pd.Series, c: v2.Config) -> pd.DataFrame:
    sg = v2.signal(b, c).to_numpy(int)
    idx = b.index
    op = b['open'].to_numpy(float)
    hi = b['high'].to_numpy(float)
    lo = b['low'].to_numpy(float)
    rec: List[dict] = []
    i = 0
    while i < len(b) - c.hold - 2:
        if sg[i] == 0:
            i += 1
            continue
        side = int(sg[i])
        ei = i + 1
        entry = float(op[ei])
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        planned = min(ei + c.hold, len(b) - 1)
        xi = planned
        xp = float(op[planned])
        reason = 'time'
        sp = entry * (1 - c.stop if side > 0 else 1 + c.stop) if c.stop > 0 else None
        tp = entry * (1 + c.target if side > 0 else 1 - c.target) if c.target > 0 else None
        for j in range(ei, planned + 1):
            stop_hit = (lo[j] <= sp if side > 0 else hi[j] >= sp) if sp is not None else False
            target_hit = (hi[j] >= tp if side > 0 else lo[j] <= tp) if tp is not None else False
            # Conservative on bars where both could have occurred: stop first.
            if stop_hit:
                xi, xp, reason = j, float(sp), 'stop'
                break
            if target_hit:
                xi, xp, reason = j, float(tp), 'target'
                break
        gross = side * (xp / entry - 1.0)
        et, xt = idx[ei], idx[xi]
        fs = float(funding_events[(funding_events.index > et) & (funding_events.index <= xt)].sum())
        net = gross - side * fs - 2 * ONE_WAY_COST
        rec.append({'entry_time': et, 'exit_time': xt, 'side': side, 'net': net, 'gross': gross, 'funding_pnl': -side * fs, 'reason': reason})
        i = max(xi, i + 1)
    return pd.DataFrame(rec)


def metrics(t: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, extra: float = 0.0, seed: int = 1) -> dict:
    if t.empty:
        return {}
    z = t[(t['entry_time'] >= start) & (t['entry_time'] <= end)].copy()
    if z.empty:
        return {}
    r = z['net'].to_numpy(float) - extra
    eq = np.cumprod(1 + r)
    years = max((end - start).total_seconds() / (365.25 * 86400), 0.05)
    cagr = float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else -1.0
    peak = np.maximum.accumulate(eq)
    dd = float(np.min(eq / peak - 1))
    win = r[r > 0]
    loss = r[r < 0]
    pf = float(win.sum() / -loss.sum()) if len(loss) else float('inf')
    if len(r) >= 20:
        rng = np.random.default_rng(seed)
        means = rng.choice(r, size=(4000, len(r)), replace=True).mean(axis=1)
        boot5 = float(np.quantile(means, 0.05))
    else:
        boot5 = float('nan')
    return {
        'trades': int(len(r)), 'cagr': cagr, 'avg_trade': float(np.mean(r)),
        'profit_factor': pf, 'max_drawdown': dd, 'win_rate': float(np.mean(r > 0)),
        'bootstrap_mean_5pct': boot5,
    }


def dev_ok(m1: dict, m2: dict) -> bool:
    return bool(
        m1.get('trades', 0) >= 25 and m2.get('trades', 0) >= 15
        and m1.get('avg_trade', -1) > 0 and m2.get('avg_trade', -1) > 0
        and m1.get('profit_factor', 0) > 1.02 and m2.get('profit_factor', 0) > 1.08
        and m1.get('max_drawdown', -1) >= -0.40 and m2.get('max_drawdown', -1) >= -0.35
    )


def fingerprint(b: pd.DataFrame, c: v2.Config) -> str:
    s = v2.signal(b.loc[DEV_START:DEV_END], c).astype(np.int8).to_numpy()
    # Exact signal identity on development data is enough to collapse irrelevant parameter variants.
    import hashlib
    return hashlib.sha1(s.tobytes()).hexdigest()[:16]


def neighbors(c: v2.Config) -> List[v2.Config]:
    lbs = [3, 6, 12, 18, 30, 48]
    holds = [1, 2, 3, 6, 9, 12]
    ths = [0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05]
    stops = [0.0, 0.012, 0.018, 0.025, 0.035]
    targets = [0.0, 0.018, 0.03, 0.045, 0.06]

    def near(vals, x):
        vals = np.asarray(vals, dtype=float)
        k = int(np.argmin(abs(vals - float(x))))
        out = [float(vals[k])]
        if k > 0: out.append(float(vals[k - 1]))
        if k + 1 < len(vals): out.append(float(vals[k + 1]))
        return out

    out = [c]
    for x in near(lbs, c.lookback):
        out.append(v2.Config(c.family, c.symbol, c.side, int(x), c.hold, c.threshold, c.aux, c.stop, c.target))
    for x in near(holds, c.hold):
        out.append(v2.Config(c.family, c.symbol, c.side, c.lookback, int(x), c.threshold, c.aux, c.stop, c.target))
    for x in near(ths, c.threshold):
        out.append(v2.Config(c.family, c.symbol, c.side, c.lookback, c.hold, x, c.aux, c.stop, c.target))
    if c.family != 'weekday_momentum':
        for x in [max(0.15, c.aux * 0.8), c.aux, c.aux * 1.2]:
            out.append(v2.Config(c.family, c.symbol, c.side, c.lookback, c.hold, c.threshold, float(x), c.stop, c.target))
    for x in near(stops, c.stop):
        out.append(v2.Config(c.family, c.symbol, c.side, c.lookback, c.hold, c.threshold, c.aux, x, c.target))
    for x in near(targets, c.target):
        out.append(v2.Config(c.family, c.symbol, c.side, c.lookback, c.hold, c.threshold, c.aux, c.stop, x))
    seen = set(); ded = []
    for x in out:
        if x.key not in seen:
            seen.add(x.key); ded.append(x)
    return ded


def temporal_stability(t: pd.DataFrame) -> dict:
    blocks = [
        (pd.Timestamp('2023-01-01', tz='UTC'), pd.Timestamp('2023-06-30 23:59:59', tz='UTC')),
        (pd.Timestamp('2023-07-01', tz='UTC'), pd.Timestamp('2023-12-31 23:59:59', tz='UTC')),
        (pd.Timestamp('2024-01-01', tz='UTC'), pd.Timestamp('2024-06-30 23:59:59', tz='UTC')),
        (pd.Timestamp('2024-07-01', tz='UTC'), pd.Timestamp('2024-12-31 23:59:59', tz='UTC')),
        (pd.Timestamp('2025-01-01', tz='UTC'), DEV_END),
    ]
    vals = []
    for i, (a, b) in enumerate(blocks):
        m = metrics(t, a, b, seed=9100 + i)
        if m.get('trades', 0) >= 5:
            vals.append(m)
    pos = sum(1 for m in vals if m.get('avg_trade', -1) > 0)
    return {'valid_blocks': len(vals), 'positive_blocks': pos, 'positive_block_ratio': pos / len(vals) if vals else 0.0}


def hold_gate(m: dict, s: dict) -> bool:
    return bool(
        m.get('trades', 0) >= 20 and m.get('cagr', -1) >= 0.08
        and m.get('avg_trade', -1) > 0 and m.get('profit_factor', 0) >= 1.20
        and m.get('max_drawdown', -1) >= -0.25 and m.get('bootstrap_mean_5pct', -1) > 0
        and s.get('cagr', -1) >= 0.08 and s.get('profit_factor', 0) >= 1.05
    )


def external_check(base: v2.Config, bars: Dict[str, pd.DataFrame], events: Dict[str, pd.Series], seed: int) -> dict:
    rows = []
    for i, symbol in enumerate(EXTERNAL):
        c = v2.Config(base.family, symbol, base.side, base.lookback, base.hold, base.threshold, base.aux, base.stop, base.target)
        t = strict_simulate(bars[symbol], events[symbol], c)
        m = metrics(t, HOLD_START, HOLD_END, seed=seed + 11 * i)
        s = metrics(t, HOLD_START, HOLD_END, extra=2 * ONE_WAY_COST, seed=seed + 11 * i + 1)
        rows.append({'symbol': symbol, **m, **{f'stress_{k}': v for k, v in s.items()}})
    d = pd.DataFrame(rows)
    valid = d[d['trades'].fillna(0) >= 10].copy() if not d.empty else d
    pos = int((valid['avg_trade'] > 0).sum()) if not valid.empty else 0
    boot = int((valid['bootstrap_mean_5pct'] > 0).sum()) if not valid.empty else 0
    stress = int((valid['stress_avg_trade'] > 0).sum()) if not valid.empty else 0
    medpf = float(valid['profit_factor'].median()) if not valid.empty else float('nan')
    passed = bool(len(valid) >= 3 and pos >= 3 and boot >= 2 and stress >= 3 and medpf > 1.05)
    return {'passed': passed, 'valid_symbols': int(len(valid)), 'positive_symbols': pos, 'bootstrap_positive_symbols': boot, 'stress_positive_symbols': stress, 'median_pf': medpf, 'details': rows}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    state = load_state()
    audited = set(state.get('audited_keys', []))
    started = pd.Timestamp.now(tz='UTC')
    save_json(STATUS, {'status': 'running', 'started_at': started.isoformat(), 'stage': 'load_development_board'})

    if not v2.DEV_BOARD.exists():
        raise RuntimeError('Development leaderboard does not exist yet')
    board = pd.read_csv(v2.DEV_BOARD).sort_values('dev_score', ascending=False)
    board = board[board['dev_gate'] == True].copy()

    w3.START_MONTH = v2.START_MONTH; w3.END_MONTH = v2.END_MONTH; w3.INTERVAL = '1h'; w3.CACHE = Path('.cache_robust_audit')
    raw = {s: v2.base_bars(s) for s in PRIMARY}
    data = v2.add_btc_context(raw)
    events = {s: w3.load_funding(s) for s in PRIMARY}

    # Collapse development-equivalent signals before any holdout is touched.
    unique_rows = []
    seen_fp = set()
    for _, r in board.head(120).iterrows():
        if str(r['key']) in audited:
            continue
        c = cfg_from_row(r)
        fp = fingerprint(data[c.symbol], c)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        unique_rows.append(r)
        if len(unique_rows) >= 20:
            break

    results = []
    passes = []
    holdout_evaluated = 0
    external_checked = 0

    for n, r in enumerate(unique_rows):
        c = cfg_from_row(r)
        b = data[c.symbol]
        fe = events[c.symbol]
        base_t = strict_simulate(b, fe, c)
        tr = metrics(base_t, DEV_START, v2.DEV_TRAIN_END, seed=20000 + n)
        va = metrics(base_t, v2.DEV_VAL_START, DEV_END, seed=21000 + n)

        neigh = neighbors(c)
        neigh_pass = 0
        neigh_val_pfs = []
        for j, nc in enumerate(neigh):
            nt = strict_simulate(b, fe, nc)
            ntr = metrics(nt, DEV_START, v2.DEV_TRAIN_END, seed=22000 + n * 100 + j)
            nva = metrics(nt, v2.DEV_VAL_START, DEV_END, seed=23000 + n * 100 + j)
            if dev_ok(ntr, nva):
                neigh_pass += 1
            if nva.get('trades', 0) >= 10 and np.isfinite(nva.get('profit_factor', np.nan)):
                neigh_val_pfs.append(float(nva['profit_factor']))
        neigh_ratio = neigh_pass / max(len(neigh), 1)
        neigh_med_pf = float(np.median(neigh_val_pfs)) if neigh_val_pfs else float('nan')
        temp = temporal_stability(base_t)

        robust_dev = bool(
            dev_ok(tr, va)
            and neigh_ratio >= 0.60
            and np.isfinite(neigh_med_pf) and neigh_med_pf >= 1.08
            and temp['valid_blocks'] >= 3 and temp['positive_block_ratio'] >= 0.60
        )

        row = {
            **asdict(c), 'key': c.key, 'robust_dev': robust_dev,
            'neighbor_count': len(neigh), 'neighbor_pass_ratio': neigh_ratio,
            'neighbor_median_val_pf': neigh_med_pf, **temp,
            'train_trades': tr.get('trades'), 'train_pf': tr.get('profit_factor'), 'train_cagr': tr.get('cagr'),
            'val_trades': va.get('trades'), 'val_pf': va.get('profit_factor'), 'val_cagr': va.get('cagr'),
        }

        # One-shot holdout: mark audited before evaluation so future runs never use it again for selection.
        audited.add(str(r['key']))
        if robust_dev:
            holdout_evaluated += 1
            hm = metrics(base_t, HOLD_START, HOLD_END, seed=24000 + n)
            hs = metrics(base_t, HOLD_START, HOLD_END, extra=2 * ONE_WAY_COST, seed=25000 + n)
            hg = hold_gate(hm, hs)
            row.update({
                'holdout_gate': hg, 'holdout_trades': hm.get('trades'), 'holdout_pf': hm.get('profit_factor'),
                'holdout_cagr': hm.get('cagr'), 'holdout_dd': hm.get('max_drawdown'),
                'holdout_boot5': hm.get('bootstrap_mean_5pct'), 'stress_pf': hs.get('profit_factor'), 'stress_cagr': hs.get('cagr'),
            })
            if hg:
                save_json(STATUS, {'status': 'running', 'started_at': started.isoformat(), 'stage': 'external_confirmation', 'holdout_evaluated': holdout_evaluated})
                all_raw = {'BTCUSDT': raw['BTCUSDT']}
                for s in EXTERNAL:
                    all_raw[s] = v2.base_bars(s)
                all_ctx = v2.add_btc_context(all_raw)
                ext_bars = {s: all_ctx[s] for s in EXTERNAL}
                ext_events = {s: w3.load_funding(s) for s in EXTERNAL}
                external_checked += 1
                ext = external_check(c, ext_bars, ext_events, seed=26000 + n)
                row['external_pass'] = ext['passed']
                row['external_median_pf'] = ext['median_pf']
                if ext['passed']:
                    payload = {
                        'detected_at': pd.Timestamp.now(tz='UTC').isoformat(), 'config': asdict(c),
                        'development_robustness': {'neighbor_pass_ratio': neigh_ratio, 'neighbor_median_val_pf': neigh_med_pf, **temp},
                        'holdout': hm, 'stress': hs, 'external': ext,
                    }
                    passes.append(payload)
        results.append(row)

    state['audited_keys'] = list(audited)[-2000:]
    if passes:
        state.setdefault('strict_passes', []).extend(passes)
        state['strict_passes'] = state['strict_passes'][-20:]
        save_json(ALERT, {'strict_passes': passes})
    save_json(STATE, state)

    new = pd.DataFrame(results)
    if AUDIT.exists():
        try:
            old = pd.read_csv(AUDIT)
            new = pd.concat([old, new], ignore_index=True)
        except Exception:
            pass
    if not new.empty:
        new = new.drop_duplicates('key', keep='last')
    new.to_csv(AUDIT, index=False)

    top = pd.DataFrame(results)
    if not top.empty:
        top = top.sort_values(['robust_dev', 'neighbor_pass_ratio', 'val_pf'], ascending=False).head(10)
        table_cols = [c for c in ['key','family','symbol','side','robust_dev','neighbor_pass_ratio','neighbor_median_val_pf','valid_blocks','positive_block_ratio','train_trades','train_pf','val_trades','val_pf','holdout_gate','holdout_trades','holdout_pf','holdout_cagr','stress_pf','stress_cagr','external_pass'] if c in top.columns]
        table = top[table_cols].to_markdown(index=False, floatfmt='.4f')
    else:
        table = 'No new unaudited development candidates this run.'

    report = '\n'.join([
        '# Robustness audit — frozen development candidates', '',
        f'- Completed: {pd.Timestamp.now(tz="UTC").isoformat()}',
        f'- New development candidates audited: {len(results)}',
        f'- Robust-development survivors: {sum(bool(x.get("robust_dev")) for x in results)}',
        f'- One-shot holdout evaluations: {holdout_evaluated}',
        f'- External confirmations: {external_checked}',
        f'- Strict passes: {len(passes)}', '',
        'Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.', '',
        table,
    ])
    REPORT.write_text(report, encoding='utf-8')
    save_json(STATUS, {
        'status': 'completed', 'started_at': started.isoformat(), 'completed_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'new_candidates_audited': len(results), 'robust_development_survivors': sum(bool(x.get('robust_dev')) for x in results),
        'holdout_evaluated': holdout_evaluated, 'external_checked': external_checked,
        'strict_passes': len(passes), 'total_audited_keys': len(audited), 'alert_file': ALERT.exists(),
    })
    print(report)


if __name__ == '__main__':
    main()
