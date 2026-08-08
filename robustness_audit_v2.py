from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import continuous_research_v2 as v2
import robustness_audit as r1
import wave3_funding_carry as w3

OUT = Path('results_robust_audit_v2')
STATE = OUT / 'state.json'
STATUS = OUT / 'status.json'
REPORT = OUT / 'latest_report.md'
AUDIT = OUT / 'audit_results.csv'
ALERT = OUT / 'ALERT.json'

PRIMARY = ['BTCUSDT', 'SOLUSDT']
EXTERNAL = ['ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT']


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'audited_keys': [], 'audited_fingerprints': [], 'strict_passes': []}


def robust_dev_gate(tr: dict, va: dict, va_stress: dict, neigh_ratio: float, neigh_med_pf: float, temp: dict) -> bool:
    # Deliberately stricter than v1. Goal is to preserve holdout rather than maximize candidate count.
    return bool(
        tr.get('trades', 0) >= 35
        and va.get('trades', 0) >= 25
        and tr.get('avg_trade', -1) > 0.0005
        and va.get('avg_trade', -1) > 0.0007
        and tr.get('profit_factor', 0) >= 1.15
        and va.get('profit_factor', 0) >= 1.25
        and tr.get('bootstrap_mean_5pct', -1) > 0
        and va.get('bootstrap_mean_5pct', -1) > 0
        and va_stress.get('avg_trade', -1) > 0
        and va_stress.get('profit_factor', 0) >= 1.08
        and neigh_ratio >= 0.75
        and np.isfinite(neigh_med_pf) and neigh_med_pf >= 1.15
        and temp.get('valid_blocks', 0) >= 4
        and temp.get('positive_block_ratio', 0) >= 0.80
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    state = load_state()
    audited_keys = set(map(str, state.get('audited_keys', [])))
    audited_fps = set(map(str, state.get('audited_fingerprints', [])))
    started = pd.Timestamp.now(tz='UTC')
    save(STATUS, {'status':'running','started_at':started.isoformat(),'stage':'loading'})

    if not v2.DEV_BOARD.exists():
        raise RuntimeError('Development leaderboard missing')
    board = pd.read_csv(v2.DEV_BOARD).sort_values('dev_score', ascending=False)
    board = board[board['dev_gate'] == True].copy()

    w3.START_MONTH = v2.START_MONTH
    w3.END_MONTH = v2.END_MONTH
    w3.INTERVAL = '1h'
    w3.CACHE = Path('.cache_robust_audit_v2')

    raw = {s: v2.base_bars(s) for s in PRIMARY}
    data = v2.add_btc_context(raw)
    events = {s: w3.load_funding(s) for s in PRIMARY}

    candidates = []
    local_fps = set()
    for _, row in board.head(250).iterrows():
        key = str(row['key'])
        if key in audited_keys:
            continue
        c = r1.cfg_from_row(row)
        fp = r1.fingerprint(data[c.symbol], c)
        # Cross-run fingerprint memory prevents repeated holdout peeking through equivalent parameterizations.
        if fp in audited_fps or fp in local_fps:
            audited_keys.add(key)
            continue
        local_fps.add(fp)
        candidates.append((row, c, fp))
        if len(candidates) >= 20:
            break

    rows = []
    passes = []
    robust_survivors = 0
    holdout_evaluated = 0
    external_checked = 0

    for n, (row, c, fp) in enumerate(candidates):
        b = data[c.symbol]
        fe = events[c.symbol]
        t = r1.strict_simulate(b, fe, c)
        tr = r1.metrics(t, r1.DEV_START, v2.DEV_TRAIN_END, seed=31000+n)
        va = r1.metrics(t, v2.DEV_VAL_START, r1.DEV_END, seed=32000+n)
        va_stress = r1.metrics(t, v2.DEV_VAL_START, r1.DEV_END, extra=2*v2.ONE_WAY_COST, seed=33000+n)

        neigh = r1.neighbors(c)
        n_pass = 0
        n_pfs = []
        for j, nc in enumerate(neigh):
            nt = r1.strict_simulate(b, fe, nc)
            ntr = r1.metrics(nt, r1.DEV_START, v2.DEV_TRAIN_END, seed=34000+n*100+j)
            nva = r1.metrics(nt, v2.DEV_VAL_START, r1.DEV_END, seed=35000+n*100+j)
            nvs = r1.metrics(nt, v2.DEV_VAL_START, r1.DEV_END, extra=2*v2.ONE_WAY_COST, seed=36000+n*100+j)
            if (
                ntr.get('avg_trade', -1) > 0 and nva.get('avg_trade', -1) > 0
                and ntr.get('profit_factor', 0) >= 1.05 and nva.get('profit_factor', 0) >= 1.10
                and nvs.get('avg_trade', -1) > 0
            ):
                n_pass += 1
            if nva.get('trades', 0) >= 10 and np.isfinite(nva.get('profit_factor', np.nan)):
                n_pfs.append(float(nva['profit_factor']))
        neigh_ratio = n_pass / max(len(neigh), 1)
        neigh_med_pf = float(np.median(n_pfs)) if n_pfs else float('nan')
        temp = r1.temporal_stability(t)

        robust_dev = robust_dev_gate(tr, va, va_stress, neigh_ratio, neigh_med_pf, temp)
        result = {
            **asdict(c), 'key':c.key, 'fingerprint':fp, 'robust_dev':robust_dev,
            'neighbor_pass_ratio':neigh_ratio, 'neighbor_median_val_pf':neigh_med_pf,
            **temp,
            'train_trades':tr.get('trades'), 'train_pf':tr.get('profit_factor'), 'train_cagr':tr.get('cagr'), 'train_boot5':tr.get('bootstrap_mean_5pct'),
            'val_trades':va.get('trades'), 'val_pf':va.get('profit_factor'), 'val_cagr':va.get('cagr'), 'val_boot5':va.get('bootstrap_mean_5pct'),
            'val_stress_pf':va_stress.get('profit_factor'), 'val_stress_cagr':va_stress.get('cagr'),
        }

        # Mark both exact key and development signal fingerprint before any holdout access.
        audited_keys.add(str(row['key']))
        audited_fps.add(fp)

        if robust_dev:
            robust_survivors += 1
            holdout_evaluated += 1
            hm = r1.metrics(t, r1.HOLD_START, r1.HOLD_END, seed=37000+n)
            hs = r1.metrics(t, r1.HOLD_START, r1.HOLD_END, extra=2*v2.ONE_WAY_COST, seed=38000+n)
            hg = r1.hold_gate(hm, hs)
            result.update({
                'holdout_gate':hg, 'holdout_trades':hm.get('trades'), 'holdout_pf':hm.get('profit_factor'),
                'holdout_cagr':hm.get('cagr'), 'holdout_dd':hm.get('max_drawdown'), 'holdout_boot5':hm.get('bootstrap_mean_5pct'),
                'stress_pf':hs.get('profit_factor'), 'stress_cagr':hs.get('cagr')
            })
            if hg:
                all_raw = {'BTCUSDT': raw['BTCUSDT']}
                for s in EXTERNAL:
                    all_raw[s] = v2.base_bars(s)
                all_ctx = v2.add_btc_context(all_raw)
                ext_bars = {s: all_ctx[s] for s in EXTERNAL}
                ext_events = {s: w3.load_funding(s) for s in EXTERNAL}
                external_checked += 1
                ext = r1.external_check(c, ext_bars, ext_events, seed=39000+n)
                result['external_pass'] = ext['passed']
                result['external_median_pf'] = ext['median_pf']
                if ext['passed']:
                    passes.append({
                        'detected_at':pd.Timestamp.now(tz='UTC').isoformat(),
                        'config':asdict(c),
                        'development_robustness':{
                            'neighbor_pass_ratio':neigh_ratio,
                            'neighbor_median_val_pf':neigh_med_pf,
                            'train_boot5':tr.get('bootstrap_mean_5pct'),
                            'val_boot5':va.get('bootstrap_mean_5pct'),
                            **temp,
                        },
                        'holdout':hm,'stress':hs,'external':ext,
                    })
        rows.append(result)

    state['audited_keys'] = list(audited_keys)[-5000:]
    state['audited_fingerprints'] = list(audited_fps)[-5000:]
    if passes:
        state.setdefault('strict_passes', []).extend(passes)
        state['strict_passes'] = state['strict_passes'][-20:]
        save(ALERT, {'strict_passes':passes})
    save(STATE, state)

    new = pd.DataFrame(rows)
    if AUDIT.exists():
        try:
            old = pd.read_csv(AUDIT)
            new = pd.concat([old,new], ignore_index=True)
        except Exception:
            pass
    if not new.empty:
        new = new.drop_duplicates('fingerprint', keep='last')
    new.to_csv(AUDIT, index=False)

    top = pd.DataFrame(rows)
    if not top.empty:
        top = top.sort_values(['robust_dev','neighbor_pass_ratio','val_pf'], ascending=False).head(10)
        cols = [x for x in ['key','family','symbol','side','robust_dev','neighbor_pass_ratio','neighbor_median_val_pf','valid_blocks','positive_block_ratio','train_trades','train_pf','train_boot5','val_trades','val_pf','val_boot5','val_stress_pf','holdout_gate','holdout_trades','holdout_pf','holdout_cagr','stress_pf','stress_cagr','external_pass'] if x in top.columns]
        table = top[cols].to_markdown(index=False, floatfmt='.4f')
    else:
        table = 'No new development fingerprints available.'

    report = '\n'.join([
        '# Robustness audit v2 — holdout-preserving', '',
        f'- Completed: {pd.Timestamp.now(tz="UTC").isoformat()}',
        f'- New unique development fingerprints audited: {len(rows)}',
        f'- Robust-development survivors: {robust_survivors}',
        f'- One-shot holdout evaluations: {holdout_evaluated}',
        f'- External confirmations: {external_checked}',
        f'- Strict passes: {len(passes)}',
        f'- Total permanently audited fingerprints: {len(audited_fps)}', '',
        'Cross-run signal fingerprints are permanently remembered. Equivalent future parameterizations cannot re-open holdout.', '',
        table,
    ])
    REPORT.write_text(report, encoding='utf-8')
    save(STATUS, {
        'status':'completed','started_at':started.isoformat(),'completed_at':pd.Timestamp.now(tz='UTC').isoformat(),
        'new_unique_fingerprints':len(rows),'robust_development_survivors':robust_survivors,
        'holdout_evaluated':holdout_evaluated,'external_checked':external_checked,'strict_passes':len(passes),
        'total_audited_fingerprints':len(audited_fps),'alert_file':ALERT.exists(),
    })
    print(report)


if __name__ == '__main__':
    main()
