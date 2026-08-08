from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path('results_microstructure')
STATE = OUT / 'state.json'
STATUS = OUT / 'status.json'
REPORT = OUT / 'latest_report.md'
LEADER = OUT / 'dev_leaderboard.csv'
ALERT = OUT / 'ALERT.json'
CACHE = Path('.cache_microstructure')

START_MONTH = '2023-01'
END_MONTH = '2026-07'
TRAIN_END = pd.Timestamp('2024-06-30 23:59:59', tz='UTC')
VAL_START = pd.Timestamp('2024-07-01', tz='UTC')
VAL_END = pd.Timestamp('2025-06-30 23:59:59', tz='UTC')
HOLD_START = pd.Timestamp('2025-07-01', tz='UTC')
HOLD_END = pd.Timestamp('2026-07-31 23:59:59', tz='UTC')
PRIMARY = ['BTCUSDT','SOLUSDT']
EXTERNAL = ['ETHUSDT','BNBUSDT','XRPUSDT']
ONE_WAY_COST = 0.0008


@dataclass(frozen=True)
class Config:
    family: str
    symbol: str
    side: str
    lookback: int
    hold: int
    q: float
    aux: float
    stop: float
    target: float

    @property
    def key(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def month_range(start: str, end: str) -> Iterable[str]:
    yield from (str(x) for x in pd.period_range(start=start, end=end, freq='M'))


def rich_kline(symbol: str) -> pd.DataFrame:
    parts = []
    base = 'https://data.binance.vision/data/futures/um/monthly/klines'
    names = ['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
    for month in month_range(START_MONTH, END_MONTH):
        fn = f'{symbol}-1h-{month}.zip'
        raw = w3.get_bytes(f'{base}/{symbol}/1h/{fn}', CACHE / 'perp' / fn)
        if raw is None:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith('.csv'))
            with z.open(name) as h:
                f = pd.read_csv(h, header=None, dtype=str)
        if f.shape[1] < 12:
            continue
        f = f.iloc[:, :12]
        f.columns = names
        f['time'] = w3.parse_timestamp(f['open_time'])
        for c in ['open','high','low','close','volume','quote_volume','taker_buy_quote']:
            f[c] = pd.to_numeric(f[c], errors='coerce')
        f = f.dropna(subset=['time','open','high','low','close','quote_volume','taker_buy_quote'])
        parts.append(f[['time','open','high','low','close','volume','quote_volume','taker_buy_quote']])
    if not parts:
        raise RuntimeError(f'No rich klines for {symbol}')
    x = pd.concat(parts, ignore_index=True).drop_duplicates('time').set_index('time').sort_index()
    return x.loc[(x.index >= pd.Timestamp('2023-01-01', tz='UTC')) & (x.index <= HOLD_END)]


def prepare(symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
    x = rich_kline(symbol)
    f = w3.load_funding(symbol)
    b = x.resample('4h', origin='start_day').agg(
        open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'),
        volume=('volume','sum'), quote_volume=('quote_volume','sum'), taker_buy_quote=('taker_buy_quote','sum')
    ).dropna()
    b['taker_ratio'] = b['taker_buy_quote'] / b['quote_volume'].replace(0, np.nan)
    prev = b['close'].shift(1)
    tr = pd.concat([(b['high']-b['low']), (b['high']-prev).abs(), (b['low']-prev).abs()], axis=1).max(axis=1)
    b['atr_pct'] = tr.rolling(21).mean() / b['close']
    b['ret1'] = b['close'].pct_change()
    b['body'] = (b['close'] - b['open']) / b['open']
    rng = (b['high'] - b['low']).replace(0, np.nan)
    b['close_loc'] = (b['close'] - b['low']) / rng
    b['upper_wick'] = (b['high'] - b[['open','close']].max(axis=1)) / b['close']
    b['lower_wick'] = (b[['open','close']].min(axis=1) - b['low']) / b['close']
    b['vol_ratio'] = b['quote_volume'] / b['quote_volume'].rolling(42).median().replace(0, np.nan)
    b['ema50'] = b['close'].ewm(span=50, adjust=False).mean()
    b['ema200'] = b['close'].ewm(span=200, adjust=False).mean()
    for lb in [3,6,12,18,30,48]:
        b[f'ret{lb}'] = b['close'].pct_change(lb)
        b[f'hi{lb}'] = b['high'].rolling(lb).max().shift(1)
        b[f'lo{lb}'] = b['low'].rolling(lb).min().shift(1)
        b[f'taker_z{lb}'] = (b['taker_ratio'] - b['taker_ratio'].rolling(lb*3).mean()) / b['taker_ratio'].rolling(lb*3).std().replace(0, np.nan)
        b[f'vol_z{lb}'] = (b['quote_volume'] - b['quote_volume'].rolling(lb*3).mean()) / b['quote_volume'].rolling(lb*3).std().replace(0, np.nan)
    ff = f.copy().sort_index()
    return b, ff


def config_space(batch: int, symbol: str) -> List[Config]:
    rng = np.random.default_rng(20260808 + batch*137 + (0 if symbol == 'BTCUSDT' else 59))
    fams = ['taker_absorption','taker_momentum','wick_reversal','low_volume_breakout','climax_reversal','funding_taker_divergence','settlement_drift','failed_breakout']
    chosen = [fams[batch % len(fams)], fams[(batch+2)%len(fams)], fams[(batch+5)%len(fams)]]
    out = []
    for fam in chosen:
        for _ in range(60):
            out.append(Config(
                fam, symbol, str(rng.choice(['long','short','both'])), int(rng.choice([3,6,12,18,30,48])),
                int(rng.choice([1,2,3,6,9])), float(rng.choice([0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90])),
                float(rng.choice([0.5,0.75,1.0,1.25,1.5,2.0,2.5])), float(rng.choice([0.0,0.012,0.018,0.025,0.035])),
                float(rng.choice([0.0,0.018,0.03,0.045,0.06]))
            ))
    seen=set(); ded=[]
    for c in out:
        if c.key not in seen:
            seen.add(c.key); ded.append(c)
    return ded


def signal(b: pd.DataFrame, cfg: Config) -> pd.Series:
    lb=cfg.lookback; r=b[f'ret{lb}']; tz=b[f'taker_z{lb}']; vz=b[f'vol_z{lb}']; hi=b[f'hi{lb}']; lo=b[f'lo{lb}']
    long=pd.Series(False,index=b.index); short=pd.Series(False,index=b.index)
    if cfg.family=='taker_absorption':
        long=(tz <= -cfg.aux) & (r > -0.01) & (b['close_loc'] > cfg.q)
        short=(tz >= cfg.aux) & (r < 0.01) & (b['close_loc'] < 1-cfg.q)
    elif cfg.family=='taker_momentum':
        long=(tz >= cfg.aux) & (r > 0) & (b['close'] > b['ema200'])
        short=(tz <= -cfg.aux) & (r < 0) & (b['close'] < b['ema200'])
    elif cfg.family=='wick_reversal':
        long=(b['lower_wick'] > b['atr_pct']*cfg.aux) & (b['close_loc'] > cfg.q) & (r < 0)
        short=(b['upper_wick'] > b['atr_pct']*cfg.aux) & (b['close_loc'] < 1-cfg.q) & (r > 0)
    elif cfg.family=='low_volume_breakout':
        long=(b['close'] > hi) & (b['vol_ratio'] < cfg.aux) & (b['close'] > b['ema200'])
        short=(b['close'] < lo) & (b['vol_ratio'] < cfg.aux) & (b['close'] < b['ema200'])
    elif cfg.family=='climax_reversal':
        long=(vz >= cfg.aux) & (r < 0) & (b['close_loc'] > cfg.q)
        short=(vz >= cfg.aux) & (r > 0) & (b['close_loc'] < 1-cfg.q)
    elif cfg.family=='funding_taker_divergence':
        # Funding is joined only as the last known event strictly before the signal bar.
        fz=b['funding_z']
        long=(fz <= -cfg.aux) & (tz >= cfg.aux*0.5) & (r <= 0)
        short=(fz >= cfg.aux) & (tz <= -cfg.aux*0.5) & (r >= 0)
    elif cfg.family=='settlement_drift':
        hour=b.index.hour
        settlement=pd.Series(np.isin(hour,[0,8,16]),index=b.index)
        long=settlement & (tz >= cfg.aux) & (r > 0)
        short=settlement & (tz <= -cfg.aux) & (r < 0)
    elif cfg.family=='failed_breakout':
        ph=(b['high'].shift(1)>hi.shift(1)); pl=(b['low'].shift(1)<lo.shift(1))
        long=pl & (b['close']>lo) & (b['close_loc']>cfg.q)
        short=ph & (b['close']<hi) & (b['close_loc']<1-cfg.q)
    s=pd.Series(0,index=b.index,dtype=int)
    if cfg.side in ('long','both'): s.loc[long.fillna(False)]=1
    if cfg.side in ('short','both'): s.loc[short.fillna(False)]=-1
    return s


def add_funding_features(b: pd.DataFrame, f: pd.Series) -> pd.DataFrame:
    out=b.copy()
    # Only information known before each 4h signal bar is allowed.
    events=f.sort_index()
    known=events.reindex(out.index, method='ffill').shift(1)
    mu=known.rolling(126).mean(); sd=known.rolling(126).std().replace(0,np.nan)
    out['funding_z']=(known-mu)/sd
    return out


def simulate(b: pd.DataFrame, funding_events: pd.Series, cfg: Config) -> pd.DataFrame:
    sig=signal(b,cfg).to_numpy(int); idx=b.index; op=b['open'].to_numpy(float); hi=b['high'].to_numpy(float); lo=b['low'].to_numpy(float)
    rec=[]; i=0
    while i < len(b)-cfg.hold-2:
        if sig[i]==0: i+=1; continue
        side=int(sig[i]); ei=i+1; entry=float(op[ei]); xi=min(ei+cfg.hold,len(b)-1); xp=float(op[xi]); reason='time'
        sp=entry*(1-cfg.stop if side>0 else 1+cfg.stop) if cfg.stop>0 else None
        tp=entry*(1+cfg.target if side>0 else 1-cfg.target) if cfg.target>0 else None
        for j in range(ei,xi+1):
            sh=(lo[j]<=sp if side>0 else hi[j]>=sp) if sp is not None else False
            th=(hi[j]>=tp if side>0 else lo[j]<=tp) if tp is not None else False
            if sh: xi=j; xp=float(sp); reason='stop'; break
            if th: xi=j; xp=float(tp); reason='target'; break
        gross=side*(xp/entry-1)
        et=idx[ei]; xt=idx[xi]
        # Strict settlement accounting: only funding timestamps after entry and up to exit.
        fs=float(funding_events[(funding_events.index>et)&(funding_events.index<=xt)].sum())
        net=gross - side*fs - 2*ONE_WAY_COST
        rec.append({'entry_time':et,'exit_time':xt,'net':net,'gross':gross,'side':side,'reason':reason})
        i=max(xi,i+1)
    return pd.DataFrame(rec)


def boot5(r: np.ndarray, seed: int) -> float:
    if len(r)<20: return float('nan')
    rng=np.random.default_rng(seed); m=rng.choice(r,size=(2000,len(r)),replace=True).mean(axis=1)
    return float(np.quantile(m,0.05))


def metrics(t: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, extra: float=0.0, seed: int=1) -> dict:
    if t.empty: return {}
    x=t[(t.entry_time>=start)&(t.entry_time<=end)].copy()
    if x.empty: return {}
    r=x.net.to_numpy(float)-extra; eq=np.cumprod(1+r); years=max((end-start).total_seconds()/(365.25*86400),0.05)
    cagr=float(eq[-1]**(1/years)-1) if eq[-1]>0 else -1.0; peak=np.maximum.accumulate(eq); dd=float(np.min(eq/peak-1))
    win=r[r>0]; loss=r[r<0]; pf=float(win.sum()/-loss.sum()) if len(loss) else float('inf')
    return {'trades':int(len(r)),'cagr':cagr,'avg_trade':float(np.mean(r)),'profit_factor':pf,'max_drawdown':dd,'win_rate':float(np.mean(r>0)),'bootstrap_mean_5pct':boot5(r,seed)}


def dev_gate(row: dict) -> bool:
    return bool(row.get('train_trades',0)>=25 and row.get('val_trades',0)>=20 and row.get('train_avg_trade',-1)>0 and row.get('val_avg_trade',-1)>0 and row.get('train_profit_factor',0)>1.1 and row.get('val_profit_factor',0)>1.15 and row.get('train_max_drawdown',-1)>=-0.35 and row.get('val_max_drawdown',-1)>=-0.30)


def hold_gate(m: dict, s: dict) -> bool:
    return bool(m.get('trades',0)>=20 and m.get('cagr',-1)>=0.08 and m.get('profit_factor',0)>=1.20 and m.get('max_drawdown',-1)>=-0.25 and m.get('bootstrap_mean_5pct',-1)>0 and s.get('cagr',-1)>=0.08 and s.get('profit_factor',0)>=1.05)


def load_state() -> dict:
    if STATE.exists():
        try: return json.loads(STATE.read_text())
        except Exception: pass
    return {'next_batch':0,'strict_passes':[]}


def save(path: Path, obj: dict):
    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True); st=load_state(); batch=int(st.get('next_batch',0)); started=pd.Timestamp.now(tz='UTC')
    save(STATUS,{'status':'running','batch':batch,'stage':'load','started_at':started.isoformat()})
    w3.START_MONTH=START_MONTH; w3.END_MONTH=END_MONTH; w3.CACHE=CACHE/'funding'
    data: Dict[str,Tuple[pd.DataFrame,pd.Series]]={}
    for sym in PRIMARY:
        b,f=prepare(sym); data[sym]=(add_funding_features(b,f),f)
    rows=[]
    for sym,(b,f) in data.items():
        for j,cfg in enumerate(config_space(batch,sym)):
            t=simulate(b,f,cfg)
            tr=metrics(t,pd.Timestamp('2023-01-01',tz='UTC'),TRAIN_END,seed=batch*10000+j)
            va=metrics(t,VAL_START,VAL_END,seed=batch*10000+j+1)
            row={**asdict(cfg),'key':cfg.key,'batch':batch}
            for p,d in [('train',tr),('val',va)]:
                for k,v in d.items(): row[f'{p}_{k}']=v
            row['development_gate']=dev_gate(row); rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(OUT/f'dev_batch_{batch:05d}.csv',index=False)
    if LEADER.exists():
        old=pd.read_csv(LEADER); all_df=pd.concat([old,df],ignore_index=True).drop_duplicates('key',keep='last')
    else: all_df=df.copy()
    all_df['dev_score']=all_df['val_profit_factor'].clip(upper=4).fillna(0)+0.35*all_df['train_profit_factor'].clip(upper=4).fillna(0)+0.25*all_df['val_cagr'].clip(-1,2).fillna(-1)
    all_df=all_df.sort_values('dev_score',ascending=False).head(500); all_df.to_csv(LEADER,index=False)
    candidates=df[df.development_gate==True].sort_values('dev_score' if 'dev_score' in df.columns else 'val_profit_factor',ascending=False).head(10)
    hold_rows=[]; strict=[]
    for k,(_,r) in enumerate(candidates.iterrows()):
        cfg=Config(r.family,r.symbol,r.side,int(r.lookback),int(r.hold),float(r.q),float(r.aux),float(r.stop),float(r.target))
        b,f=data[cfg.symbol]; t=simulate(b,f,cfg)
        hm=metrics(t,HOLD_START,HOLD_END,seed=99000+batch*100+k); hs=metrics(t,HOLD_START,HOLD_END,extra=2*ONE_WAY_COST,seed=99100+batch*100+k)
        rr={'key':cfg.key,**asdict(cfg),**{f'hold_{a}':v for a,v in hm.items()},**{f'stress_{a}':v for a,v in hs.items()}}; rr['hold_gate']=hold_gate(hm,hs); hold_rows.append(rr)
        if rr['hold_gate']: strict.append((cfg,rr))
    pd.DataFrame(hold_rows).to_csv(OUT/f'holdout_batch_{batch:05d}.csv',index=False)
    external_passes=[]
    if strict:
        extdata={}
        for sym in EXTERNAL:
            b,f=prepare(sym); extdata[sym]=(add_funding_features(b,f),f)
        for cfg,rr in strict[:3]:
            details=[]
            for sym,(b,f) in extdata.items():
                c2=Config(cfg.family,sym,cfg.side,cfg.lookback,cfg.hold,cfg.q,cfg.aux,cfg.stop,cfg.target)
                t=simulate(b,f,c2); m=metrics(t,HOLD_START,HOLD_END,seed=1234+batch); s=metrics(t,HOLD_START,HOLD_END,extra=2*ONE_WAY_COST,seed=2234+batch)
                details.append({'symbol':sym,**m,**{f'stress_{k}':v for k,v in s.items()}})
            ed=pd.DataFrame(details); valid=ed[ed.trades.fillna(0)>=10] if not ed.empty else ed
            passed=bool(len(valid)>=2 and (valid.avg_trade>0).sum()>=2 and (valid.bootstrap_mean_5pct>0).sum()>=2 and valid.profit_factor.median()>1.05 and (valid.stress_avg_trade>0).sum()>=2)
            if passed: external_passes.append({'batch':batch,'config':asdict(cfg),'holdout':rr,'external':details})
    if external_passes:
        save(ALERT,{'strict_passes':external_passes,'detected_at':pd.Timestamp.now(tz='UTC').isoformat()})
        st.setdefault('strict_passes',[]).extend(external_passes); st['strict_passes']=st['strict_passes'][-20:]
    st['next_batch']=batch+1; st['last_completed_batch']=batch; st['last_completed_at']=pd.Timestamp.now(tz='UTC').isoformat(); save(STATE,st)
    top=all_df.head(10); cols=[c for c in ['key','family','symbol','side','lookback','hold','q','aux','stop','target','train_trades','train_profit_factor','train_cagr','val_trades','val_profit_factor','val_cagr','train_max_drawdown','val_max_drawdown','dev_score'] if c in top.columns]
    report='\n'.join(['# Microstructure research','',f'- Batch: {batch}',f'- Tested: {len(df)}',f'- Development survivors: {int(df.development_gate.sum())}',f'- Holdout evaluated: {len(hold_rows)}',f'- Holdout gates passed: {len(strict)}',f'- External strict passes: {len(external_passes)}','','## Development leaderboard (holdout hidden)','',top[cols].to_markdown(index=False,floatfmt='.4f')])
    REPORT.write_text(report,encoding='utf-8')
    save(STATUS,{'status':'completed','batch':batch,'started_at':started.isoformat(),'completed_at':pd.Timestamp.now(tz='UTC').isoformat(),'tested':len(df),'development_survivors':int(df.development_gate.sum()),'holdout_evaluated':len(hold_rows),'holdout_gates_passed':len(strict),'strict_passes':len(external_passes),'alert_file':ALERT.exists()})
    print(report)

if __name__=='__main__': main()
