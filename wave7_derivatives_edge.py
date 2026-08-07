from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path('results_wave7')
CACHE = Path('.cache_wave7_metrics')
START_MONTH = '2023-01'
END_MONTH = '2026-07'
TRAIN_END = pd.Timestamp('2024-12-31 23:59:59', tz='UTC')
TEST_START = pd.Timestamp('2025-01-01', tz='UTC')
ONE_WAY_COST = 0.0008
SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','LINKUSDT','LTCUSDT','AVAXUSDT']
METRICS_URL = 'https://data.binance.vision/data/futures/um/monthly/metrics'


@dataclass(frozen=True)
class Config:
    family: str
    signal_name: str
    side: int
    tp: float
    sl: float
    hold: int

    @property
    def name(self) -> str:
        d = 'L' if self.side == 1 else 'S'
        return f'{self.signal_name}_{d}_tp{self.tp:.3f}_sl{self.sl:.3f}_h{self.hold}'


def months():
    return [str(x) for x in pd.period_range(START_MONTH, END_MONTH, freq='M')]


def parse_time(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors='coerce')
    valid = n.dropna()
    if not valid.empty:
        med = float(valid.abs().median())
        if med > 1e14:
            return pd.to_datetime(n, unit='us', utc=True, errors='coerce')
        if med > 1e11:
            return pd.to_datetime(n, unit='ms', utc=True, errors='coerce')
        if med > 1e8:
            return pd.to_datetime(n, unit='s', utc=True, errors='coerce')
    return pd.to_datetime(s, utc=True, errors='coerce')


def load_metrics(symbol: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for ym in months():
        fname = f'{symbol}-metrics-{ym}.zip'
        url = f'{METRICS_URL}/{symbol}/{fname}'
        raw = w3.get_bytes(url, CACHE / symbol / fname)
        if raw is None:
            print('MISSING metrics', fname)
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                member = next(x for x in zf.namelist() if x.lower().endswith('.csv'))
                with zf.open(member) as fh:
                    frame = pd.read_csv(fh)
        except Exception as exc:
            print('BAD metrics', fname, exc)
            continue
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        tcol = next((c for c in frame.columns if 'time' in c), None)
        if tcol is None:
            print('NO TIME COL', fname, frame.columns.tolist())
            continue
        frame['time'] = parse_time(frame[tcol])
        wanted = [
            'sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio',
            'sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio'
        ]
        for c in wanted:
            frame[c] = pd.to_numeric(frame[c], errors='coerce') if c in frame.columns else np.nan
        frame = frame.dropna(subset=['time']).set_index('time').sort_index()
        parts.append(frame[wanted])
    if not parts:
        raise RuntimeError(f'No metrics for {symbol}')
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    agg = {
        'sum_open_interest':'last','sum_open_interest_value':'last',
        'count_toptrader_long_short_ratio':'last','sum_toptrader_long_short_ratio':'last',
        'count_long_short_ratio':'last','sum_taker_long_short_vol_ratio':'mean'
    }
    return out.resample('4h', origin='start_day').agg(agg)


def rsi(series: pd.Series, period: int = 7) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))


def load_data() -> Dict[str, pd.DataFrame]:
    w3.START_MONTH = START_MONTH
    w3.END_MONTH = END_MONTH
    w3.CACHE = Path('.cache_wave3')
    result: Dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        print('Loading', s)
        perp = w3.load_kline(s, 'perp')
        spot = w3.load_kline(s, 'spot')
        funding = w3.load_funding(s)
        metrics = load_metrics(s)
        bars = perp.resample('4h', origin='start_day').agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum'}
        ).dropna()
        spot4 = spot['close'].resample('4h', origin='start_day').last()
        bars['spot_close'] = spot4.reindex(bars.index)
        bars['basis'] = bars['close'] / bars['spot_close'] - 1
        bars['funding_event'] = funding.resample('4h', origin='start_day').sum().reindex(bars.index).fillna(0.0)
        bars = bars.join(metrics.reindex(bars.index), how='left')
        metric_cols = [
            'sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio',
            'sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio'
        ]
        bars[metric_cols] = bars[metric_cols].ffill(limit=2)
        bars['ema200'] = bars['close'].ewm(span=200, adjust=False).mean()
        bars['rsi7'] = rsi(bars['close'])
        tr = pd.concat([
            bars['high']-bars['low'],
            (bars['high']-bars['close'].shift()).abs(),
            (bars['low']-bars['close'].shift()).abs(),
        ], axis=1).max(axis=1)
        bars['atr_pct'] = tr.rolling(21).mean()/bars['close']
        bars['atr_rank126'] = bars['atr_pct'].rolling(126).rank(pct=True)
        bars['funding24'] = bars['funding_event'].rolling(6).sum()
        bars['funding_z126'] = (bars['funding24']-bars['funding24'].rolling(126).mean())/bars['funding24'].rolling(126).std().replace(0,np.nan)
        bars['basis_z126'] = (bars['basis']-bars['basis'].rolling(126).mean())/bars['basis'].rolling(126).std().replace(0,np.nan)
        bars['oi_chg6'] = bars['sum_open_interest_value'].pct_change(6)
        bars['oi_chg18'] = bars['sum_open_interest_value'].pct_change(18)
        bars['ret6'] = bars['close'].pct_change(6)
        for lb in (6,18):
            bars[f'prior_high_{lb}'] = bars['high'].rolling(lb).max().shift(1)
            bars[f'prior_low_{lb}'] = bars['low'].rolling(lb).min().shift(1)
        result[s] = bars
    start = max(x.index.min() for x in result.values())
    end = min(x.index.max() for x in result.values())
    return {s: x.loc[start:end].copy() for s,x in result.items()}


def aligned(data: Dict[str,pd.DataFrame], field: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({s:data[s][field].reindex(idx) for s in SYMBOLS}, index=idx)


def signal_map(data: Dict[str,pd.DataFrame], idx: pd.DatetimeIndex):
    close = aligned(data,'close',idx)
    atr_rank = aligned(data,'atr_rank126',idx)
    funding = aligned(data,'funding24',idx)
    funding_z = aligned(data,'funding_z126',idx)
    basis_z = aligned(data,'basis_z126',idx)
    oi6 = aligned(data,'oi_chg6',idx)
    oi18 = aligned(data,'oi_chg18',idx)
    retail = aligned(data,'count_long_short_ratio',idx)
    top = aligned(data,'sum_toptrader_long_short_ratio',idx)
    taker = aligned(data,'sum_taker_long_short_vol_ratio',idx)
    btc = data['BTCUSDT'].reindex(idx)
    bull = (btc['close'] > btc['ema200']).to_numpy()[:,None]
    bear = (btc['close'] < btc['ema200']).to_numpy()[:,None]
    out = {}

    for lb in (6,18):
        hi = aligned(data,f'prior_high_{lb}',idx)
        lo = aligned(data,f'prior_low_{lb}',idx)
        for comp in (0.20,0.35):
            for oilb, oi in ((6,oi6),(18,oi18)):
                for oi_th in (0.02,0.05):
                    # Crowded longs unwind: positive funding / long crowd / sell-flow / downside break.
                    eligible_s = (
                        bear & (close < lo).to_numpy() & (atr_rank <= comp).to_numpy()
                        & (oi >= oi_th).to_numpy() & (funding > 0).to_numpy()
                        & ((retail > 1.02) | (top > 1.02)).to_numpy() & (taker < 1.0).to_numpy()
                    )
                    score_s = (oi.clip(lower=0)*10 + funding_z.clip(lower=0) + basis_z.clip(lower=0) + (1/taker.clip(lower=0.05))).where(eligible_s)
                    name_s = f'crowded_long_break_lb{lb}_c{comp:g}_oi{oilb}_{oi_th:g}'
                    out[(name_s,'crowded_long_break',-1)] = score_s.to_numpy()

                    # Crowded shorts squeeze: negative funding / short crowd / buy-flow / upside break.
                    eligible_l = (
                        bull & (close > hi).to_numpy() & (atr_rank <= comp).to_numpy()
                        & (oi >= oi_th).to_numpy() & (funding < 0).to_numpy()
                        & ((retail < 0.98) | (top < 0.98)).to_numpy() & (taker > 1.0).to_numpy()
                    )
                    score_l = (oi.clip(lower=0)*10 + (-funding_z).clip(lower=0) + (-basis_z).clip(lower=0) + taker.clip(upper=5)).where(eligible_l)
                    name_l = f'crowded_short_squeeze_lb{lb}_c{comp:g}_oi{oilb}_{oi_th:g}'
                    out[(name_l,'crowded_short_squeeze',1)] = score_l.to_numpy()

        # Basis dislocation resolving with breakout; no compression requirement.
        for z in (1.0,1.5,2.0):
            eligible_s = bear & (close < lo).to_numpy() & (basis_z >= z).to_numpy() & (oi6 > 0.02).to_numpy() & (taker < 1).to_numpy()
            out[(f'positive_basis_break_lb{lb}_z{z:g}','basis_break',-1)] = (basis_z + oi6*10).where(eligible_s).to_numpy()
            eligible_l = bull & (close > hi).to_numpy() & (basis_z <= -z).to_numpy() & (oi6 > 0.02).to_numpy() & (taker > 1).to_numpy()
            out[(f'negative_basis_squeeze_lb{lb}_z{z:g}','basis_squeeze',1)] = ((-basis_z) + oi6*10).where(eligible_l).to_numpy()
    return out


def simulate(cfg: Config, score: np.ndarray, idx: pd.DatetimeIndex, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, funding: np.ndarray):
    rec=[]; active=-1; entry=stop=target=funding_cost=0.0; entry_i=bars=0
    n,_=score.shape
    for i in range(1,n):
        if active>=0:
            funding_cost += cfg.side*float(funding[i,active])
            sh = low[i,active] <= stop if cfg.side==1 else high[i,active] >= stop
            th = high[i,active] >= target if cfg.side==1 else low[i,active] <= target
            if sh: xp,reason=stop,'stop'
            elif th: xp,reason=target,'target'
            else:
                bars+=1
                if bars<cfg.hold or np.isnan(close[i,active]): continue
                xp,reason=close[i,active],'time'
            gross=cfg.side*(xp/entry-1)
            rec.append({'symbol':SYMBOLS[active],'entry_time':idx[entry_i],'exit_time':idx[i],'gross_return':gross,'funding_cost':funding_cost,'net_return':gross-2*ONE_WAY_COST-funding_cost,'reason':reason})
            active=-1; continue
        prev=score[i-1]
        if np.all(np.isnan(prev)): continue
        j=int(np.nanargmax(prev))
        if np.isnan(open_[i,j]) or open_[i,j]<=0: continue
        active=j; entry_i=i; bars=0; funding_cost=0.0; entry=float(open_[i,j])
        stop=entry*(1-cfg.sl) if cfg.side==1 else entry*(1+cfg.sl)
        target=entry*(1+cfg.tp) if cfg.side==1 else entry*(1-cfg.tp)
    return pd.DataFrame(rec)


def bootstrap5(x: pd.Series):
    if len(x)<20: return float('nan')
    a=x.to_numpy(float); rng=np.random.default_rng(20260807)
    means=rng.choice(a,size=(2000,len(a)),replace=True).mean(1)
    return float(np.quantile(means,0.05))


def stats(t: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, extra_cost=0.0):
    if t.empty: return {'trades':0,'cagr':-1.0,'avg_trade':0.0,'profit_factor':0.0,'max_drawdown':0.0,'win_rate':0.0,'bootstrap5':float('nan'),'positive_year_pct':0.0}
    r=t['net_return'].astype(float)-extra_cost
    eq=(1+r).cumprod(); years=max((end-start).total_seconds()/(365.25*86400),0.01)
    cagr=float(eq.iloc[-1]**(1/years)-1) if eq.iloc[-1]>0 else -1.0
    wins=r[r>0]; losses=r[r<0]
    yr=pd.DataFrame({'year':pd.to_datetime(t['entry_time'],utc=True).dt.year,'r':r}).groupby('year')['r'].apply(lambda z:(1+z).prod()-1)
    return {
        'trades':int(len(r)),'cagr':cagr,'total_return':float(eq.iloc[-1]-1),'avg_trade':float(r.mean()),
        'profit_factor':float(wins.sum()/-losses.sum()) if len(losses) else float('inf'),
        'max_drawdown':float((eq/eq.cummax()-1).min()),'win_rate':float((r>0).mean()),
        'median_win':float(wins.median()) if len(wins) else 0.0,'avg_win':float(wins.mean()) if len(wins) else 0.0,
        'avg_loss':float(losses.mean()) if len(losses) else 0.0,'bootstrap5':bootstrap5(r),
        'positive_year_pct':float((yr>0).mean()) if len(yr) else 0.0,
        'target_hit_rate':float((t['reason']=='target').mean())
    }


def train_score(r):
    if r['trades']<35 or r['cagr']<0.08 or r['profit_factor']<1.10 or r['max_drawdown']<-0.40: return -1e9
    return 2*math.log(max(r['profit_factor'],1e-9)) + 2*r['cagr'] + 200*r['avg_trade'] + r['max_drawdown']


def main():
    OUT.mkdir(exist_ok=True)
    data=load_data(); idx=data['BTCUSDT'].index.sort_values(); smap=signal_map(data,idx)
    open_=aligned(data,'open',idx).to_numpy(float); high=aligned(data,'high',idx).to_numpy(float); low=aligned(data,'low',idx).to_numpy(float); close=aligned(data,'close',idx).to_numpy(float); funding=aligned(data,'funding_event',idx).fillna(0).to_numpy(float)
    rows=[]
    train_start=max(idx.min(),pd.Timestamp('2023-01-01',tz='UTC')); test_end=idx.max()
    for (sig,fam,side),score in smap.items():
        for tp in (0.025,0.03,0.04):
            for sl in (0.012,0.018,0.025):
                for hold in (6,12,24):
                    cfg=Config(fam,sig,side,tp,sl,hold); trades=simulate(cfg,score,idx,open_,high,low,close,funding)
                    if trades.empty: continue
                    trades['entry_time']=pd.to_datetime(trades['entry_time'],utc=True); trades['exit_time']=pd.to_datetime(trades['exit_time'],utc=True)
                    tr=trades[trades.entry_time<=TRAIN_END]; te=trades[trades.entry_time>=TEST_START]
                    a=stats(tr,train_start,TRAIN_END); b=stats(te,TEST_START,test_end); stress=stats(te,TEST_START,test_end,extra_cost=ONE_WAY_COST)
                    rows.append({'name':cfg.name,'signal':sig,'family':fam,'side':side,'tp':tp,'sl':sl,'hold':hold,'split':'train',**a})
                    rows.append({'name':cfg.name,'signal':sig,'family':fam,'side':side,'tp':tp,'sl':sl,'hold':hold,'split':'test',**b,'stress_cagr':stress['cagr'],'stress_pf':stress['profit_factor'],'stress_avg_trade':stress['avg_trade']})
    df=pd.DataFrame(rows); df.to_csv(OUT/'all_results.csv',index=False)
    selected=[]
    for fam in sorted(df.family.unique()):
        tr=df[(df.family==fam)&(df.split=='train')].copy(); tr['score']=tr.apply(train_score,axis=1); best=tr.sort_values('score',ascending=False).iloc[0]
        if best.score<=-1e8: continue
        row=df[(df.name==best['name'])&(df.split=='test')].iloc[0].copy()
        row['pass8']=bool(row.trades>=40 and row.cagr>=0.08 and row.profit_factor>=1.25 and row.max_drawdown>=-0.25 and row.bootstrap5>0 and row.positive_year_pct>=1.0 and row.stress_cagr>=0.08 and row.stress_pf>=1.10)
        selected.append(row)
    sel=pd.DataFrame(selected); sel.to_csv(OUT/'selected_oos.csv',index=False)
    report=['# Wave 7 — derivatives crowding/OI edge','',f'- Data: {idx.min()} to {idx.max()}','- Train-only selection: 2023–2024; strict OOS: 2025–2026-07.','- Binance USD-M 4h; official kline/funding plus 5m futures metrics aggregated to 4h.','- One position at a time; next-bar-open entry; stop-first on ambiguous bars.','- Base one-way cost 0.08%; stress adds another 0.08% round trip-equivalent per trade.','- Reporting floor: CAGR >= 8%.','']
    if not sel.empty:
        cols=['name','family','trades','cagr','profit_factor','max_drawdown','win_rate','avg_trade','bootstrap5','positive_year_pct','stress_cagr','stress_pf','pass8']
        shown=sel[sel.cagr>=0.08].sort_values(['pass8','cagr'],ascending=[False,False])
        report += ['## Train-selected OOS models with CAGR >= 8%','', shown[cols].to_markdown(index=False,floatfmt='.4f') if not shown.empty else 'None.','']
        report += [f'PASS count: **{int(sel.pass8.sum())}**.']
    else: report += ['No family survived train selection.']
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    print('\n'.join(report))


if __name__=='__main__':
    main()
