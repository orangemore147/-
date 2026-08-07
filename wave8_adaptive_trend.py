from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path('results_wave8')
SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','LINKUSDT','LTCUSDT','AVAXUSDT']
START_MONTH = '2022-01'
END_MONTH = '2026-07'
TRAIN_END = pd.Timestamp('2024-12-31 23:59:59', tz='UTC')
TEST_START = pd.Timestamp('2025-01-01', tz='UTC')
ONE_WAY_COST = 0.0008
BARS_YEAR = 4 * 365.25


@dataclass(frozen=True)
class Config:
    lookback: int
    threshold: float
    atr_mult: float
    target_vol: float
    cap: float

    @property
    def name(self):
        return f'adaptive_L{self.lookback}_th{self.threshold:g}_atr{self.atr_mult:g}_tv{self.target_vol:g}_cap{self.cap:g}'


def load_data() -> Dict[str,pd.DataFrame]:
    w3.START_MONTH = START_MONTH
    w3.END_MONTH = END_MONTH
    w3.CACHE = Path('.cache_wave8')
    out={}
    for s in SYMBOLS:
        print('Loading',s)
        bars=w3.load_kline(s,'perp').resample('6h',origin='start_day').agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum'}
        ).dropna()
        fund=w3.load_funding(s).resample('6h',origin='start_day').sum().reindex(bars.index).fillna(0.0)
        bars['funding']=fund
        prev=bars['close'].shift(1)
        tr=pd.concat([bars.high-bars.low,(bars.high-prev).abs(),(bars.low-prev).abs()],axis=1).max(axis=1)
        bars['atr14']=tr.rolling(14).mean()
        out[s]=bars
    start=max(x.index.min() for x in out.values()); end=min(x.index.max() for x in out.values())
    return {s:x.loc[start:end].copy() for s,x in out.items()}


def aligned(data,field,idx):
    return pd.DataFrame({s:data[s][field].reindex(idx) for s in SYMBOLS},index=idx)


def raw_positions(data: Dict[str,pd.DataFrame], idx: pd.DatetimeIndex, lookback:int, threshold:float, atr_mult:float):
    close=aligned(data,'close',idx); high=aligned(data,'high',idx); low=aligned(data,'low',idx); atr=aligned(data,'atr14',idx)
    mom=close.pct_change(lookback)
    state=np.zeros((len(idx),len(SYMBOLS)),dtype=np.int8)
    stop=np.full(len(SYMBOLS),np.nan)
    for i in range(1,len(idx)):
        for j in range(len(SYMBOLS)):
            prev_state=state[i-1,j]
            c=float(close.iat[i,j]) if pd.notna(close.iat[i,j]) else np.nan
            a=float(atr.iat[i,j]) if pd.notna(atr.iat[i,j]) else np.nan
            if not np.isfinite(c) or not np.isfinite(a):
                state[i,j]=0; stop[j]=np.nan; continue
            if prev_state==1:
                stop[j]=max(stop[j], c-atr_mult*a)
                if low.iat[i,j] <= stop[j]:
                    state[i,j]=0; stop[j]=np.nan
                else:
                    state[i,j]=1
            elif prev_state==-1:
                stop[j]=min(stop[j], c+atr_mult*a)
                if high.iat[i,j] >= stop[j]:
                    state[i,j]=0; stop[j]=np.nan
                else:
                    state[i,j]=-1
            else:
                m=mom.iat[i,j]
                if pd.isna(m):
                    state[i,j]=0
                elif m>threshold:
                    state[i,j]=1; stop[j]=c-atr_mult*a
                elif m<-threshold:
                    state[i,j]=-1; stop[j]=c+atr_mult*a
                else:
                    state[i,j]=0
    pos=pd.DataFrame(state,index=idx,columns=SYMBOLS,dtype=float)
    # Signals decided after bar close, effective next bar.
    return pos.shift(1).fillna(0.0)


def normalize_70_30(pos: pd.DataFrame):
    w=pd.DataFrame(0.0,index=pos.index,columns=pos.columns)
    longs=pos.gt(0); shorts=pos.lt(0)
    nl=longs.sum(axis=1).replace(0,np.nan); ns=shorts.sum(axis=1).replace(0,np.nan)
    w=w.mask(longs, (0.70/nl).to_numpy()[:,None])
    w=w.mask(shorts, (-0.30/ns).to_numpy()[:,None])
    return w.fillna(0.0)


def portfolio(data,idx,cfg:Config,extra_cost=0.0):
    close=aligned(data,'close',idx); funding=aligned(data,'funding',idx).fillna(0.0)
    ret=close.pct_change().fillna(0.0)
    base=normalize_70_30(raw_positions(data,idx,cfg.lookback,cfg.threshold,cfg.atr_mult))
    gross=(base*ret).sum(axis=1)
    # Risk scale uses only prior realized strategy volatility.
    rv=gross.rolling(120,min_periods=40).std()*math.sqrt(BARS_YEAR)
    lev=(cfg.target_vol/rv.replace(0,np.nan)).clip(0,cfg.cap).shift(1).fillna(0.0)
    w=base.mul(lev,axis=0)
    gross=(w*ret).sum(axis=1)
    turnover=w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    funding_cost=(w*funding).sum(axis=1)
    net=gross-turnover*(ONE_WAY_COST+extra_cost)-funding_cost
    return net.clip(lower=-0.95),w


def metrics(r:pd.Series,start:pd.Timestamp,end:pd.Timestamp):
    r=r.loc[start:end].dropna()
    if len(r)<200: return {}
    eq=(1+r).cumprod(); years=(end-start).total_seconds()/(365.25*86400)
    cagr=float(eq.iloc[-1]**(1/years)-1) if eq.iloc[-1]>0 else -1.0
    vol=float(r.std()*math.sqrt(BARS_YEAR)); sharpe=float(r.mean()*BARS_YEAR/vol) if vol>0 else np.nan
    dd=float((eq/eq.cummax()-1).min())
    monthly=(1+r).resample('ME').prod()-1
    yearly=(1+r).resample('YE').prod()-1
    return {'cagr':cagr,'sharpe':sharpe,'max_drawdown':dd,'avg_month':float(monthly.mean()),'worst_month':float(monthly.min()),'positive_month_pct':float((monthly>0).mean()),'month_ge_10_pct':float((monthly>=0.10).mean()),'positive_year_pct':float((yearly>0).mean()),'total_return':float(eq.iloc[-1]-1)}


def score(m):
    if not m or m['cagr']<0.08 or m['max_drawdown']<-0.45 or not np.isfinite(m['sharpe']): return -1e9
    return 2*m['sharpe']+m['cagr']+m['max_drawdown']


def main():
    OUT.mkdir(exist_ok=True)
    data=load_data(); idx=data['BTCUSDT'].index.sort_values(); train_start=max(idx.min(),pd.Timestamp('2022-01-01',tz='UTC')); test_end=idx.max()
    configs=[Config(lb,th,a,tv,cap) for lb in (20,40,80) for th in (0.03,0.06,0.10) for a in (2.0,2.5,3.0,3.5) for tv,cap in ((0.20,1.5),(0.30,2.0))]
    rows=[]; cache={}
    for cfg in configs:
        print('Testing',cfg.name)
        r,w=portfolio(data,idx,cfg); cache[cfg.name]=(r,w)
        tr=metrics(r,train_start,TRAIN_END); te=metrics(r,TEST_START,test_end)
        stress,_=portfolio(data,idx,cfg,extra_cost=ONE_WAY_COST); st=metrics(stress,TEST_START,test_end)
        rows.append({'name':cfg.name,'split':'train',**tr})
        rows.append({'name':cfg.name,'split':'test',**te,'stress_cagr':st.get('cagr',np.nan),'stress_sharpe':st.get('sharpe',np.nan),'stress_dd':st.get('max_drawdown',np.nan)})
    df=pd.DataFrame(rows); df.to_csv(OUT/'all_results.csv',index=False)
    tr=df[df.split=='train'].copy(); tr['score']=tr.apply(score,axis=1); best=tr.sort_values('score',ascending=False).iloc[0]
    selected=df[(df.name==best['name'])&(df.split=='test')].copy()
    selected['pass8']=selected.apply(lambda x: bool(x.cagr>=0.08 and x.sharpe>=1.0 and x.max_drawdown>=-0.30 and x.positive_year_pct>=1.0 and x.stress_cagr>=0.08 and x.stress_sharpe>=0.75),axis=1)
    selected.to_csv(OUT/'selected_oos.csv',index=False)
    top=df[(df.split=='test')&(df.cagr>=0.08)].sort_values('sharpe',ascending=False).head(15)
    cols=['name','cagr','sharpe','max_drawdown','avg_month','worst_month','positive_month_pct','month_ge_10_pct','positive_year_pct','stress_cagr','stress_sharpe']
    report=['# Wave 8 — AdaptiveTrend replication','',f'- Data: {idx.min()} to {idx.max()}','- Binance USD-M perpetual 6h; realized funding included.','- Base one-way cost 0.08%; cost-stress doubles execution cost.','- Parameter selection only on 2022–2024; OOS 2025–2026-07.','- Long/short gross allocation 70/30, ATR trailing exits, volatility scaling.','', '## Train-selected model — OOS','',selected[cols+['pass8']].to_markdown(index=False,floatfmt='.4f'),'','## OOS diagnostic models above 8% CAGR (not valid for selection)','',top[cols].to_markdown(index=False,floatfmt='.4f') if not top.empty else 'None.']
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8'); print('\n'.join(report))


if __name__=='__main__': main()
