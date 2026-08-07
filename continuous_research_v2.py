from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path("results_continuous_v2")
STATE = OUT / "state.json"
STATUS = OUT / "status.json"
DEV_BOARD = OUT / "dev_leaderboard.csv"
REPORT = OUT / "latest_report.md"
ALERT = OUT / "ALERT.json"

START_MONTH = "2023-01"
END_MONTH = "2026-07"
DEV_TRAIN_END = pd.Timestamp("2024-06-30 23:59:59", tz="UTC")
DEV_VAL_START = pd.Timestamp("2024-07-01", tz="UTC")
DEV_VAL_END = pd.Timestamp("2025-03-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-04-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-07-31 23:59:59", tz="UTC")
PRIMARY = ["BTCUSDT", "SOLUSDT"]
EXTERNAL = ["ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]
ONE_WAY_COST = 0.0008


@dataclass(frozen=True)
class Config:
    family: str
    symbol: str
    side: str
    lookback: int
    hold: int
    threshold: float
    aux: float
    stop: float
    target: float

    @property
    def key(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"next_batch": 0, "strict_passes": []}


def base_bars(symbol: str) -> pd.DataFrame:
    k = w3.load_kline(symbol, "perp")
    f = w3.load_funding(symbol)
    b = k.resample("4h", origin="start_day").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
    ).dropna()
    b["funding"] = f.resample("4h", origin="start_day").sum().reindex(b.index).fillna(0.0)
    prev = b["close"].shift(1)
    tr = pd.concat([b["high"]-b["low"], (b["high"]-prev).abs(), (b["low"]-prev).abs()], axis=1).max(axis=1)
    b["atr_pct"] = tr.rolling(21).mean() / b["close"]
    b["atr_rank"] = b["atr_pct"].rolling(126).rank(pct=True)
    b["ema50"] = b["close"].ewm(span=50, adjust=False).mean()
    b["ema200"] = b["close"].ewm(span=200, adjust=False).mean()
    b["ema_gap"] = b["ema50"] / b["ema200"] - 1.0
    b["vol_ratio"] = b["volume"] / b["volume"].rolling(42).median().replace(0, np.nan)
    fm = b["funding"].rolling(126).mean()
    fs = b["funding"].rolling(126).std().replace(0, np.nan)
    b["funding_z"] = (b["funding"] - fm) / fs
    b["funding_mean6"] = b["funding"].rolling(6).mean()
    for lb in [1, 3, 6, 12, 18, 30, 48]:
        b[f"ret{lb}"] = b["close"].pct_change(lb)
        if lb > 1:
            b[f"prior_high{lb}"] = b["high"].rolling(lb).max().shift(1)
            b[f"prior_low{lb}"] = b["low"].rolling(lb).min().shift(1)
    return b.loc[(b.index >= pd.Timestamp("2023-01-01", tz="UTC")) & (b.index <= HOLDOUT_END)].copy()


def add_btc_context(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    btc = data["BTCUSDT"]
    ctx = pd.DataFrame(index=btc.index)
    ctx["btc_ret6"] = btc["ret6"]
    ctx["btc_ret18"] = btc["ret18"]
    ctx["btc_up"] = btc["close"] > btc["ema200"]
    ctx["btc_down"] = btc["close"] < btc["ema200"]
    out = {}
    for s, b in data.items():
        x = b.copy()
        for c in ctx.columns:
            x[c] = ctx[c].reindex(x.index)
        x["rel_ret6"] = x["ret6"] - x["btc_ret6"]
        x["rel_ret18"] = x["ret18"] - x["btc_ret18"]
        out[s] = x
    return out


def random_configs(batch: int, symbol: str, n: int = 120) -> List[Config]:
    rng = np.random.default_rng(20260808 + batch*104729 + (0 if symbol == "BTCUSDT" else 997))
    fams = [
        "trend_pullback", "funding_reversal", "funding_persistence", "compression_break",
        "climax_reversal", "failed_breakout", "relative_strength", "btc_regime_reversal",
        "settlement_reversal", "weekday_momentum",
    ]
    out: List[Config] = []
    for _ in range(n):
        fam = str(rng.choice(fams))
        if symbol == "BTCUSDT" and fam in ("relative_strength", "btc_regime_reversal"):
            fam = str(rng.choice(["trend_pullback", "funding_reversal", "compression_break", "climax_reversal"]))
        lb = int(rng.choice([3,6,12,18,30,48]))
        hold = int(rng.choice([1,2,3,6,9,12]))
        side = str(rng.choice(["long","short","both"]))
        threshold = float(rng.choice([0.003,0.006,0.01,0.015,0.02,0.03,0.05]))
        if fam == "weekday_momentum":
            aux = float(rng.integers(0,7))
        elif fam == "settlement_reversal":
            aux = float(rng.choice([0.5,0.75,1.0,1.5,2.0]))
        else:
            aux = float(rng.choice([0.15,0.25,0.35,0.5,0.75,1.0,1.5,2.0]))
        stop = float(rng.choice([0.0,0.012,0.018,0.025,0.035]))
        target = float(rng.choice([0.0,0.018,0.03,0.045,0.06]))
        out.append(Config(fam,symbol,side,lb,hold,threshold,aux,stop,target))
    return dedupe(out)


def mutate_from_dev(symbol: str, batch: int, limit: int = 40) -> List[Config]:
    if not DEV_BOARD.exists():
        return []
    try:
        board = pd.read_csv(DEV_BOARD)
    except Exception:
        return []
    board = board[board["symbol"] == symbol].sort_values("dev_score", ascending=False).head(8)
    if board.empty:
        return []
    rng = np.random.default_rng(910000 + batch*7919 + (0 if symbol == "BTCUSDT" else 13))
    out: List[Config] = []
    lbs = np.array([3,6,12,18,30,48])
    holds = np.array([1,2,3,6,9,12])
    ths = np.array([0.003,0.006,0.01,0.015,0.02,0.03,0.05])
    for _, r in board.iterrows():
        base = Config(str(r.family), symbol, str(r.side), int(r.lookback), int(r.hold), float(r.threshold), float(r.aux), float(r.stop), float(r.target))
        for _ in range(5):
            lb = int(rng.choice(lbs[np.argsort(abs(lbs-base.lookback))[:3]]))
            hold = int(rng.choice(holds[np.argsort(abs(holds-base.hold))[:3]]))
            th = float(rng.choice(ths[np.argsort(abs(ths-base.threshold))[:3]]))
            aux = base.aux
            if base.family not in ("weekday_momentum",):
                aux = float(max(0.15, base.aux * rng.choice([0.75,1.0,1.25])))
            out.append(Config(base.family,symbol,base.side,lb,hold,th,aux,base.stop,base.target))
    return dedupe(out)[:limit]


def dedupe(xs: List[Config]) -> List[Config]:
    seen=set(); out=[]
    for x in xs:
        if x.key not in seen:
            seen.add(x.key); out.append(x)
    return out


def signal(b: pd.DataFrame, c: Config) -> pd.Series:
    lb=c.lookback; ret=b[f"ret{lb}"]
    ph=b[f"prior_high{lb}"]; pl=b[f"prior_low{lb}"]
    up=b["close"]>b["ema200"]; down=b["close"]<b["ema200"]
    long=pd.Series(False,index=b.index); short=long.copy()
    if c.family=="trend_pullback":
        long=up & (ret<=-c.threshold) & (b["close"]>b["ema50"]) & (b["ret1"]>0)
        short=down & (ret>=c.threshold) & (b["close"]<b["ema50"]) & (b["ret1"]<0)
    elif c.family=="funding_reversal":
        long=(b["funding_z"]<=-c.aux) & (ret<=-c.threshold)
        short=(b["funding_z"]>=c.aux) & (ret>=c.threshold)
    elif c.family=="funding_persistence":
        long=up & (b["funding_z"]>=c.aux) & (ret>=c.threshold)
        short=down & (b["funding_z"]<=-c.aux) & (ret<=-c.threshold)
    elif c.family=="compression_break":
        comp=b["atr_rank"]<=min(max(c.aux,0.15),0.75)
        long=comp & up & (b["close"]>ph)
        short=comp & down & (b["close"]<pl)
    elif c.family=="climax_reversal":
        hv=b["atr_rank"]>=0.75; vv=b["vol_ratio"]>=max(c.aux,1.0)
        long=hv & vv & (ret<=-c.threshold)
        short=hv & vv & (ret>=c.threshold)
    elif c.family=="failed_breakout":
        prev_hi=b["high"].shift(1)>ph.shift(1); prev_lo=b["low"].shift(1)<pl.shift(1)
        long=prev_lo & (b["close"]>pl) & (b["ret1"]>0)
        short=prev_hi & (b["close"]<ph) & (b["ret1"]<0)
    elif c.family=="relative_strength":
        rr=b["rel_ret18"] if lb>=18 else b["rel_ret6"]
        long=b["btc_up"].fillna(False) & (rr>=c.threshold)
        short=b["btc_down"].fillna(False) & (rr<=-c.threshold)
    elif c.family=="btc_regime_reversal":
        long=b["btc_up"].fillna(False) & (ret<=-c.threshold) & (b["funding_z"]<=-c.aux)
        short=b["btc_down"].fillna(False) & (ret>=c.threshold) & (b["funding_z"]>=c.aux)
    elif c.family=="settlement_reversal":
        settle=b.index.hour.isin([0,8,16])
        long=pd.Series(settle,index=b.index) & (b["funding_z"]<=-c.aux) & (ret<0)
        short=pd.Series(settle,index=b.index) & (b["funding_z"]>=c.aux) & (ret>0)
    elif c.family=="weekday_momentum":
        day=(b.index.dayofweek==int(c.aux)%7)
        long=pd.Series(day,index=b.index) & up & (ret>=c.threshold)
        short=pd.Series(day,index=b.index) & down & (ret<=-c.threshold)
    s=pd.Series(0,index=b.index,dtype=int)
    if c.side in ("long","both"): s.loc[long.fillna(False)]=1
    if c.side in ("short","both"): s.loc[short.fillna(False)]=-1
    return s


def simulate(b: pd.DataFrame, c: Config) -> pd.DataFrame:
    sg=signal(b,c).to_numpy(int); idx=b.index
    op=b.open.to_numpy(float); hi=b.high.to_numpy(float); lo=b.low.to_numpy(float); funding=b.funding.to_numpy(float)
    rec=[]; i=0
    while i < len(b)-c.hold-2:
        if sg[i]==0: i+=1; continue
        side=int(sg[i]); ei=i+1; entry=float(op[ei]); planned=min(ei+c.hold,len(b)-1)
        xi=planned; xp=float(op[planned]); reason="time"
        sp=entry*(1-c.stop if side>0 else 1+c.stop) if c.stop>0 else None
        tp=entry*(1+c.target if side>0 else 1-c.target) if c.target>0 else None
        for j in range(ei,planned+1):
            sh=(lo[j]<=sp if side>0 else hi[j]>=sp) if sp is not None else False
            th=(hi[j]>=tp if side>0 else lo[j]<=tp) if tp is not None else False
            if sh: xi=j; xp=float(sp); reason="stop"; break
            if th: xi=j; xp=float(tp); reason="target"; break
        gross=side*(xp/entry-1.0)
        fpnl=-side*float(np.nansum(funding[ei:xi+1]))
        rec.append({"entry_time":idx[ei],"exit_time":idx[xi],"net":gross+fpnl-2*ONE_WAY_COST,"gross":gross,"funding_pnl":fpnl,"side":side,"reason":reason})
        i=max(xi,i+1)
    return pd.DataFrame(rec)


def bootstrap5(r: np.ndarray, seed: int) -> float:
    if len(r)<20: return float("nan")
    rng=np.random.default_rng(seed); means=rng.choice(r,size=(3000,len(r)),replace=True).mean(axis=1)
    return float(np.quantile(means,0.05))


def metrics(t: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, extra_cost: float=0.0, seed: int=1) -> dict:
    if t.empty: return {}
    z=t[(t.entry_time>=start)&(t.entry_time<=end)].copy()
    if z.empty: return {}
    r=z.net.to_numpy(float)-extra_cost; eq=np.cumprod(1+r)
    years=max((end-start).total_seconds()/(365.25*86400),0.05)
    cagr=float(eq[-1]**(1/years)-1) if eq[-1]>0 else -1.0
    peak=np.maximum.accumulate(eq); dd=float(np.min(eq/peak-1))
    w=r[r>0]; l=r[r<0]; pf=float(w.sum()/-l.sum()) if len(l) else float("inf")
    return {"trades":int(len(r)),"cagr":cagr,"total_return":float(eq[-1]-1),"avg_trade":float(r.mean()),"win_rate":float((r>0).mean()),"profit_factor":pf,"max_drawdown":dd,"bootstrap_mean_5pct":bootstrap5(r,seed)}


def development_row(c: Config, t: pd.DataFrame, seed: int) -> dict:
    tr=metrics(t,pd.Timestamp("2023-01-01",tz="UTC"),DEV_TRAIN_END,seed=seed)
    va=metrics(t,DEV_VAL_START,DEV_VAL_END,seed=seed+1)
    row={**asdict(c),"key":c.key}
    for p,d in (("train",tr),("val",va)):
        for k,v in d.items(): row[f"{p}_{k}"]=v
    row["dev_gate"]=dev_gate(row)
    row["dev_score"]=dev_score(row)
    return row


def dev_gate(r: dict) -> bool:
    return bool(r.get("train_trades",0)>=30 and r.get("val_trades",0)>=20 and r.get("train_avg_trade",-1)>0 and r.get("val_avg_trade",-1)>0 and r.get("train_profit_factor",0)>1.02 and r.get("val_profit_factor",0)>1.10 and r.get("train_max_drawdown",-1)>=-0.40 and r.get("val_max_drawdown",-1)>=-0.35)


def dev_score(r: dict) -> float:
    if not dev_gate(r): return -999.0
    pf=min(float(r.get("train_profit_factor",0)),float(r.get("val_profit_factor",0)))
    cagr=min(float(r.get("train_cagr",-1)),float(r.get("val_cagr",-1)))
    dd=max(abs(float(r.get("train_max_drawdown",-1))),abs(float(r.get("val_max_drawdown",-1))))
    return pf + 0.35*np.clip(cagr,-1,2) - 0.25*dd


def holdout_row(c: Config, t: pd.DataFrame, seed: int) -> dict:
    te=metrics(t,HOLDOUT_START,HOLDOUT_END,seed=seed)
    st=metrics(t,HOLDOUT_START,HOLDOUT_END,extra_cost=2*ONE_WAY_COST,seed=seed+1)
    row={"key":c.key}
    for p,d in (("test",te),("stress",st)):
        for k,v in d.items(): row[f"{p}_{k}"]=v
    row["holdout_gate"]=holdout_gate(row)
    return row


def holdout_gate(r: dict) -> bool:
    return bool(r.get("test_trades",0)>=20 and r.get("test_cagr",-1)>=0.08 and r.get("test_avg_trade",-1)>0 and r.get("test_profit_factor",0)>=1.20 and r.get("test_max_drawdown",-1)>=-0.25 and r.get("test_bootstrap_mean_5pct",-1)>0 and r.get("stress_cagr",-1)>=0.08 and r.get("stress_profit_factor",0)>=1.05)


def external_check(c: Config, ext_data: Dict[str,pd.DataFrame], seed: int) -> dict:
    rows=[]
    for i,(s,b) in enumerate(ext_data.items()):
        cc=Config(c.family,s,c.side,c.lookback,c.hold,c.threshold,c.aux,c.stop,c.target)
        m=metrics(simulate(b,cc),HOLDOUT_START,HOLDOUT_END,seed=seed+i*17)
        st=metrics(simulate(b,cc),HOLDOUT_START,HOLDOUT_END,extra_cost=2*ONE_WAY_COST,seed=seed+i*17+1)
        rows.append({"symbol":s,**m,**{f"stress_{k}":v for k,v in st.items()}})
    df=pd.DataFrame(rows); valid=df[df.trades.fillna(0)>=10].copy() if not df.empty and "trades" in df else pd.DataFrame()
    pos=int((valid.avg_trade>0).sum()) if not valid.empty else 0
    boot=int((valid.bootstrap_mean_5pct>0).sum()) if not valid.empty else 0
    stress=int((valid.stress_avg_trade>0).sum()) if not valid.empty else 0
    medpf=float(valid.profit_factor.median()) if not valid.empty else float("nan")
    passed=bool(len(valid)>=3 and pos>=3 and boot>=2 and stress>=3 and medpf>1.05)
    return {"passed":passed,"valid_symbols":int(len(valid)),"positive_symbols":pos,"bootstrap_positive_symbols":boot,"stress_positive_symbols":stress,"median_pf":medpf,"details":rows}


def merge_dev(df: pd.DataFrame) -> pd.DataFrame:
    if DEV_BOARD.exists():
        try: old=pd.read_csv(DEV_BOARD); df=pd.concat([old,df],ignore_index=True)
        except Exception: pass
    df=df.drop_duplicates("key",keep="last").sort_values("dev_score",ascending=False).head(500)
    df.to_csv(DEV_BOARD,index=False); return df


def main() -> None:
    OUT.mkdir(exist_ok=True); st=load_state(); batch=int(st.get("next_batch",0)); started=pd.Timestamp.now(tz="UTC")
    save_json(STATUS,{"status":"running","batch":batch,"started_at":started.isoformat(),"stage":"loading"})
    w3.START_MONTH=START_MONTH; w3.END_MONTH=END_MONTH; w3.INTERVAL="1h"; w3.CACHE=Path(".cache_continuous_v2")
    raw={s:base_bars(s) for s in PRIMARY}; data=add_btc_context(raw)
    rows=[]; cfg_map={}; tested=0
    for s,b in data.items():
        cfgs=dedupe(random_configs(batch,s)+mutate_from_dev(s,batch))
        for j,c in enumerate(cfgs):
            t=simulate(b,c); r=development_row(c,t,20260808+batch*10000+j); r["batch"]=batch
            rows.append(r); cfg_map[c.key]=(c,t); tested+=1
    dev=pd.DataFrame(rows); board=merge_dev(dev)
    survivors=dev[dev.dev_gate==True].sort_values("dev_score",ascending=False).head(12)
    hold_rows=[]; prelim=[]
    for k,(_,r) in enumerate(survivors.iterrows()):
        c,t=cfg_map[r.key]; h=holdout_row(c,t,880000+batch*100+k); hold_rows.append({**r.to_dict(),**h})
        if h["holdout_gate"]: prelim.append((c,{**r.to_dict(),**h}))
    pd.DataFrame(hold_rows).to_csv(OUT/f"holdout_batch_{batch:05d}.csv",index=False)
    dev.to_csv(OUT/f"dev_batch_{batch:05d}.csv",index=False)
    ext_checked=0; passes=[]
    if prelim:
        save_json(STATUS,{"status":"running","batch":batch,"started_at":started.isoformat(),"stage":"external_confirmation","tested":tested,"holdout_candidates":len(prelim)})
        eraw={"BTCUSDT":raw["BTCUSDT"]}
        for s in EXTERNAL: eraw[s]=base_bars(s)
        ext_data=add_btc_context(eraw); ext_data={s:ext_data[s] for s in EXTERNAL}
        for i,(c,row) in enumerate(prelim[:3]):
            ext_checked+=1; ext=external_check(c,ext_data,770000+batch*100+i)
            if ext["passed"]:
                passes.append({"detected_at":pd.Timestamp.now(tz="UTC").isoformat(),"batch":batch,"config":asdict(c),"development":{"train_pf":row.get("train_profit_factor"),"val_pf":row.get("val_profit_factor"),"train_cagr":row.get("train_cagr"),"val_cagr":row.get("val_cagr")},"holdout":{k.replace("test_",""):v for k,v in row.items() if str(k).startswith("test_")},"stress":{k.replace("stress_",""):v for k,v in row.items() if str(k).startswith("stress_")},"external":ext})
    if passes:
        save_json(ALERT,{"strict_passes":passes}); st.setdefault("strict_passes",[]).extend(passes); st["strict_passes"]=st["strict_passes"][-20:]
    st["next_batch"]=batch+1; st["last_completed_batch"]=batch; st["last_completed_at"]=pd.Timestamp.now(tz="UTC").isoformat(); save_json(STATE,st)
    top=board.head(10)
    cols=["key","family","symbol","side","lookback","hold","threshold","aux","stop","target","train_trades","train_profit_factor","train_cagr","val_trades","val_profit_factor","val_cagr","train_max_drawdown","val_max_drawdown","dev_score"]
    cols=[c for c in cols if c in top]
    rep="\n".join(["# Continuous crypto research v2","",f"- Batch: {batch}",f"- Completed: {pd.Timestamp.now(tz='UTC').isoformat()}",f"- Tested this batch: {tested}",f"- Development-gate survivors: {int(dev.dev_gate.sum())}",f"- Holdout candidates evaluated: {len(survivors)}",f"- Holdout strict candidates: {len(prelim)}",f"- External confirmations checked: {ext_checked}",f"- Strict passes: {len(passes)}","","## Development leaderboard (holdout intentionally hidden)","",top[cols].to_markdown(index=False,floatfmt=".4f")])
    REPORT.write_text(rep,encoding="utf-8")
    save_json(STATUS,{"status":"completed","batch":batch,"started_at":started.isoformat(),"completed_at":pd.Timestamp.now(tz="UTC").isoformat(),"tested":tested,"development_survivors":int(dev.dev_gate.sum()),"holdout_evaluated":len(survivors),"holdout_strict_candidates":len(prelim),"external_checked":ext_checked,"strict_passes":len(passes),"alert_file":ALERT.exists()})
    print(rep)


if __name__=="__main__":
    main()
