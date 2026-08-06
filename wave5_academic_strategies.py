from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests

OUT = Path("results_wave5")
CACHE = Path(".cache_wave5")
START = "2014-01-01"
END = "2026-08-06"
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01", tz="UTC")
ONE_WAY_COST = 0.0008
TRADING_DAYS = 252

STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "INTC",
    "ORCL", "IBM", "CSCO", "PEP", "MCD", "GE", "MA", "BABA", "LLY",
    "UNH", "ASML",
]
ALL = STOCKS + ["SPY"]


@dataclass(frozen=True)
class Candidate:
    family: str
    params: Tuple

    @property
    def name(self) -> str:
        return self.family + "_" + "-".join(str(x) for x in self.params)


def yahoo_url(ticker: str) -> str:
    p1 = int(pd.Timestamp(START, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=2)).timestamp())
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits"
    )


def load_ticker(ticker: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{ticker}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        return df

    last_error = None
    for attempt in range(5):
        try:
            r = requests.get(yahoo_url(ticker), timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            payload = r.json()["chart"]["result"][0]
            ts = pd.to_datetime(payload["timestamp"], unit="s", utc=True).normalize()
            q = payload["indicators"]["quote"][0]
            adj = payload["indicators"].get("adjclose", [{}])[0].get("adjclose")
            close = adj if adj is not None else q["close"]
            df = pd.DataFrame(
                {
                    "open": q["open"], "high": q["high"], "low": q["low"],
                    "close": close, "volume": q["volume"],
                }, index=ts,
            ).dropna(subset=["close"])
            df.index.name = "time"
            df.reset_index().to_csv(path, index=False)
            return df
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{ticker}: {last_error}")


def load_all() -> Dict[str, pd.DataFrame]:
    return {t: load_ticker(t) for t in ALL}


def master_index(data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    return data["SPY"].index.sort_values()


def aligned_close(data: Dict[str, pd.DataFrame], idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({t: data[t]["close"].reindex(idx) for t in STOCKS}, index=idx)


def aligned_ohlc(data: Dict[str, pd.DataFrame], idx: pd.DatetimeIndex, field: str) -> pd.DataFrame:
    return pd.DataFrame({t: data[t][field].reindex(idx) for t in STOCKS}, index=idx)


def rebalance_mask(n: int, every: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[np.arange(n) % every == 0] = True
    return mask


def normalize_rows(w: pd.DataFrame) -> pd.DataFrame:
    denom = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(denom, axis=0).fillna(0.0)


def invvol_weights(close: pd.DataFrame, spy: pd.Series, sma: int, vol_lb: int, reb: int) -> pd.DataFrame:
    ret = close.pct_change()
    vol = ret.rolling(vol_lb).std() * math.sqrt(TRADING_DAYS)
    trend = close > close.rolling(sma).mean()
    regime = spy > spy.rolling(200).mean()
    raw = (1 / vol.replace(0, np.nan)).where(trend, 0.0)
    raw = normalize_rows(raw).where(regime, 0.0)
    mask = rebalance_mask(len(close), reb)
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    out.loc[mask] = raw.loc[mask]
    return out.ffill().fillna(0.0).shift(1).fillna(0.0)


def dualema_weights(close: pd.DataFrame, spy: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    active = (ef > es).astype(float)
    active = normalize_rows(active)
    active = active.where(spy > spy.rolling(200).mean(), 0.0)
    return active.shift(1).fillna(0.0)


def topmom_weights(close: pd.DataFrame, spy: pd.Series, lookback: int, topn: int, reb: int) -> pd.DataFrame:
    mom = close.pct_change(lookback)
    trend = close > close.rolling(200).mean()
    mask = rebalance_mask(len(close), reb)
    raw = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for t in close.index[mask]:
        if pd.isna(spy.loc[t]) or pd.isna(spy.rolling(200).mean().loc[t]) or spy.loc[t] <= spy.rolling(200).mean().loc[t]:
            raw.loc[t] = 0.0
            continue
        row = mom.loc[t].where(trend.loc[t]).dropna().sort_values(ascending=False)
        chosen = row.head(topn).index
        raw.loc[t] = 0.0
        if len(chosen):
            raw.loc[t, chosen] = 1.0 / len(chosen)
    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def lowvol_mom_weights(close: pd.DataFrame, spy: pd.Series, mom_lb: int, topn: int, reb: int) -> pd.DataFrame:
    mom = close.pct_change(mom_lb)
    vol = close.pct_change().rolling(63).std() * math.sqrt(TRADING_DAYS)
    trend = close > close.rolling(200).mean()
    mask = rebalance_mask(len(close), reb)
    raw = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    spy_ma = spy.rolling(200).mean()
    for t in close.index[mask]:
        raw.loc[t] = 0.0
        if pd.isna(spy_ma.loc[t]) or spy.loc[t] <= spy_ma.loc[t]:
            continue
        valid = trend.loc[t] & (mom.loc[t] > 0) & vol.loc[t].notna()
        if valid.sum() == 0:
            continue
        mom_rank = mom.loc[t, valid].rank(pct=True)
        lowvol_rank = (-vol.loc[t, valid]).rank(pct=True)
        score = 0.6 * mom_rank + 0.4 * lowvol_rank
        chosen = score.nlargest(topn).index
        inv = 1 / vol.loc[t, chosen]
        raw.loc[t, chosen] = inv / inv.sum()
    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def atr_trend_weights(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, spy: pd.Series, breakout: int, atr_mult: float, topn: int, reb: int) -> pd.DataFrame:
    prev = close.shift(1)
    tr = pd.DataFrame(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - prev).abs().to_numpy(),
            (low - prev).abs().to_numpy(),
        ]), index=close.index, columns=close.columns,
    )
    atr = tr.rolling(20).mean()
    entry = close >= high.rolling(breakout).max().shift(1)
    active = pd.DataFrame(False, index=close.index, columns=close.columns)
    stop = pd.Series(np.nan, index=close.columns, dtype=float)
    state = pd.Series(False, index=close.columns)
    for t in close.index:
        for s in close.columns:
            c = close.at[t, s]
            a = atr.at[t, s]
            if pd.isna(c) or pd.isna(a):
                continue
            if not state[s] and bool(entry.at[t, s]):
                state[s] = True
                stop[s] = c - atr_mult * a
            elif state[s]:
                stop[s] = max(stop[s], c - atr_mult * a)
                if c < stop[s]:
                    state[s] = False
                    stop[s] = np.nan
            active.at[t, s] = state[s]
    mom = close.pct_change(126)
    mask = rebalance_mask(len(close), reb)
    raw = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    spy_ma = spy.rolling(200).mean()
    for t in close.index[mask]:
        raw.loc[t] = 0.0
        if pd.isna(spy_ma.loc[t]) or spy.loc[t] <= spy_ma.loc[t]:
            continue
        candidates = mom.loc[t].where(active.loc[t]).dropna().nlargest(topn).index
        if len(candidates):
            raw.loc[t, candidates] = 1.0 / len(candidates)
    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def portfolio_returns(close: pd.DataFrame, weights: pd.DataFrame, funding_drag: float = 0.0) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    gross = (weights * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    exposure = weights.abs().sum(axis=1)
    return gross - turnover * ONE_WAY_COST - exposure * funding_drag / TRADING_DAYS


def vol_scale_returns(raw: pd.Series, target: float, cap: float, funding_drag: float = 0.0) -> pd.Series:
    rv = raw.rolling(21).std() * math.sqrt(TRADING_DAYS)
    lev = (target / rv.replace(0, np.nan)).clip(0, cap).shift(1).fillna(0.0)
    return lev * raw - lev * funding_drag / TRADING_DAYS


def metrics(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if len(r) < 252:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * math.sqrt(TRADING_DAYS)
    sharpe = r.mean() * TRADING_DAYS / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    monthly = (1 + r).resample("ME").prod() - 1
    return {
        "cagr": float(cagr), "sharpe": float(sharpe), "max_drawdown": float(dd.min()),
        "avg_month": float(monthly.mean()), "median_month": float(monthly.median()),
        "worst_month": float(monthly.min()), "best_month": float(monthly.max()),
        "positive_month_pct": float((monthly > 0).mean()),
        "month_ge_10_pct": float((monthly >= 0.10).mean()), "months": int(len(monthly)),
    }


def train_score(m: Dict[str, float]) -> float:
    if not m or not np.isfinite(m.get("sharpe", np.nan)):
        return -1e9
    if m["max_drawdown"] < -0.45:
        return -1e9
    return 2.0 * m["sharpe"] + 0.5 * m["cagr"] + m["max_drawdown"]


def candidates() -> List[Candidate]:
    out = []
    for sma in [100, 150, 200]:
        for vol in [21, 63]:
            out.append(Candidate("inverse_vol", (sma, vol, 21)))
    for fast in [20, 50]:
        for slow in [100, 150, 200]:
            if fast < slow:
                out.append(Candidate("dual_ema", (fast, slow)))
    for lb in [126, 252]:
        for topn in [3, 5]:
            for reb in [5, 21]:
                out.append(Candidate("risk_managed_momentum", (lb, topn, reb, 20, 150)))
                out.append(Candidate("risk_managed_momentum", (lb, topn, reb, 35, 200)))
                out.append(Candidate("lowvol_momentum", (lb, topn, reb)))
    for br in [126, 252]:
        for atr in [3.0, 4.0]:
            out.append(Candidate("atr_trend", (br, atr, 10, 5)))
    out.append(Candidate("ensemble", (20, 150)))
    out.append(Candidate("ensemble", (35, 200)))
    return out


def build_candidate(c: Candidate, close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, spy: pd.Series) -> pd.Series:
    if c.family == "inverse_vol":
        w = invvol_weights(close, spy, *c.params)
        return portfolio_returns(close, w)
    if c.family == "dual_ema":
        w = dualema_weights(close, spy, *c.params)
        return portfolio_returns(close, w)
    if c.family == "risk_managed_momentum":
        lb, topn, reb, target_pct, cap100 = c.params
        w = topmom_weights(close, spy, lb, topn, reb)
        raw = portfolio_returns(close, w)
        return vol_scale_returns(raw, target_pct / 100, cap100 / 100)
    if c.family == "lowvol_momentum":
        w = lowvol_mom_weights(close, spy, *c.params)
        return portfolio_returns(close, w)
    if c.family == "atr_trend":
        w = atr_trend_weights(close, high, low, spy, *c.params)
        return portfolio_returns(close, w)
    if c.family == "ensemble":
        target_pct, cap100 = c.params
        w1 = invvol_weights(close, spy, 150, 21, 21)
        w2 = dualema_weights(close, spy, 50, 150)
        w3 = topmom_weights(close, spy, 126, 3, 5)
        raw = portfolio_returns(close, (w1 + w2 + w3) / 3)
        return vol_scale_returns(raw, target_pct / 100, cap100 / 100)
    raise ValueError(c.family)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_all()
    idx = master_index(data)
    close = aligned_close(data, idx)
    high = aligned_ohlc(data, idx, "high")
    low = aligned_ohlc(data, idx, "low")
    spy = data["SPY"]["close"].reindex(idx)

    cache: Dict[str, pd.Series] = {}
    rows = []
    for c in candidates():
        print("testing", c.name)
        r = build_candidate(c, close, high, low, spy)
        cache[c.name] = r
        for split, s in [("train", r.loc[:TRAIN_END]), ("test", r.loc[TEST_START:]), ("full", r)]:
            rows.append({"candidate": c.name, "family": c.family, "split": split, **metrics(s)})
    all_results = pd.DataFrame(rows)
    all_results.to_csv(OUT / "all_results.csv", index=False)

    selected_rows = []
    selected_returns = {}
    for family in sorted(all_results.family.unique()):
        tr = all_results[(all_results.family == family) & (all_results.split == "train")].copy()
        tr["score"] = tr.apply(lambda x: train_score(x.to_dict()), axis=1)
        best = tr.sort_values("score", ascending=False).iloc[0]
        name = best.candidate
        selected_returns[family] = cache[name]
        te = all_results[(all_results.candidate == name) & (all_results.split == "test")].iloc[0].to_dict()
        selected_rows.append(te)
    selected = pd.DataFrame(selected_rows).sort_values("sharpe", ascending=False)
    selected["passes_sharpe_1_5"] = selected["sharpe"] >= 1.5
    selected.to_csv(OUT / "selected_oos.csv", index=False)

    # Equal-weight selected-family ensemble, evaluated independently.
    family_ensemble = pd.concat(selected_returns, axis=1).mean(axis=1)
    ensemble_rows = []
    for drag in [0.0, 0.10, 0.20]:
        adjusted = family_ensemble - drag / TRADING_DAYS
        ensemble_rows.append({"annual_funding_drag": drag, **metrics(adjusted.loc[TEST_START:])})
    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df.to_csv(OUT / "family_ensemble_sensitivity.csv", index=False)

    cols = ["candidate", "family", "cagr", "sharpe", "max_drawdown", "avg_month", "worst_month", "positive_month_pct", "month_ge_10_pct", "passes_sharpe_1_5"]
    lines = [
        "# Wave 5 — academic strategy replication",
        "",
        f"- Data: {idx.min()} to {idx.max()}",
        f"- Universe: {', '.join(STOCKS)}",
        "- Train selection: 2014–2022; strict OOS: 2023–2026-08",
        f"- One-way execution cost: {ONE_WAY_COST:.2%}",
        "- Signals use completed daily closes and affect the next daily return.",
        "",
        "## Train-selected family models — OOS",
        "",
        selected[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Selected-family ensemble funding-drag sensitivity",
        "",
        ensemble_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "Sharpe >= 1.5 is a screening threshold, not proof of future profitability. Survivorship bias, stock-perpetual funding, overnight mark-price deviations, and model-selection risk remain.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
