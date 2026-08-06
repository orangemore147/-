from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
TRADE_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
START = "2015-01-01"
END = "2026-08-01"
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01", tz="UTC")
OUTDIR = Path("results_wave4")
CACHE = Path(".cache_equities")
ONE_WAY_COST = 0.0008  # assumed stock-contract taker fee + slippage
DAYS_YEAR = 252

RISK_PROFILES = {
    "base": (0.20, 1.5),
    "balanced": (0.35, 2.0),
    "aggressive": (0.55, 3.0),
}


@dataclass(frozen=True)
class Candidate:
    family: str
    params: Tuple[int, ...]

    @property
    def name(self) -> str:
        return self.family + "_" + "-".join(str(x) for x in self.params)


def yahoo_chart(ticker: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{ticker}.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        return frame

    p1 = int(pd.Timestamp(START, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(END, tz="UTC").timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits"
    )
    headers = {"User-Agent": "Mozilla/5.0 equity-research/1.0"}
    last_error = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=60, headers=headers)
            response.raise_for_status()
            payload = response.json()["chart"]["result"][0]
            timestamps = pd.to_datetime(payload["timestamp"], unit="s", utc=True).normalize()
            quote = payload["indicators"]["quote"][0]
            adj = payload["indicators"].get("adjclose", [{}])[0].get("adjclose")
            frame = pd.DataFrame(
                {
                    "time": timestamps,
                    "open": quote["open"],
                    "high": quote["high"],
                    "low": quote["low"],
                    "close": quote["close"],
                    "volume": quote["volume"],
                    "adjclose": adj if adj is not None else quote["close"],
                }
            ).dropna(subset=["adjclose"])
            # Scale raw OHLC by adjusted/raw close ratio to make split/dividend history consistent.
            ratio = frame["adjclose"] / frame["close"].replace(0, np.nan)
            for col in ["open", "high", "low", "close"]:
                frame[col] = frame[col] * ratio
            frame = frame.drop_duplicates("time").set_index("time").sort_index()
            frame.to_csv(path)
            return frame
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Yahoo download failed for {ticker}: {last_error}")


def align(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common = None
    for ticker, frame in data.items():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) < 1000:
        raise RuntimeError("Insufficient common history")
    common = common.sort_values()
    return {ticker: frame.loc[common].copy() for ticker, frame in data.items()}


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def normalize_positive(row: pd.Series, top_k: int | None = None) -> pd.Series:
    row = row.clip(lower=0).fillna(0.0)
    if top_k is not None and (row > 0).sum() > top_k:
        keep = row.nlargest(top_k).index
        row.loc[~row.index.isin(keep)] = 0.0
    total = row.sum()
    return row / total if total > 0 else row * 0.0


def build_weights(data: Dict[str, pd.DataFrame], candidate: Candidate) -> pd.DataFrame:
    index = next(iter(data.values())).index
    close = pd.DataFrame({t: data[t]["adjclose"] for t in TICKERS}, index=index)
    weights = pd.DataFrame(0.0, index=index, columns=TRADE_TICKERS)
    spy = close["SPY"]
    qqq = close["QQQ"]

    if candidate.family == "sma_long":
        (days,) = candidate.params
        regime = spy > spy.rolling(days, min_periods=days).mean()
        for ticker in TRADE_TICKERS:
            own = close[ticker] > close[ticker].rolling(days, min_periods=days).mean()
            weights[ticker] = (regime & own).astype(float) / len(TRADE_TICKERS)

    elif candidate.family == "dual_ema":
        fast, slow = candidate.params
        regime = qqq > qqq.ewm(span=slow, adjust=False, min_periods=slow).mean()
        for ticker in TRADE_TICKERS:
            ef = close[ticker].ewm(span=fast, adjust=False, min_periods=slow).mean()
            es = close[ticker].ewm(span=slow, adjust=False, min_periods=slow).mean()
            weights[ticker] = (regime & (ef > es)).astype(float) / len(TRADE_TICKERS)

    elif candidate.family == "top_momentum":
        lookback, top_k, rebalance, regime_days = candidate.params
        scores = close[TRADE_TICKERS].pct_change(lookback)
        regime = spy > spy.rolling(regime_days, min_periods=regime_days).mean()
        raw = pd.DataFrame(np.nan, index=index, columns=TRADE_TICKERS)
        for i in range(len(index)):
            if i % rebalance != 0:
                continue
            if not bool(regime.iloc[i]):
                raw.iloc[i] = 0.0
                continue
            row = scores.iloc[i].dropna()
            positive = row[row > 0].nlargest(top_k)
            signal = pd.Series(0.0, index=TRADE_TICKERS)
            if len(positive):
                signal.loc[positive.index] = positive
            raw.iloc[i] = normalize_positive(signal, top_k)
        weights = raw.ffill().fillna(0.0)

    elif candidate.family == "inverse_vol":
        regime_days, vol_days, rebalance = candidate.params
        regime = spy > spy.rolling(regime_days, min_periods=regime_days).mean()
        vol = close[TRADE_TICKERS].pct_change().rolling(vol_days, min_periods=vol_days).std()
        raw = pd.DataFrame(np.nan, index=index, columns=TRADE_TICKERS)
        for i in range(len(index)):
            if i % rebalance != 0:
                continue
            if not bool(regime.iloc[i]):
                raw.iloc[i] = 0.0
                continue
            inv = 1 / vol.iloc[i].replace(0, np.nan)
            raw.iloc[i] = normalize_positive(inv)
        weights = raw.ffill().fillna(0.0)

    elif candidate.family == "pullback":
        trend_days, entry_rsi, exit_rsi = candidate.params
        regime = spy > spy.rolling(200, min_periods=200).mean()
        for ticker in TRADE_TICKERS:
            trend = close[ticker].rolling(trend_days, min_periods=trend_days).mean()
            rv = rsi(close[ticker], 14)
            state = 0.0
            out = pd.Series(0.0, index=index)
            for i in range(len(index)):
                if state == 0 and regime.iloc[i] and close[ticker].iloc[i] > trend.iloc[i] and rv.iloc[i] < entry_rsi:
                    state = 1.0
                elif state == 1 and (rv.iloc[i] > exit_rsi or close[ticker].iloc[i] < trend.iloc[i] or not regime.iloc[i]):
                    state = 0.0
                out.iloc[i] = state
            weights[ticker] = out / len(TRADE_TICKERS)

    elif candidate.family == "breakout":
        entry_days, exit_days = candidate.params
        regime = qqq > qqq.rolling(200, min_periods=200).mean()
        for ticker in TRADE_TICKERS:
            upper = close[ticker].rolling(entry_days, min_periods=entry_days).max().shift(1)
            lower = close[ticker].rolling(exit_days, min_periods=exit_days).min().shift(1)
            state = 0.0
            out = pd.Series(0.0, index=index)
            for i in range(len(index)):
                if state == 0 and regime.iloc[i] and pd.notna(upper.iloc[i]) and close[ticker].iloc[i] > upper.iloc[i]:
                    state = 1.0
                elif state == 1 and (not regime.iloc[i] or (pd.notna(lower.iloc[i]) and close[ticker].iloc[i] < lower.iloc[i])):
                    state = 0.0
                out.iloc[i] = state
            weights[ticker] = out / len(TRADE_TICKERS)

    elif candidate.family == "single_rotation":
        lookback, rebalance, regime_days = candidate.params
        scores = close[TRADE_TICKERS].pct_change(lookback)
        regime = qqq > qqq.rolling(regime_days, min_periods=regime_days).mean()
        raw = pd.DataFrame(np.nan, index=index, columns=TRADE_TICKERS)
        for i in range(len(index)):
            if i % rebalance != 0:
                continue
            signal = pd.Series(0.0, index=TRADE_TICKERS)
            row = scores.iloc[i].dropna()
            if bool(regime.iloc[i]) and len(row) and row.max() > 0:
                signal.loc[row.idxmax()] = 1.0
            raw.iloc[i] = signal
        weights = raw.ffill().fillna(0.0)

    else:
        raise ValueError(candidate.family)

    # Signals are from completed daily close; apply from following close-to-close period.
    return weights.shift(1).fillna(0.0)


def strategy_returns(
    data: Dict[str, pd.DataFrame], base_weights: pd.DataFrame, target_vol: float, cap: float
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    returns = pd.DataFrame(
        {t: data[t]["adjclose"].pct_change().fillna(0.0) for t in TRADE_TICKERS},
        index=base_weights.index,
    )
    unlevered = (base_weights * returns).sum(axis=1)
    rolling_vol = unlevered.rolling(63, min_periods=21).std() * math.sqrt(DAYS_YEAR)
    leverage = (target_vol / rolling_vol.replace(0, np.nan)).clip(0, cap).shift(1).fillna(0.0)
    actual = base_weights.mul(leverage, axis=0)
    gross = (actual * returns).sum(axis=1)
    turnover = actual.diff().abs().sum(axis=1).fillna(actual.abs().sum(axis=1))
    net = gross - turnover * ONE_WAY_COST
    return net.clip(lower=-0.95), actual, turnover


def metrics(r: pd.Series, weights: pd.DataFrame, turnover: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / DAYS_YEAR
    monthly = (1 + r).resample("ME").prod() - 1
    drawdown = equity / equity.cummax() - 1
    vol = r.std() * math.sqrt(DAYS_YEAR)
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else -1.0
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(r.mean() * DAYS_YEAR / vol) if vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "avg_month": float(monthly.mean()),
        "median_month": float(monthly.median()),
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "positive_month_pct": float((monthly > 0).mean()),
        "month_ge_10_pct": float((monthly >= 0.10).mean()),
        "month_le_minus10_pct": float((monthly <= -0.10).mean()),
        "months": int(len(monthly)),
        "avg_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "annual_turnover": float(turnover.mean() * DAYS_YEAR),
    }


def grid() -> List[Candidate]:
    candidates: List[Candidate] = []
    for days in [50, 100, 150, 200]:
        candidates.append(Candidate("sma_long", (days,)))
    for fast in [10, 20, 50]:
        for slow in [100, 150, 200]:
            if fast < slow:
                candidates.append(Candidate("dual_ema", (fast, slow)))
    for lookback in [21, 63, 126, 252]:
        for top_k in [1, 2, 3, 5]:
            for rebalance in [5, 21, 63]:
                candidates.append(Candidate("top_momentum", (lookback, top_k, rebalance, 200)))
    for regime in [100, 150, 200]:
        for vol in [21, 63]:
            for rebalance in [5, 21]:
                candidates.append(Candidate("inverse_vol", (regime, vol, rebalance)))
    for trend in [50, 100, 200]:
        for entry in [30, 35, 40]:
            for exit_ in [55, 60, 65]:
                candidates.append(Candidate("pullback", (trend, entry, exit_)))
    for entry in [20, 50, 100]:
        for exit_ in [10, 20, 50]:
            if exit_ < entry:
                candidates.append(Candidate("breakout", (entry, exit_)))
    for lookback in [21, 63, 126, 252]:
        for rebalance in [5, 21, 63]:
            candidates.append(Candidate("single_rotation", (lookback, rebalance, 200)))
    return candidates


def train_score(row: pd.Series) -> float:
    if not np.isfinite(row.get("sharpe", np.nan)) or row["cagr"] <= 0:
        return -1e9
    if row["max_drawdown"] < -0.50:
        return -1e9
    return (
        1.8 * row["sharpe"]
        + 0.7 * row["positive_month_pct"]
        + 0.5 * row["cagr"]
        + 0.8 * row["max_drawdown"]
        - 0.001 * row["annual_turnover"]
    )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    data = align({ticker: yahoo_chart(ticker) for ticker in TICKERS})
    index = next(iter(data.values())).index
    print(f"Data {index.min()} to {index.max()}, {len(index)} sessions")

    rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, str], Tuple[pd.Series, pd.DataFrame, pd.Series]] = {}
    candidates = grid()

    for number, candidate in enumerate(candidates, 1):
        print(f"[{number}/{len(candidates)}] {candidate.name}")
        base = build_weights(data, candidate)
        for profile, (target, cap) in RISK_PROFILES.items():
            net, actual, turnover = strategy_returns(data, base, target, cap)
            cache[(candidate.name, profile)] = (net, actual, turnover)
            for split, selector in [
                ("train", net.index <= TRAIN_END),
                ("test", net.index >= TEST_START),
                ("full", np.ones(len(net), dtype=bool)),
            ]:
                row: Dict[str, object] = {
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "params": json.dumps(candidate.params),
                    "profile": profile,
                    "split": split,
                }
                row.update(metrics(net.loc[selector], actual.loc[selector], turnover.loc[selector]))
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "all_candidates.csv", index=False)

    selected_rows: List[Dict[str, object]] = []
    selected_series: Dict[Tuple[str, str], pd.Series] = {}
    for profile in RISK_PROFILES:
        train = results[(results.profile == profile) & (results.split == "train")].copy()
        train["score"] = train.apply(train_score, axis=1)
        for family in sorted(train.family.unique()):
            best = train[train.family == family].sort_values("score", ascending=False).iloc[0]
            test = results[
                (results.candidate == best.candidate)
                & (results.profile == profile)
                & (results.split == "test")
            ].iloc[0].to_dict()
            test["train_score"] = float(best.score)
            test["train_cagr"] = float(best.cagr)
            test["train_sharpe"] = float(best.sharpe)
            selected_rows.append(test)
            selected_series[(family, profile)] = cache[(str(best.candidate), profile)][0]

    selected = pd.DataFrame(selected_rows)
    selected["passes_oos"] = (
        (selected.cagr > 0)
        & (selected.sharpe >= 0.75)
        & (selected.max_drawdown >= -0.35)
        & (selected.positive_month_pct >= 0.55)
        & (selected.avg_month > 0)
    )
    selected.to_csv(OUTDIR / "selected_oos.csv", index=False)

    ensemble_rows: List[Dict[str, object]] = []
    for profile in RISK_PROFILES:
        series = [v for (family, p), v in selected_series.items() if p == profile]
        ensemble = pd.concat(series, axis=1).mean(axis=1)
        for split, subset in [
            ("train", ensemble.loc[:TRAIN_END]),
            ("test", ensemble.loc[TEST_START:]),
            ("full", ensemble),
        ]:
            fake_w = pd.DataFrame({"ensemble": (subset != 0).astype(float)}, index=subset.index)
            fake_turnover = pd.Series(0.0, index=subset.index)
            row: Dict[str, object] = {"profile": profile, "split": split}
            row.update(metrics(subset, fake_w, fake_turnover))
            ensemble_rows.append(row)
    ensembles = pd.DataFrame(ensemble_rows)
    ensembles.to_csv(OUTDIR / "ensembles.csv", index=False)

    cols = [
        "candidate", "family", "profile", "passes_oos", "cagr", "sharpe",
        "max_drawdown", "avg_month", "worst_month", "positive_month_pct",
        "month_ge_10_pct", "month_le_minus10_pct", "annual_turnover",
    ]
    ordered = selected.sort_values(["passes_oos", "sharpe", "cagr"], ascending=[False, False, False])
    lines = [
        "# Wave 4 — US mega-cap stock-contract strategies",
        "",
        f"- Data: Yahoo adjusted daily prices, {index.min()} to {index.max()}",
        f"- Trade universe: {', '.join(TRADE_TICKERS)}",
        f"- One-way assumed execution cost: {ONE_WAY_COST:.2%}",
        "- Parameter selection: 2015–2022; strict OOS: 2023–2026-07",
        "- Signals use completed close and are applied to the next close-to-close return",
        "",
        "## Train-selected family models — OOS",
        "",
        ordered[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Family ensemble — OOS",
        "",
        ensembles[ensembles.split == "test"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OOS pass rule",
        "",
        "CAGR > 0; Sharpe >= 0.75; max drawdown >= -35%; positive months >= 55%; average month > 0.",
        "A 10% month is measured as a historical frequency, never treated as a guaranteed target.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
