from __future__ import annotations

import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1h"
START_MONTH = "2023-01"
END_MONTH = "2026-07"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
OUTDIR = Path("results")
CACHE = Path(".cache")
ONE_WAY_COST = 0.0008  # 0.06% taker fee + 0.02% slippage, per one-way notional change
ANNUAL_HOURS = 24 * 365
TRAIN_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

RISK_PROFILES = {
    "base": {"target_vol": 0.35, "max_leverage": 2.0},
    "aggressive": {"target_vol": 0.70, "max_leverage": 3.0},
}


@dataclass(frozen=True)
class Candidate:
    family: str
    params: Tuple[int, ...]

    @property
    def name(self) -> str:
        p = "-".join(str(x) for x in self.params)
        return f"{self.family}_{p}"


def month_range(start: str, end: str) -> Iterable[str]:
    for p in pd.period_range(start=start, end=end, freq="M"):
        yield str(p)


def download_zip(url: str, dest: Path, attempts: int = 4) -> bytes | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()

    headers = {"User-Agent": "Mozilla/5.0 crypto-backtest/1.0"}
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, headers=headers, timeout=45)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return r.content
        except requests.RequestException as exc:
            if attempt == attempts:
                print(f"FAILED {url}: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


def load_symbol(symbol: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    for ym in month_range(START_MONTH, END_MONTH):
        fname = f"{symbol}-{INTERVAL}-{ym}.zip"
        url = f"{BASE_URL}/{symbol}/{INTERVAL}/{fname}"
        raw = download_zip(url, CACHE / fname)
        if raw is None:
            print(f"SKIP missing {fname}")
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(member) as fh:
                    df = pd.read_csv(fh, header=None, names=cols)
        except Exception as exc:
            print(f"SKIP invalid {fname}: {exc}")
            continue
        parts.append(df)

    if not parts:
        raise RuntimeError(f"No data downloaded for {symbol}")

    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df[["time", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]]
        .dropna()
        .drop_duplicates("time")
        .set_index("time")
        .sort_index()
    )
    return df


def align_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common = None
    for df in data.values():
        common = df.index if common is None else common.intersection(df.index)
    assert common is not None
    common = common.sort_values()
    return {s: df.loc[common].copy() for s, df in data.items()}


def stateful_breakout_signal(close: pd.Series, lookback: int) -> pd.Series:
    upper = close.rolling(lookback, min_periods=lookback).max().shift(1)
    lower = close.rolling(lookback, min_periods=lookback).min().shift(1)
    event = pd.Series(np.nan, index=close.index, dtype=float)
    event[close > upper] = 1.0
    event[close < lower] = -1.0
    return event.ffill().fillna(0.0)


def build_weights(data: Dict[str, pd.DataFrame], candidate: Candidate) -> pd.DataFrame:
    idx = next(iter(data.values())).index
    weights = pd.DataFrame(0.0, index=idx, columns=SYMBOLS)

    if candidate.family == "ema":
        fast, slow = candidate.params
        for s, df in data.items():
            ema_fast = df["close"].ewm(span=fast, adjust=False, min_periods=slow).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
            sig = np.sign(ema_fast - ema_slow)
            weights[s] = pd.Series(sig, index=idx).shift(1).fillna(0.0) / len(SYMBOLS)

    elif candidate.family == "tsmom":
        (lookback,) = candidate.params
        for s, df in data.items():
            mom = df["close"].pct_change(lookback)
            vol = df["close"].pct_change().rolling(lookback, min_periods=lookback).std()
            z = mom / (vol * math.sqrt(lookback)).replace(0, np.nan)
            sig = pd.Series(np.where(z > 0.20, 1.0, np.where(z < -0.20, -1.0, 0.0)), index=idx)
            weights[s] = sig.shift(1).fillna(0.0) / len(SYMBOLS)

    elif candidate.family == "donchian":
        (lookback,) = candidate.params
        for s, df in data.items():
            sig = stateful_breakout_signal(df["close"], lookback)
            weights[s] = sig.shift(1).fillna(0.0) / len(SYMBOLS)

    elif candidate.family == "xsmom":
        lookback, rebalance_hours = candidate.params
        scores = pd.DataFrame(
            {s: df["close"].pct_change(lookback) for s, df in data.items()},
            index=idx,
        )
        raw = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
        rebalance_mask = np.arange(len(idx)) % rebalance_hours == 0
        for t in idx[rebalance_mask]:
            row = scores.loc[t].dropna()
            if len(row) < 4:
                continue
            ranked = row.sort_values()
            raw.loc[t, ranked.index[0]] = -0.5
            raw.loc[t, ranked.index[-1]] = 0.5
            for middle in ranked.index[1:-1]:
                raw.loc[t, middle] = 0.0
        weights = raw.ffill().fillna(0.0).shift(1).fillna(0.0)

    elif candidate.family == "flowtrend":
        lookback, slow = candidate.params
        for s, df in data.items():
            buy_ratio = (df["taker_buy_quote"] / df["quote_volume"].replace(0, np.nan)).clip(0, 1)
            flow = buy_ratio.rolling(lookback, min_periods=lookback).mean() - 0.5
            trend = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
            sig = np.where((flow > 0.01) & (df["close"] > trend), 1.0,
                           np.where((flow < -0.01) & (df["close"] < trend), -1.0, 0.0))
            weights[s] = pd.Series(sig, index=idx).shift(1).fillna(0.0) / len(SYMBOLS)
    else:
        raise ValueError(candidate.family)

    return weights.astype(float)


def portfolio_components(
    data: Dict[str, pd.DataFrame], weights: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    rets = pd.DataFrame(
        {s: df["close"].pct_change().fillna(0.0) for s, df in data.items()},
        index=weights.index,
    )
    gross = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return gross, turnover


def apply_risk_profile(
    gross: pd.Series,
    turnover: pd.Series,
    target_vol: float,
    max_leverage: float,
) -> pd.Series:
    rolling_vol = gross.rolling(24 * 30, min_periods=24 * 10).std() * math.sqrt(ANNUAL_HOURS)
    lev = (target_vol / rolling_vol.replace(0, np.nan)).clip(lower=0.0, upper=max_leverage)
    lev = lev.shift(1).fillna(0.0)
    net = lev * gross - lev * turnover * ONE_WAY_COST
    return net.clip(lower=-0.95)


def metrics(r: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    if len(r) < 24 * 30:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / ANNUAL_HOURS
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 and years > 0 else -1.0
    ann_vol = r.std() * math.sqrt(ANNUAL_HOURS)
    sharpe = (r.mean() * ANNUAL_HOURS) / ann_vol if ann_vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    monthly = (1 + r).resample("ME").prod() - 1
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    pf = pos / neg if neg > 0 else np.nan

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "profit_factor": float(pf),
        "avg_month": float(monthly.mean()),
        "median_month": float(monthly.median()),
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "positive_month_pct": float((monthly > 0).mean()),
        "month_ge_10_pct": float((monthly >= 0.10).mean()),
        "months": int(len(monthly)),
    }


def candidate_grid() -> List[Candidate]:
    out: List[Candidate] = []
    for fast in [24, 48, 72]:
        for slow in [168, 336, 504]:
            if fast < slow:
                out.append(Candidate("ema", (fast, slow)))
    for lb in [72, 168, 336, 720]:
        out.append(Candidate("tsmom", (lb,)))
        out.append(Candidate("donchian", (lb,)))
    for lb in [72, 168, 336]:
        for reb in [8, 24]:
            out.append(Candidate("xsmom", (lb, reb)))
    for lb in [24, 72]:
        for slow in [168, 336]:
            out.append(Candidate("flowtrend", (lb, slow)))
    return out


def score_train(m: Dict[str, float]) -> float:
    if not m or not np.isfinite(m["sharpe"]):
        return -1e9
    if m["max_drawdown"] < -0.55:
        return -1e9
    return (
        1.5 * m["sharpe"]
        + 2.0 * m["avg_month"]
        + 0.5 * m["positive_month_pct"]
        + 0.5 * m["max_drawdown"]
    )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)

    print("Downloading data...")
    data = {s: load_symbol(s) for s in SYMBOLS}
    data = align_data(data)
    idx = next(iter(data.values())).index
    print(f"Aligned observations: {len(idx):,} from {idx.min()} to {idx.max()}")

    candidates = candidate_grid()
    rows: List[Dict[str, object]] = []
    monthly_columns: Dict[str, pd.Series] = {}
    return_cache: Dict[Tuple[str, str], pd.Series] = {}

    for candidate in candidates:
        print("Testing", candidate.name)
        weights = build_weights(data, candidate)
        gross, turnover = portfolio_components(data, weights)

        for profile, cfg in RISK_PROFILES.items():
            net = apply_risk_profile(gross, turnover, **cfg)
            return_cache[(candidate.name, profile)] = net

            train = net.loc[:TRAIN_END]
            test = net.loc[TEST_START:]
            full = net
            for split_name, series in [("train", train), ("test", test), ("full", full)]:
                m = metrics(series)
                row: Dict[str, object] = {
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "params": json.dumps(candidate.params),
                    "profile": profile,
                    "split": split_name,
                }
                row.update(m)
                rows.append(row)

            monthly_columns[f"{candidate.name}|{profile}"] = (1 + net).resample("ME").prod() - 1

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "all_candidates.csv", index=False)

    selected_rows: List[Dict[str, object]] = []
    selected_series: Dict[str, pd.Series] = {}
    for profile in RISK_PROFILES:
        train_rows = results[(results["profile"] == profile) & (results["split"] == "train")].copy()
        train_rows["selection_score"] = train_rows.apply(
            lambda row: score_train(row.to_dict()), axis=1
        )
        for family in sorted(train_rows["family"].unique()):
            fam = train_rows[train_rows["family"] == family].sort_values(
                "selection_score", ascending=False
            )
            best = fam.iloc[0]
            candidate_name = str(best["candidate"])
            selected_series[f"{family}|{profile}"] = return_cache[(candidate_name, profile)]
            test_row = results[
                (results["candidate"] == candidate_name)
                & (results["profile"] == profile)
                & (results["split"] == "test")
            ].iloc[0].to_dict()
            test_row["selected_on_train_score"] = float(best["selection_score"])
            selected_rows.append(test_row)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUTDIR / "selected_out_of_sample.csv", index=False)

    ensemble_rows: List[Dict[str, object]] = []
    ensemble_monthly: Dict[str, pd.Series] = {}
    for profile in RISK_PROFILES:
        fam_series = [s for key, s in selected_series.items() if key.endswith(f"|{profile}")]
        if not fam_series:
            continue
        ens = pd.concat(fam_series, axis=1).mean(axis=1)
        for split_name, series in [
            ("train", ens.loc[:TRAIN_END]),
            ("test", ens.loc[TEST_START:]),
            ("full", ens),
        ]:
            row = {"candidate": "selected_family_ensemble", "profile": profile, "split": split_name}
            row.update(metrics(series))
            ensemble_rows.append(row)
        ensemble_monthly[f"ensemble|{profile}"] = (1 + ens).resample("ME").prod() - 1

    ensemble = pd.DataFrame(ensemble_rows)
    ensemble.to_csv(OUTDIR / "ensemble.csv", index=False)

    monthly = pd.DataFrame({**monthly_columns, **ensemble_monthly}).sort_index()
    monthly.to_csv(OUTDIR / "monthly_returns.csv")

    lines: List[str] = []
    lines.append("# Crypto futures strategy scan")
    lines.append("")
    lines.append(f"- Data: Binance USD-M perpetual 1h klines, {idx.min()} to {idx.max()}")
    lines.append(f"- Symbols: {', '.join(SYMBOLS)}")
    lines.append(f"- Conservative one-way execution cost: {ONE_WAY_COST:.2%}")
    lines.append(f"- Training period ends: {TRAIN_END}")
    lines.append(f"- Strict out-of-sample test begins: {TEST_START}")
    lines.append("")
    lines.append("## Selected family models — out of sample")
    lines.append("")
    report_cols = [
        "candidate", "family", "profile", "cagr", "sharpe", "max_drawdown",
        "avg_month", "worst_month", "positive_month_pct", "month_ge_10_pct",
    ]
    lines.append(selected[report_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Selected-family ensemble")
    lines.append("")
    ens_test = ensemble[ensemble["split"] == "test"]
    ens_cols = [
        "candidate", "profile", "cagr", "sharpe", "max_drawdown",
        "avg_month", "worst_month", "positive_month_pct", "month_ge_10_pct",
    ]
    lines.append(ens_test[ens_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append(
        "A model is not considered deployable merely because average monthly return exceeds 10%. "
        "It must also remain profitable out of sample, have Sharpe above 1.0, maximum drawdown better "
        "than -30%, positive months above 55%, and no single month dominating the result."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `all_candidates.csv`: every parameter set and split")
    lines.append("- `selected_out_of_sample.csv`: one train-selected model per family")
    lines.append("- `ensemble.csv`: ensemble results")
    lines.append("- `monthly_returns.csv`: monthly return series")
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
