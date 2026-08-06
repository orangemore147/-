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

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT",
]
START_MONTH = "2023-01"
END_MONTH = "2026-07"
INTERVAL = "1h"
CACHE = Path(".cache_wave3")
OUTDIR = Path("results_wave3")
TRAIN_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
HOURS_YEAR = 24 * 365

FUTURES_KLINES = "https://data.binance.vision/data/futures/um/monthly/klines"
SPOT_KLINES = "https://data.binance.vision/data/spot/monthly/klines"
FUNDING = "https://data.binance.vision/data/futures/um/monthly/fundingRate"

# Per pair, one unit means long $1 spot and short $1 perpetual.
# Baseline entry or exit cost:
# spot taker 0.10% + spot slip 0.02% + perp taker 0.06% + perp slip 0.02% = 0.20%.
COST_PROFILES = {
    "taker": 0.0020,
    "optimistic_limit": 0.0014,
}


@dataclass(frozen=True)
class Config:
    family: str
    lookback_events: int
    top_k: int
    threshold_bps: float
    rebalance_events: int
    keep_bps: float

    @property
    def name(self) -> str:
        return (
            f"{self.family}_lb{self.lookback_events}_k{self.top_k}_"
            f"th{self.threshold_bps:g}_reb{self.rebalance_events}_keep{self.keep_bps:g}"
        )


def month_range(start: str, end: str) -> Iterable[str]:
    yield from (str(x) for x in pd.period_range(start=start, end=end, freq="M"))


def get_bytes(url: str, path: Path, retries: int = 4) -> bytes | None:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    headers = {"User-Agent": "Mozilla/5.0 funding-carry-research/1.0"}
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return response.content
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"FAILED {url}: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


def read_zip(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as handle:
            return pd.read_csv(handle, header=None, dtype=str)


def parse_timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    median = float(valid.abs().median())
    unit = "us" if median > 1e14 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def load_kline(symbol: str, market: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    base = SPOT_KLINES if market == "spot" else FUTURES_KLINES
    cache_prefix = "spot" if market == "spot" else "perp"
    names = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    for month in month_range(START_MONTH, END_MONTH):
        filename = f"{symbol}-{INTERVAL}-{month}.zip"
        url = f"{base}/{symbol}/{INTERVAL}/{filename}"
        raw = get_bytes(url, CACHE / cache_prefix / filename)
        if raw is None:
            print(f"MISSING {market} {filename}")
            continue
        frame = read_zip(raw)
        if frame.shape[1] < 12:
            print(f"BAD COLUMNS {market} {filename}: {frame.shape[1]}")
            continue
        frame = frame.iloc[:, :12]
        frame.columns = names
        frame["time"] = parse_timestamp(frame["open_time"])
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["time", "close"])
        parts.append(frame[["time", "open", "high", "low", "close", "volume", "quote_volume"]])
    if not parts:
        raise RuntimeError(f"No {market} klines for {symbol}")
    result = pd.concat(parts, ignore_index=True)
    return result.drop_duplicates("time").set_index("time").sort_index()


def load_funding(symbol: str) -> pd.Series:
    parts: List[pd.DataFrame] = []
    for month in month_range(START_MONTH, END_MONTH):
        filename = f"{symbol}-fundingRate-{month}.zip"
        url = f"{FUNDING}/{symbol}/{filename}"
        raw = get_bytes(url, CACHE / "funding" / filename)
        if raw is None:
            print(f"MISSING funding {filename}")
            continue
        frame = read_zip(raw)
        if frame.empty:
            continue

        first_row = [str(x).lower() for x in frame.iloc[0].tolist()]
        has_header = any("time" in x or "rate" in x for x in first_row)
        if has_header:
            frame.columns = first_row
            frame = frame.iloc[1:].copy()
            time_candidates = [c for c in frame.columns if "time" in str(c)]
            rate_candidates = [c for c in frame.columns if "rate" in str(c)]
            if not time_candidates or not rate_candidates:
                continue
            time_col, rate_col = time_candidates[0], rate_candidates[-1]
        else:
            time_col = frame.columns[0]
            rate_col = frame.columns[-1]

        parsed = pd.DataFrame({
            "time": parse_timestamp(frame[time_col]),
            "rate": pd.to_numeric(frame[rate_col], errors="coerce"),
        }).dropna()
        parts.append(parsed)

    if not parts:
        raise RuntimeError(f"No funding data for {symbol}")
    result = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    result["time"] = result["time"].dt.floor("h")
    return result.drop_duplicates("time", keep="last").set_index("time")["rate"]


def load_all() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    spot: Dict[str, pd.DataFrame] = {}
    perp: Dict[str, pd.DataFrame] = {}
    funding_cols: Dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        print(f"Loading {symbol}")
        spot[symbol] = load_kline(symbol, "spot")
        perp[symbol] = load_kline(symbol, "perp")
        funding_cols[symbol] = load_funding(symbol)

    common = None
    for symbol in SYMBOLS:
        idx = spot[symbol].index.intersection(perp[symbol].index)
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        raise RuntimeError("No common spot/perp timeline")
    common = common.sort_values()
    spot = {s: spot[s].loc[common] for s in SYMBOLS}
    perp = {s: perp[s].loc[common] for s in SYMBOLS}
    funding = pd.DataFrame(funding_cols).reindex(common)
    return spot, perp, funding


def configs() -> List[Config]:
    out: List[Config] = []
    for lb in [1, 3, 6, 12]:
        for k in [1, 2, 3, 5]:
            for threshold in [0.0, 0.5, 1.0, 2.0, 5.0]:  # basis points per funding event
                for rebalance in [1, 3, 6]:
                    out.append(Config("ranked_carry", lb, k, threshold, rebalance, threshold / 2))
    return out


def event_scores(funding: pd.DataFrame, lookback: int) -> pd.DataFrame:
    # Funding events are sparse hourly observations. Rolling is over event count per symbol.
    scores = pd.DataFrame(index=funding.index, columns=funding.columns, dtype=float)
    for symbol in funding.columns:
        observed = funding[symbol].dropna()
        rolled = observed.rolling(lookback, min_periods=lookback).mean()
        scores.loc[rolled.index, symbol] = rolled
    return scores


def build_weights(funding: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    scores = event_scores(funding, cfg.lookback_events)
    weights = pd.DataFrame(np.nan, index=funding.index, columns=SYMBOLS)
    held: List[str] = []
    event_number = 0
    threshold = cfg.threshold_bps / 10000.0
    keep_threshold = cfg.keep_bps / 10000.0

    event_times = funding.index[funding.notna().any(axis=1)]
    for time_ in event_times:
        if event_number % cfg.rebalance_events != 0:
            event_number += 1
            continue
        row = scores.loc[time_].dropna().sort_values(ascending=False)

        # Hysteresis: existing holdings survive at a lower threshold, reducing turnover.
        survivors = [s for s in held if s in row.index and row[s] >= keep_threshold]
        eligible = [s for s in row.index if row[s] >= threshold and s not in survivors]
        ranked = survivors + eligible
        ranked = sorted(ranked, key=lambda s: row[s], reverse=True)[: cfg.top_k]
        held = ranked

        w = pd.Series(0.0, index=SYMBOLS)
        if held:
            w.loc[held] = 1.0 / len(held)
        weights.loc[time_] = w
        event_number += 1

    # A decision made after the event is applied from the next hourly bar.
    return weights.ffill().fillna(0.0).shift(1).fillna(0.0)


def strategy_returns(
    spot: Dict[str, pd.DataFrame],
    perp: Dict[str, pd.DataFrame],
    funding: pd.DataFrame,
    weights: pd.DataFrame,
    pair_one_way_cost: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    spot_ret = pd.DataFrame({s: spot[s]["close"].pct_change().fillna(0.0) for s in SYMBOLS})
    perp_ret = pd.DataFrame({s: perp[s]["close"].pct_change().fillna(0.0) for s in SYMBOLS})

    basis_pnl = (weights * (spot_ret - perp_ret)).sum(axis=1)
    funding_pnl = (weights * funding.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * pair_one_way_cost
    net = basis_pnl + funding_pnl - costs
    return net.clip(lower=-0.95), basis_pnl, funding_pnl, turnover


def metrics(
    returns: pd.Series,
    basis_pnl: pd.Series,
    funding_pnl: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
) -> Dict[str, float]:
    r = returns.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / HOURS_YEAR
    monthly = (1 + r).resample("ME").prod() - 1
    dd = eq / eq.cummax() - 1
    vol = r.std() * math.sqrt(HOURS_YEAR)
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else -1.0
    active = weights.abs().sum(axis=1) > 0
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(r.mean() * HOURS_YEAR / vol) if vol > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "avg_month": float(monthly.mean()),
        "median_month": float(monthly.median()),
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "positive_month_pct": float((monthly > 0).mean()),
        "month_ge_10_pct": float((monthly >= 0.10).mean()),
        "month_le_minus10_pct": float((monthly <= -0.10).mean()),
        "funding_return_sum": float(funding_pnl.sum()),
        "basis_return_sum": float(basis_pnl.sum()),
        "annual_turnover": float(turnover.mean() * HOURS_YEAR),
        "active_time_pct": float(active.mean()),
        "months": int(len(monthly)),
    }


def train_score(row: pd.Series) -> float:
    if not np.isfinite(row.get("sharpe", np.nan)) or row["cagr"] <= 0:
        return -1e9
    if row["max_drawdown"] < -0.35:
        return -1e9
    return (
        2.0 * row["sharpe"]
        + 1.0 * row["positive_month_pct"]
        + 0.5 * row["cagr"]
        + 1.0 * row["max_drawdown"]
        - 0.0005 * row["annual_turnover"]
    )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    spot, perp, funding = load_all()
    index = funding.index
    print(f"Common timeline {index.min()} to {index.max()}, {len(index):,} hours")
    print("Funding observations", int(funding.notna().sum().sum()))

    rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]] = {}

    scan = configs()
    for number, cfg in enumerate(scan, 1):
        print(f"[{number}/{len(scan)}] {cfg.name}")
        weights = build_weights(funding, cfg)
        for cost_name, cost in COST_PROFILES.items():
            net, basis, fund, turnover = strategy_returns(spot, perp, funding, weights, cost)
            cache[(cfg.name, cost_name)] = (net, basis, fund, turnover, weights)
            for split, selector in [
                ("train", net.index <= TRAIN_END),
                ("test", net.index >= TEST_START),
                ("full", np.ones(len(net), dtype=bool)),
            ]:
                row: Dict[str, object] = {
                    "candidate": cfg.name,
                    "family": cfg.family,
                    "lookback_events": cfg.lookback_events,
                    "top_k": cfg.top_k,
                    "threshold_bps": cfg.threshold_bps,
                    "rebalance_events": cfg.rebalance_events,
                    "keep_bps": cfg.keep_bps,
                    "cost_profile": cost_name,
                    "split": split,
                }
                row.update(metrics(
                    net.loc[selector], basis.loc[selector], fund.loc[selector],
                    turnover.loc[selector], weights.loc[selector]
                ))
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "all_candidates.csv", index=False)

    selected_rows: List[Dict[str, object]] = []
    for cost_name in COST_PROFILES:
        train = results[(results.cost_profile == cost_name) & (results.split == "train")].copy()
        train["score"] = train.apply(train_score, axis=1)
        best = train.sort_values("score", ascending=False).iloc[0]
        test = results[
            (results.candidate == best.candidate)
            & (results.cost_profile == cost_name)
            & (results.split == "test")
        ].iloc[0].to_dict()
        test["train_score"] = float(best.score)
        test["train_cagr"] = float(best.cagr)
        test["train_sharpe"] = float(best.sharpe)
        selected_rows.append(test)

    selected = pd.DataFrame(selected_rows)
    selected["passes_oos"] = (
        (selected.cagr > 0)
        & (selected.sharpe >= 1.0)
        & (selected.max_drawdown >= -0.20)
        & (selected.positive_month_pct >= 0.55)
        & (selected.avg_month > 0)
    )
    selected.to_csv(OUTDIR / "selected_oos.csv", index=False)

    # Parameter robustness: count all OOS-positive variants, not used for selecting live rules.
    test_all = results[results.split == "test"].copy()
    robustness = test_all.groupby("cost_profile").agg(
        candidates=("candidate", "count"),
        positive_cagr=("cagr", lambda x: int((x > 0).sum())),
        sharpe_ge_1=("sharpe", lambda x: int((x >= 1).sum())),
        drawdown_better_20=("max_drawdown", lambda x: int((x >= -0.20).sum())),
        median_cagr=("cagr", "median"),
        median_sharpe=("sharpe", "median"),
    ).reset_index()
    robustness.to_csv(OUTDIR / "robustness.csv", index=False)

    cols = [
        "candidate", "cost_profile", "passes_oos", "cagr", "sharpe",
        "max_drawdown", "avg_month", "worst_month", "positive_month_pct",
        "month_ge_10_pct", "funding_return_sum", "basis_return_sum",
        "annual_turnover", "active_time_pct",
    ]
    lines = [
        "# Wave 3 — delta-neutral funding carry",
        "",
        f"- Data: {index.min()} to {index.max()}",
        f"- Symbols: {', '.join(SYMBOLS)}",
        "- Structure: long spot + short equal-notional perpetual",
        "- Signal uses only funding events already settled; position starts next hour",
        "- Parameter selection: 2023–2024; strict OOS: 2025–2026-07",
        "- Taker pair cost: 0.20% per entry or exit; 0.40% complete round trip",
        "",
        "## Train-selected model tested out of sample",
        "",
        selected[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OOS parameter robustness",
        "",
        robustness.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pass rule",
        "",
        "OOS CAGR > 0, Sharpe >= 1, max drawdown >= -20%, positive months >= 55%, average month > 0.",
        "This backtest excludes exchange default, withdrawal freezes, spot custody risk, margin-liquidation mechanics, taxes and cross-exchange transfer delays.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
