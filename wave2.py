from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import backtest as b
from run_fixed import load_symbol_fixed

b.load_symbol = load_symbol_fixed

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT",
]
OUTDIR = Path("results_wave2")
COST = 0.0008
HOURS_YEAR = 24 * 365
TRAIN_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

RISK_PROFILES = {
    "conservative": (0.25, 1.5),
    "balanced": (0.50, 2.5),
    "aggressive": (0.80, 3.0),
}


@dataclass(frozen=True)
class Candidate:
    family: str
    params: Tuple[int, ...]

    @property
    def name(self) -> str:
        return self.family + "_" + "-".join(map(str, self.params))


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rebalance_mask(index: pd.DatetimeIndex, hours: int) -> np.ndarray:
    anchor = index[0]
    elapsed = ((index - anchor) / pd.Timedelta(hours=1)).astype(int)
    return np.asarray(elapsed % hours == 0)


def normalise_long(row: pd.Series, max_names: int | None = None) -> pd.Series:
    row = row.clip(lower=0).fillna(0.0)
    if max_names is not None and (row > 0).sum() > max_names:
        keep = row.nlargest(max_names).index
        row.loc[~row.index.isin(keep)] = 0.0
    total = row.sum()
    return row / total if total > 0 else row * 0.0


def stateful_donchian(close: pd.Series, entry_hours: int, exit_hours: int, mask: np.ndarray) -> pd.Series:
    upper = close.rolling(entry_hours, min_periods=entry_hours).max().shift(1)
    lower = close.rolling(exit_hours, min_periods=exit_hours).min().shift(1)
    state = 0.0
    out = pd.Series(0.0, index=close.index)
    for i, t in enumerate(close.index):
        if mask[i]:
            if state == 0.0 and pd.notna(upper.iloc[i]) and close.iloc[i] > upper.iloc[i]:
                state = 1.0
            elif state == 1.0 and pd.notna(lower.iloc[i]) and close.iloc[i] < lower.iloc[i]:
                state = 0.0
        out.iloc[i] = state
    return out


def stateful_pullback(
    close: pd.Series,
    trend: pd.Series,
    rsi_value: pd.Series,
    entry_rsi: int,
    exit_rsi: int,
    mask: np.ndarray,
) -> pd.Series:
    state = 0.0
    out = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        if mask[i]:
            if state == 0.0 and close.iloc[i] > trend.iloc[i] and rsi_value.iloc[i] < entry_rsi:
                state = 1.0
            elif state == 1.0 and (rsi_value.iloc[i] > exit_rsi or close.iloc[i] < trend.iloc[i]):
                state = 0.0
        out.iloc[i] = state
    return out


def stateful_bollinger(
    close: pd.Series,
    z: pd.Series,
    btc_bull: pd.Series,
    entry_z: float,
    exit_z: float,
    mask: np.ndarray,
) -> pd.Series:
    state = 0.0
    out = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        if mask[i]:
            if state == 0.0 and bool(btc_bull.iloc[i]) and z.iloc[i] < entry_z:
                state = 1.0
            elif state == 1.0 and (z.iloc[i] > exit_z or not bool(btc_bull.iloc[i])):
                state = 0.0
        out.iloc[i] = state
    return out


def build_weights(data: Dict[str, pd.DataFrame], c: Candidate) -> pd.DataFrame:
    idx = next(iter(data.values())).index
    close = pd.DataFrame({s: data[s]["close"] for s in SYMBOLS}, index=idx)
    raw = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
    btc = close["BTCUSDT"]

    if c.family == "ema_long":
        (days,) = c.params
        span = days * 24
        mask = rebalance_mask(idx, 24)
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        for i in np.flatnonzero(mask):
            sig = (close.iloc[i] > ema.iloc[i]).astype(float)
            raw.iloc[i] = normalise_long(sig)

    elif c.family == "dual_ema_long":
        fast_days, slow_days = c.params
        mask = rebalance_mask(idx, 24)
        fast = close.ewm(span=fast_days * 24, adjust=False, min_periods=slow_days * 24).mean()
        slow = close.ewm(span=slow_days * 24, adjust=False, min_periods=slow_days * 24).mean()
        for i in np.flatnonzero(mask):
            sig = (fast.iloc[i] > slow.iloc[i]).astype(float)
            raw.iloc[i] = normalise_long(sig)

    elif c.family == "top_momentum":
        lookback_days, top_k, rebalance_days, regime_days = c.params
        mask = rebalance_mask(idx, 24 * rebalance_days)
        scores = close.pct_change(lookback_days * 24)
        btc_ema = btc.ewm(span=regime_days * 24, adjust=False, min_periods=regime_days * 24).mean()
        for i in np.flatnonzero(mask):
            if pd.isna(btc_ema.iloc[i]) or btc.iloc[i] <= btc_ema.iloc[i]:
                raw.iloc[i] = 0.0
                continue
            row = scores.iloc[i].replace([np.inf, -np.inf], np.nan).dropna()
            row = row[row > 0].nlargest(top_k)
            sig = pd.Series(0.0, index=SYMBOLS)
            sig.loc[row.index] = row.clip(lower=0)
            raw.iloc[i] = normalise_long(sig, max_names=top_k)

    elif c.family == "inverse_vol_basket":
        regime_days, vol_days = c.params
        mask = rebalance_mask(idx, 24 * 7)
        btc_ema = btc.ewm(span=regime_days * 24, adjust=False, min_periods=regime_days * 24).mean()
        vol = close.pct_change().rolling(vol_days * 24, min_periods=vol_days * 12).std()
        for i in np.flatnonzero(mask):
            if pd.isna(btc_ema.iloc[i]) or btc.iloc[i] <= btc_ema.iloc[i]:
                raw.iloc[i] = 0.0
                continue
            inv = 1 / vol.iloc[i].replace(0, np.nan)
            raw.iloc[i] = normalise_long(inv)

    elif c.family == "donchian_long":
        entry_bars, exit_bars = c.params
        mask = rebalance_mask(idx, 4)
        states = pd.DataFrame(index=idx, columns=SYMBOLS, dtype=float)
        for s in SYMBOLS:
            states[s] = stateful_donchian(close[s], entry_bars * 4, exit_bars * 4, mask)
        for i in np.flatnonzero(mask):
            raw.iloc[i] = normalise_long(states.iloc[i])

    elif c.family == "pullback_long":
        ema_bars, entry_rsi, exit_rsi = c.params
        mask = rebalance_mask(idx, 4)
        states = pd.DataFrame(index=idx, columns=SYMBOLS, dtype=float)
        for s in SYMBOLS:
            trend = close[s].ewm(span=ema_bars * 4, adjust=False, min_periods=ema_bars * 4).mean()
            rv = rsi(close[s], 14 * 4)
            states[s] = stateful_pullback(close[s], trend, rv, entry_rsi, exit_rsi, mask)
        for i in np.flatnonzero(mask):
            raw.iloc[i] = normalise_long(states.iloc[i])

    elif c.family == "bollinger_bull":
        window_days, entry_x10, exit_x10, regime_days = c.params
        mask = rebalance_mask(idx, 24)
        mean = close.rolling(window_days * 24, min_periods=window_days * 24).mean()
        std = close.rolling(window_days * 24, min_periods=window_days * 24).std()
        z = (close - mean) / std.replace(0, np.nan)
        btc_ema = btc.ewm(span=regime_days * 24, adjust=False, min_periods=regime_days * 24).mean()
        bull = btc > btc_ema
        states = pd.DataFrame(index=idx, columns=SYMBOLS, dtype=float)
        for s in SYMBOLS:
            states[s] = stateful_bollinger(
                close[s], z[s], bull, -entry_x10 / 10.0, exit_x10 / 10.0, mask
            )
        for i in np.flatnonzero(mask):
            raw.iloc[i] = normalise_long(states.iloc[i], max_names=3)

    elif c.family == "btc_rotation":
        lookback_days, rebalance_days, regime_days = c.params
        mask = rebalance_mask(idx, 24 * rebalance_days)
        subset = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        scores = close[subset].pct_change(lookback_days * 24)
        btc_ema = btc.ewm(span=regime_days * 24, adjust=False, min_periods=regime_days * 24).mean()
        for i in np.flatnonzero(mask):
            sig = pd.Series(0.0, index=SYMBOLS)
            if pd.notna(btc_ema.iloc[i]) and btc.iloc[i] > btc_ema.iloc[i]:
                row = scores.iloc[i].dropna()
                if len(row) and row.max() > 0:
                    sig.loc[row.idxmax()] = 1.0
            raw.iloc[i] = sig

    else:
        raise ValueError(c.family)

    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def apply_profile(
    data: Dict[str, pd.DataFrame], base_weights: pd.DataFrame, target_vol: float, cap: float
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    rets = pd.DataFrame({s: data[s]["close"].pct_change().fillna(0.0) for s in SYMBOLS})
    unlevered = (base_weights * rets).sum(axis=1)
    rolling = unlevered.rolling(24 * 30, min_periods=24 * 10).std() * math.sqrt(HOURS_YEAR)
    leverage = (target_vol / rolling.replace(0, np.nan)).clip(0, cap).shift(1).fillna(0.0)
    actual = base_weights.mul(leverage, axis=0)
    gross = (actual * rets).sum(axis=1)
    turnover = actual.diff().abs().sum(axis=1).fillna(actual.abs().sum(axis=1))
    net = gross - turnover * COST
    return net.clip(lower=-0.95), actual, turnover


def metrics(r: pd.Series, weights: pd.DataFrame, turnover: pd.Series) -> Dict[str, float]:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / HOURS_YEAR
    monthly = (1 + r).resample("ME").prod() - 1
    dd = eq / eq.cummax() - 1
    vol = r.std() * math.sqrt(HOURS_YEAR)
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else -1.0
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
        "months": int(len(monthly)),
        "avg_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "annual_turnover": float(turnover.mean() * HOURS_YEAR),
    }


def grid() -> List[Candidate]:
    out: List[Candidate] = []
    for d in [50, 100, 150, 200]:
        out.append(Candidate("ema_long", (d,)))
    for f in [20, 50, 100]:
        for s in [100, 150, 200]:
            if f < s:
                out.append(Candidate("dual_ema_long", (f, s)))
    for lb in [30, 60, 90, 180]:
        for k in [1, 2, 3]:
            for reb in [1, 7]:
                out.append(Candidate("top_momentum", (lb, k, reb, 200)))
    for regime in [100, 150, 200]:
        for vol in [20, 60]:
            out.append(Candidate("inverse_vol_basket", (regime, vol)))
    for entry in [20, 50, 100]:
        for exit_ in [10, 20, 50]:
            if exit_ < entry:
                out.append(Candidate("donchian_long", (entry, exit_)))
    for ema in [100, 200]:
        for ent in [30, 35, 40]:
            for ex in [55, 60, 65]:
                out.append(Candidate("pullback_long", (ema, ent, ex)))
    for window in [10, 20, 40]:
        for entry_x10 in [15, 20, 25]:
            for exit_x10 in [-5, 0, 5]:
                out.append(Candidate("bollinger_bull", (window, entry_x10, exit_x10, 200)))
    for lb in [30, 60, 90, 180]:
        for reb in [1, 7, 30]:
            out.append(Candidate("btc_rotation", (lb, reb, 200)))
    return out


def train_score(row: pd.Series) -> float:
    if not np.isfinite(row.get("sharpe", np.nan)):
        return -1e9
    if row["max_drawdown"] < -0.50 or row["cagr"] <= 0:
        return -1e9
    return (
        1.8 * row["sharpe"]
        + 0.8 * row["positive_month_pct"]
        + 0.5 * row["cagr"]
        + 0.8 * row["max_drawdown"]
        - 0.001 * row["annual_turnover"]
    )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    b.SYMBOLS = SYMBOLS
    print("Downloading and aligning 10 liquid perpetual contracts...")
    data = {s: b.load_symbol(s) for s in SYMBOLS}
    data = b.align_data(data)
    idx = next(iter(data.values())).index
    print(f"Data {idx.min()} to {idx.max()}, {len(idx):,} hourly bars")

    rows: List[Dict[str, object]] = []
    returns_cache: Dict[Tuple[str, str], pd.Series] = {}
    weights_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    turnover_cache: Dict[Tuple[str, str], pd.Series] = {}

    candidates = grid()
    for n, c in enumerate(candidates, 1):
        print(f"[{n}/{len(candidates)}] {c.name}")
        base = build_weights(data, c)
        for profile, (target, cap) in RISK_PROFILES.items():
            net, actual, turnover = apply_profile(data, base, target, cap)
            returns_cache[(c.name, profile)] = net
            weights_cache[(c.name, profile)] = actual
            turnover_cache[(c.name, profile)] = turnover
            for split, selector in [
                ("train", net.index <= TRAIN_END),
                ("test", net.index >= TEST_START),
                ("full", np.ones(len(net), dtype=bool)),
            ]:
                r = net.loc[selector]
                w = actual.loc[selector]
                t = turnover.loc[selector]
                row: Dict[str, object] = {
                    "candidate": c.name,
                    "family": c.family,
                    "params": json.dumps(c.params),
                    "profile": profile,
                    "split": split,
                }
                row.update(metrics(r, w, t))
                rows.append(row)

    all_results = pd.DataFrame(rows)
    all_results.to_csv(OUTDIR / "all_candidates.csv", index=False)

    selected_rows: List[Dict[str, object]] = []
    selected_series: Dict[Tuple[str, str], pd.Series] = {}
    for profile in RISK_PROFILES:
        train = all_results[(all_results.profile == profile) & (all_results.split == "train")].copy()
        train["score"] = train.apply(train_score, axis=1)
        for family in sorted(train.family.unique()):
            best = train[train.family == family].sort_values("score", ascending=False).iloc[0]
            name = str(best.candidate)
            test = all_results[
                (all_results.candidate == name)
                & (all_results.profile == profile)
                & (all_results.split == "test")
            ].iloc[0].to_dict()
            test["train_score"] = float(best.score)
            test["train_cagr"] = float(best.cagr)
            test["train_sharpe"] = float(best.sharpe)
            selected_rows.append(test)
            selected_series[(family, profile)] = returns_cache[(name, profile)]

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
        available = [v for (fam, p), v in selected_series.items() if p == profile]
        ensemble = pd.concat(available, axis=1).mean(axis=1)
        for split, series in [
            ("train", ensemble.loc[:TRAIN_END]),
            ("test", ensemble.loc[TEST_START:]),
            ("full", ensemble),
        ]:
            fake_w = pd.DataFrame({"ensemble": np.where(series != 0, 1.0, 0.0)}, index=series.index)
            fake_turn = pd.Series(0.0, index=series.index)
            row = {"profile": profile, "split": split}
            row.update(metrics(series, fake_w, fake_turn))
            ensemble_rows.append(row)
    ensembles = pd.DataFrame(ensemble_rows)
    ensembles.to_csv(OUTDIR / "ensembles.csv", index=False)

    oos = selected.sort_values(["passes_oos", "sharpe", "cagr"], ascending=[False, False, False])
    cols = [
        "candidate", "family", "profile", "passes_oos", "cagr", "sharpe",
        "max_drawdown", "avg_month", "worst_month", "positive_month_pct",
        "month_ge_10_pct", "month_le_minus10_pct", "annual_turnover",
    ]
    lines = [
        "# Wave 2 — long/cash and low-frequency crypto strategy scan",
        "",
        f"- Data: {idx.min()} to {idx.max()}",
        f"- Symbols: {', '.join(SYMBOLS)}",
        f"- One-way trading cost: {COST:.2%}",
        "- Parameter selection: 2023–2024 only",
        "- Strict out-of-sample: 2025–2026-07",
        "",
        "## Train-selected models, tested out of sample",
        "",
        oos[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Family ensemble",
        "",
        ensembles[ensembles.split == "test"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## OOS pass rule",
        "",
        "CAGR > 0; Sharpe >= 0.75; max drawdown >= -35%; positive months >= 55%; average month > 0.",
        "A 10% month is reported as a frequency, not treated as a guarantee.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
