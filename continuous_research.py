from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUT = Path("results_continuous")
STATE_PATH = OUT / "state.json"
STATUS_PATH = OUT / "status.json"
LEADERBOARD_PATH = OUT / "leaderboard.csv"
REPORT_PATH = OUT / "latest_report.md"
ALERT_PATH = OUT / "ALERT.json"

START_MONTH = "2023-01"
END_MONTH = "2026-07"
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59", tz="UTC")
VAL_START = pd.Timestamp("2024-07-01", tz="UTC")
VAL_END = pd.Timestamp("2025-06-30 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2025-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-07-31 23:59:59", tz="UTC")

PRIMARY_SYMBOLS = ["BTCUSDT", "SOLUSDT"]
EXTERNAL_SYMBOLS = ["ETHUSDT", "BNBUSDT", "XRPUSDT"]
ONE_WAY_COST = 0.0008
BARS_PER_YEAR_4H = 6 * 365.25
MIN_TRAIN_TRADES = 20
MIN_VAL_TRADES = 15
MIN_TEST_TRADES = 20


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


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"next_batch": 0, "strict_passes": []}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def prepare_symbol(symbol: str) -> pd.DataFrame:
    k = w3.load_kline(symbol, "perp")
    f = w3.load_funding(symbol)
    bars = k.resample("4h", origin="start_day").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
    ).dropna()
    bars["funding"] = f.resample("4h", origin="start_day").sum().reindex(bars.index).fillna(0.0)

    prev = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev).abs(),
            (bars["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars["atr_pct"] = tr.rolling(21).mean() / bars["close"]
    bars["atr_rank126"] = bars["atr_pct"].rolling(126).rank(pct=True)
    bars["ema50"] = bars["close"].ewm(span=50, adjust=False).mean()
    bars["ema200"] = bars["close"].ewm(span=200, adjust=False).mean()
    bars["vol_med42"] = bars["volume"].rolling(42).median()
    bars["vol_ratio"] = bars["volume"] / bars["vol_med42"].replace(0, np.nan)
    bars["funding_mean6"] = bars["funding"].rolling(6).mean()
    fm = bars["funding"].rolling(126).mean()
    fs = bars["funding"].rolling(126).std().replace(0, np.nan)
    bars["funding_z"] = (bars["funding"] - fm) / fs
    for lb in [3, 6, 12, 18, 30, 48]:
        bars[f"ret{lb}"] = bars["close"].pct_change(lb)
        bars[f"prior_high{lb}"] = bars["high"].rolling(lb).max().shift(1)
        bars[f"prior_low{lb}"] = bars["low"].rolling(lb).min().shift(1)
    return bars.loc[(bars.index >= pd.Timestamp("2023-01-01", tz="UTC")) & (bars.index <= TEST_END)].copy()


def config_space(batch: int, symbol: str) -> List[Config]:
    rng = np.random.default_rng(20260808 + batch * 1009 + (0 if symbol == "BTCUSDT" else 97))
    families = [
        "trend",
        "compression_breakout",
        "funding_reversal",
        "funding_trend",
        "exhaustion",
        "regime_switch",
        "volume_breakout",
        "failed_breakout",
    ]
    focus = families[batch % len(families)]
    chosen = [focus, families[(batch + 3) % len(families)], families[(batch + 5) % len(families)]]
    out: List[Config] = []
    for family in chosen:
        for _ in range(18):
            lb = int(rng.choice([3, 6, 12, 18, 30, 48]))
            hold = int(rng.choice([1, 2, 3, 6, 9, 12]))
            side = str(rng.choice(["long", "short", "both"]))
            threshold = float(rng.choice([0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05]))
            aux = float(rng.choice([0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]))
            stop = float(rng.choice([0.0, 0.012, 0.018, 0.025, 0.035]))
            target = float(rng.choice([0.0, 0.018, 0.03, 0.045, 0.06]))
            out.append(Config(family, symbol, side, lb, hold, threshold, aux, stop, target))
    seen = set()
    deduped = []
    for cfg in out:
        if cfg.key not in seen:
            seen.add(cfg.key)
            deduped.append(cfg)
    return deduped


def make_signal(b: pd.DataFrame, cfg: Config) -> pd.Series:
    lb = cfg.lookback
    ret = b[f"ret{lb}"]
    ph = b[f"prior_high{lb}"]
    pl = b[f"prior_low{lb}"]
    uptrend = b["close"] > b["ema200"]
    downtrend = b["close"] < b["ema200"]
    strong_vol = b["vol_ratio"] >= max(cfg.aux, 1.0)
    funding_pos = b["funding_z"] >= cfg.aux
    funding_neg = b["funding_z"] <= -cfg.aux
    compressed = b["atr_rank126"] <= min(max(cfg.aux, 0.15), 0.75)
    high_vol = b["atr_rank126"] >= min(max(cfg.aux, 0.5), 0.95)

    long = pd.Series(False, index=b.index)
    short = pd.Series(False, index=b.index)

    if cfg.family == "trend":
        long = uptrend & (ret >= cfg.threshold)
        short = downtrend & (ret <= -cfg.threshold)
    elif cfg.family == "compression_breakout":
        long = compressed & (b["close"] > ph) & uptrend
        short = compressed & (b["close"] < pl) & downtrend
    elif cfg.family == "funding_reversal":
        long = funding_neg & (ret <= -cfg.threshold)
        short = funding_pos & (ret >= cfg.threshold)
    elif cfg.family == "funding_trend":
        long = funding_pos & uptrend & (ret >= cfg.threshold)
        short = funding_neg & downtrend & (ret <= -cfg.threshold)
    elif cfg.family == "exhaustion":
        long = high_vol & strong_vol & (ret <= -cfg.threshold)
        short = high_vol & strong_vol & (ret >= cfg.threshold)
    elif cfg.family == "regime_switch":
        long = (high_vol & uptrend & (ret >= cfg.threshold)) | (compressed & (ret <= -cfg.threshold))
        short = (high_vol & downtrend & (ret <= -cfg.threshold)) | (compressed & (ret >= cfg.threshold))
    elif cfg.family == "volume_breakout":
        long = strong_vol & (b["close"] > ph) & (ret >= 0)
        short = strong_vol & (b["close"] < pl) & (ret <= 0)
    elif cfg.family == "failed_breakout":
        prev_hi = b["high"].shift(1) > ph.shift(1)
        prev_lo = b["low"].shift(1) < pl.shift(1)
        long = prev_lo & (b["close"] > pl) & (ret > -cfg.threshold)
        short = prev_hi & (b["close"] < ph) & (ret < cfg.threshold)

    sig = pd.Series(0, index=b.index, dtype=int)
    if cfg.side in ("long", "both"):
        sig.loc[long.fillna(False)] = 1
    if cfg.side in ("short", "both"):
        sig.loc[short.fillna(False)] = -1
    return sig


def simulate(b: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sig = make_signal(b, cfg).to_numpy(int)
    idx = b.index
    op = b["open"].to_numpy(float)
    hi = b["high"].to_numpy(float)
    lo = b["low"].to_numpy(float)
    funding = b["funding"].to_numpy(float)

    recs: List[dict] = []
    i = 0
    while i < len(b) - cfg.hold - 2:
        if sig[i] == 0:
            i += 1
            continue
        side = int(sig[i])
        entry_i = i + 1
        if not np.isfinite(op[entry_i]) or op[entry_i] <= 0:
            i += 1
            continue
        entry = float(op[entry_i])
        planned_exit = min(entry_i + cfg.hold, len(b) - 1)
        exit_i = planned_exit
        exit_price = float(op[planned_exit])
        reason = "time"

        stop_price = None
        target_price = None
        if cfg.stop > 0:
            stop_price = entry * (1 - cfg.stop if side > 0 else 1 + cfg.stop)
        if cfg.target > 0:
            target_price = entry * (1 + cfg.target if side > 0 else 1 - cfg.target)

        for j in range(entry_i, planned_exit + 1):
            stop_hit = False
            target_hit = False
            if stop_price is not None:
                stop_hit = lo[j] <= stop_price if side > 0 else hi[j] >= stop_price
            if target_price is not None:
                target_hit = hi[j] >= target_price if side > 0 else lo[j] <= target_price
            if stop_hit:
                exit_i, exit_price, reason = j, float(stop_price), "stop"
                break
            if target_hit:
                exit_i, exit_price, reason = j, float(target_price), "target"
                break

        gross = side * (exit_price / entry - 1.0)
        fsum = float(np.nansum(funding[entry_i : exit_i + 1]))
        funding_pnl = -side * fsum
        net = gross + funding_pnl - 2 * ONE_WAY_COST
        recs.append(
            {
                "entry_time": idx[entry_i],
                "exit_time": idx[exit_i],
                "side": side,
                "entry": entry,
                "exit": exit_price,
                "gross": gross,
                "funding_pnl": funding_pnl,
                "net": net,
                "reason": reason,
            }
        )
        i = max(exit_i, i + 1)
    return pd.DataFrame(recs)


def bootstrap5(r: np.ndarray, seed: int) -> float:
    if len(r) < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(2500, len(r)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05))


def metrics(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, extra_roundtrip_cost: float = 0.0, seed: int = 1) -> dict:
    if trades.empty:
        return {}
    t = trades[(trades["entry_time"] >= start) & (trades["entry_time"] <= end)].copy()
    if t.empty:
        return {}
    r = t["net"].to_numpy(float) - extra_roundtrip_cost
    eq = np.cumprod(1.0 + r)
    years = max((end - start).total_seconds() / (365.25 * 86400), 0.05)
    cagr = float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else -1.0
    peak = np.maximum.accumulate(eq)
    dd = float(np.min(eq / peak - 1))
    wins = r[r > 0]
    losses = r[r < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) else float("inf")
    yearly = pd.DataFrame({"year": t["entry_time"].dt.year.to_numpy(), "r": r}).groupby("year")["r"].apply(lambda z: np.prod(1 + z) - 1)
    return {
        "trades": int(len(r)),
        "cagr": cagr,
        "total_return": float(eq[-1] - 1),
        "avg_trade": float(np.mean(r)),
        "win_rate": float(np.mean(r > 0)),
        "profit_factor": pf,
        "max_drawdown": dd,
        "bootstrap_mean_5pct": bootstrap5(r, seed),
        "positive_year_pct": float(np.mean(yearly.to_numpy() > 0)) if len(yearly) else float("nan"),
    }


def row_for_config(cfg: Config, trades: pd.DataFrame, seed: int) -> dict:
    tr = metrics(trades, pd.Timestamp("2023-01-01", tz="UTC"), TRAIN_END, seed=seed)
    va = metrics(trades, VAL_START, VAL_END, seed=seed + 1)
    te = metrics(trades, TEST_START, TEST_END, seed=seed + 2)
    stress = metrics(trades, TEST_START, TEST_END, extra_roundtrip_cost=2 * ONE_WAY_COST, seed=seed + 3)
    row = {**asdict(cfg), "key": cfg.key}
    for prefix, d in [("train", tr), ("val", va), ("test", te), ("stress", stress)]:
        for k, v in d.items():
            row[f"{prefix}_{k}"] = v
    return row


def basic_gate(row: dict) -> bool:
    return bool(
        row.get("train_trades", 0) >= MIN_TRAIN_TRADES
        and row.get("val_trades", 0) >= MIN_VAL_TRADES
        and row.get("train_avg_trade", -1) > 0
        and row.get("val_avg_trade", -1) > 0
        and row.get("val_profit_factor", 0) > 1.05
    )


def strict_gate(row: dict) -> bool:
    return bool(
        basic_gate(row)
        and row.get("test_trades", 0) >= MIN_TEST_TRADES
        and row.get("test_cagr", -1) >= 0.08
        and row.get("test_avg_trade", -1) > 0
        and row.get("test_profit_factor", 0) >= 1.20
        and row.get("test_max_drawdown", -1) >= -0.25
        and row.get("test_bootstrap_mean_5pct", -1) > 0
        and row.get("stress_cagr", -1) >= 0.08
        and row.get("stress_profit_factor", 0) >= 1.05
    )


def config_from_row(row: dict, symbol: str) -> Config:
    return Config(
        family=str(row["family"]),
        symbol=symbol,
        side=str(row["side"]),
        lookback=int(row["lookback"]),
        hold=int(row["hold"]),
        threshold=float(row["threshold"]),
        aux=float(row["aux"]),
        stop=float(row["stop"]),
        target=float(row["target"]),
    )


def external_confirmation(row: dict, external_data: Dict[str, pd.DataFrame], seed: int) -> dict:
    rows = []
    for i, (symbol, bars) in enumerate(external_data.items()):
        cfg = config_from_row(row, symbol)
        trades = simulate(bars, cfg)
        m = metrics(trades, TEST_START, TEST_END, seed=seed + i * 17)
        s = metrics(trades, TEST_START, TEST_END, extra_roundtrip_cost=2 * ONE_WAY_COST, seed=seed + i * 17 + 1)
        rows.append({"symbol": symbol, **m, **{f"stress_{k}": v for k, v in s.items()}})
    ext = pd.DataFrame(rows)
    valid = ext[ext["trades"].fillna(0) >= 10].copy() if not ext.empty else ext
    positive = int((valid["avg_trade"] > 0).sum()) if not valid.empty else 0
    pf_med = float(valid["profit_factor"].median()) if not valid.empty else float("nan")
    boot_positive = int((valid["bootstrap_mean_5pct"] > 0).sum()) if not valid.empty else 0
    stress_positive = int((valid["stress_avg_trade"] > 0).sum()) if not valid.empty else 0
    passed = bool(len(valid) >= 2 and positive >= 2 and pf_med > 1.05 and boot_positive >= 2 and stress_positive >= 2)
    return {
        "passed": passed,
        "valid_symbols": int(len(valid)),
        "positive_symbols": positive,
        "median_pf": pf_med,
        "bootstrap_positive_symbols": boot_positive,
        "stress_positive_symbols": stress_positive,
        "details": rows,
    }


def merge_leaderboard(current: pd.DataFrame) -> pd.DataFrame:
    if LEADERBOARD_PATH.exists():
        try:
            old = pd.read_csv(LEADERBOARD_PATH)
            merged = pd.concat([old, current], ignore_index=True)
        except Exception:
            merged = current.copy()
    else:
        merged = current.copy()
    merged = merged.drop_duplicates(subset=["key"], keep="last")
    for c in ["val_profit_factor", "val_cagr", "test_profit_factor", "test_cagr"]:
        if c not in merged:
            merged[c] = np.nan
    merged["rank_score"] = (
        merged["val_profit_factor"].clip(upper=3).fillna(0)
        + 0.5 * merged["val_cagr"].clip(lower=-1, upper=2).fillna(-1)
        + 0.25 * merged["test_profit_factor"].clip(upper=3).fillna(0)
    )
    merged = merged.sort_values("rank_score", ascending=False).head(300)
    merged.to_csv(LEADERBOARD_PATH, index=False)
    return merged


def cleanup_old_batches(keep: int = 168) -> None:
    files = sorted(OUT.glob("batch_*.csv"))
    for p in files[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass


def main() -> None:
    OUT.mkdir(exist_ok=True)
    state = load_state()
    batch = int(state.get("next_batch", 0))
    run_started = pd.Timestamp.now(tz="UTC")
    save_json(STATUS_PATH, {"status": "running", "batch": batch, "started_at": run_started.isoformat(), "stage": "loading_primary_data"})

    w3.START_MONTH = START_MONTH
    w3.END_MONTH = END_MONTH
    w3.INTERVAL = "1h"
    w3.CACHE = Path(".cache_continuous")

    primary: Dict[str, pd.DataFrame] = {}
    for symbol in PRIMARY_SYMBOLS:
        primary[symbol] = prepare_symbol(symbol)

    rows: List[dict] = []
    tested = 0
    for symbol, bars in primary.items():
        for j, cfg in enumerate(config_space(batch, symbol)):
            trades = simulate(bars, cfg)
            row = row_for_config(cfg, trades, seed=20260808 + batch * 1000 + j)
            row["batch"] = batch
            row["basic_gate"] = basic_gate(row)
            row["strict_gate_pre_external"] = strict_gate(row)
            rows.append(row)
            tested += 1

    batch_df = pd.DataFrame(rows)
    batch_path = OUT / f"batch_{batch:05d}.csv"
    batch_df.to_csv(batch_path, index=False)
    leaderboard = merge_leaderboard(batch_df)

    prelim = batch_df[batch_df["strict_gate_pre_external"] == True].copy()
    strict_passes = []
    external_checked = 0
    if not prelim.empty:
        save_json(STATUS_PATH, {"status": "running", "batch": batch, "started_at": run_started.isoformat(), "stage": "external_confirmation", "tested": tested, "preliminary_candidates": int(len(prelim))})
        external_data = {s: prepare_symbol(s) for s in EXTERNAL_SYMBOLS}
        for k, (_, row) in enumerate(prelim.sort_values("test_profit_factor", ascending=False).head(3).iterrows()):
            external_checked += 1
            ext = external_confirmation(row.to_dict(), external_data, seed=20260808 + batch * 100 + k)
            if ext["passed"]:
                payload = {
                    "detected_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "batch": batch,
                    "config": {k2: row[k2] for k2 in ["key", "family", "symbol", "side", "lookback", "hold", "threshold", "aux", "stop", "target"]},
                    "test": {c.replace("test_", ""): row[c] for c in row.index if c.startswith("test_")},
                    "stress": {c.replace("stress_", ""): row[c] for c in row.index if c.startswith("stress_")},
                    "external_confirmation": ext,
                }
                strict_passes.append(payload)

    if strict_passes:
        save_json(ALERT_PATH, {"strict_passes": strict_passes})
        state.setdefault("strict_passes", []).extend(strict_passes)
        state["strict_passes"] = state["strict_passes"][-20:]

    state["next_batch"] = batch + 1
    state["last_completed_batch"] = batch
    state["last_completed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    save_json(STATE_PATH, state)
    cleanup_old_batches()

    top = leaderboard.head(10).copy()
    cols = [
        "key", "family", "symbol", "side", "lookback", "hold", "threshold", "aux", "stop", "target",
        "train_trades", "train_profit_factor", "train_cagr",
        "val_trades", "val_profit_factor", "val_cagr",
        "test_trades", "test_profit_factor", "test_cagr", "test_max_drawdown", "test_bootstrap_mean_5pct",
        "stress_profit_factor", "stress_cagr",
    ]
    cols = [c for c in cols if c in top.columns]
    report = "\n".join([
        "# Continuous crypto research",
        "",
        f"- Batch: {batch}",
        f"- Completed: {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"- Tested this batch: {tested}",
        f"- Basic-gate survivors: {int(batch_df['basic_gate'].sum())}",
        f"- Pre-external strict candidates: {int(batch_df['strict_gate_pre_external'].sum())}",
        f"- External confirmations checked: {external_checked}",
        f"- Strict passes this batch: {len(strict_passes)}",
        "",
        "## Leaderboard (research only; not a trading recommendation)",
        "",
        top[cols].to_markdown(index=False, floatfmt=".4f"),
    ])
    REPORT_PATH.write_text(report, encoding="utf-8")
    save_json(STATUS_PATH, {
        "status": "completed",
        "batch": batch,
        "started_at": run_started.isoformat(),
        "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "tested": tested,
        "basic_gate_survivors": int(batch_df["basic_gate"].sum()),
        "pre_external_candidates": int(batch_df["strict_gate_pre_external"].sum()),
        "external_checked": external_checked,
        "strict_passes": len(strict_passes),
        "alert_file": ALERT_PATH.exists(),
    })
    print(report)


if __name__ == "__main__":
    main()
