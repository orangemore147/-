from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import wave3_funding_carry as w3
import wave5_academic_strategies as w5

OUT = Path("results_wave6")
STOCK_TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
STOCK_TEST_START = pd.Timestamp("2023-01-01", tz="UTC")
CRYPTO_TRAIN_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
CRYPTO_TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
STOCK_ONE_WAY_COST = 0.00042 + 0.00030
CRYPTO_ONE_WAY_COST = 0.00060 + 0.00020


@dataclass(frozen=True)
class Config:
    market: str
    family: str
    side: int
    lookback: int
    aux: float
    tp: float
    sl: float
    max_hold: int

    @property
    def name(self) -> str:
        side = "L" if self.side == 1 else "S"
        return (
            f"{self.market}_{self.family}_{side}_lb{self.lookback}_a{self.aux:g}"
            f"_tp{self.tp:.3f}_sl{self.sl:.3f}_h{self.max_hold}"
        )


@dataclass
class Trade:
    symbol: str
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    score: float
    bars: int = 0
    funding: float = 0.0


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def stock_data() -> Dict[str, pd.DataFrame]:
    raw = w5.load_all()
    output: Dict[str, pd.DataFrame] = {}
    for symbol, frame in raw.items():
        df = frame.copy()
        df["sma50"] = df["close"].rolling(50).mean()
        df["sma200"] = df["close"].rolling(200).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
        df["rsi3"] = rsi(df["close"], 3)
        df["ret5"] = df["close"].pct_change(5)
        df["ret126"] = df["close"].pct_change(126)
        df["vol21"] = df["close"].pct_change().rolling(21).std()
        for lookback in (20, 55):
            df[f"prior_high_{lookback}"] = df["high"].rolling(lookback).max().shift(1)
        output[symbol] = df
    return output


def crypto_data() -> Dict[str, pd.DataFrame]:
    w3.START_MONTH = "2022-01"
    w3.END_MONTH = "2026-07"
    w3.CACHE = Path(".cache_wave6_crypto")
    output: Dict[str, pd.DataFrame] = {}
    for symbol in w3.SYMBOLS:
        print("Loading crypto", symbol)
        bars = w3.load_kline(symbol, "perp")
        funding = w3.load_funding(symbol)
        bars = bars.resample("4h", origin="start_day").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}
        ).dropna()
        bars["funding_event"] = funding.resample("4h", origin="start_day").sum().reindex(bars.index).fillna(0.0)
        bars["ema50"] = bars["close"].ewm(span=50, adjust=False).mean()
        bars["ema200"] = bars["close"].ewm(span=200, adjust=False).mean()
        bars["rsi7"] = rsi(bars["close"], 7)
        bars["ret42"] = bars["close"].pct_change(42)
        bars["vol42"] = bars["close"].pct_change().rolling(42).std()
        bars["volume_ratio"] = bars["quote_volume"] / bars["quote_volume"].rolling(42).median()
        tr = pd.concat(
            [bars["high"] - bars["low"], (bars["high"] - bars["close"].shift()).abs(), (bars["low"] - bars["close"].shift()).abs()],
            axis=1,
        ).max(axis=1)
        bars["atr_pct"] = tr.rolling(21).mean() / bars["close"]
        bars["atr_rank126"] = bars["atr_pct"].rolling(126).rank(pct=True)
        for lookback in (6, 18):
            bars[f"prior_high_{lookback}"] = bars["high"].rolling(lookback).max().shift(1)
            bars[f"prior_low_{lookback}"] = bars["low"].rolling(lookback).min().shift(1)
        output[symbol] = bars
    start = max(frame.index.min() for frame in output.values())
    end = min(frame.index.max() for frame in output.values())
    return {symbol: frame.loc[start:end].copy() for symbol, frame in output.items()}


def stock_signal(cfg: Config, row: pd.Series, spy: pd.Series) -> float | None:
    if pd.isna(spy["sma200"]) or spy["close"] <= spy["sma200"]:
        return None
    vol = max(float(row.get("vol21", np.nan)), 1e-8)
    if cfg.family == "breakout":
        high = row.get(f"prior_high_{cfg.lookback}")
        if pd.notna(high) and row["close"] > high and row["close"] > row["sma200"] and row["ret126"] > 0:
            return float(row["ret126"] / vol)
    elif cfg.family == "pullback":
        if row["close"] > row["sma200"] and row["sma50"] > row["sma200"] and row["rsi3"] <= cfg.aux and row["ret126"] > 0:
            return float((cfg.aux - row["rsi3"] + 1) * row["ret126"] / vol)
    elif cfg.family == "trend_pullback":
        if row["ema20"] > row["ema100"] > row["sma200"] and row["ret5"] <= -cfg.aux and row["ret126"] > 0:
            return float((-row["ret5"] + row["ret126"]) / vol)
    return None


def crypto_signal(cfg: Config, row: pd.Series, btc: pd.Series) -> float | None:
    bull = btc["close"] > btc["ema200"]
    bear = btc["close"] < btc["ema200"]
    vol = max(float(row.get("vol42", np.nan)), 1e-8)
    if cfg.family == "breakout":
        level = row.get(f"prior_high_{cfg.lookback}" if cfg.side == 1 else f"prior_low_{cfg.lookback}")
        direction = row["close"] > level if cfg.side == 1 else row["close"] < level
        regime = bull if cfg.side == 1 else bear
        trend = row["close"] > row["ema200"] if cfg.side == 1 else row["close"] < row["ema200"]
        if pd.notna(level) and regime and direction and trend and row["volume_ratio"] >= cfg.aux:
            return float(max(cfg.side * row["ret42"], 0) / vol)
    elif cfg.family == "pullback":
        if cfg.side == 1 and bull and row["close"] > row["ema200"] and row["rsi7"] <= cfg.aux and row["ret42"] > 0:
            return float((cfg.aux - row["rsi7"] + 1) * row["ret42"] / vol)
        if cfg.side == -1 and bear and row["close"] < row["ema200"] and row["rsi7"] >= 100 - cfg.aux and row["ret42"] < 0:
            return float((row["rsi7"] - 100 + cfg.aux + 1) * -row["ret42"] / vol)
    elif cfg.family == "squeeze":
        level = row.get(f"prior_high_{cfg.lookback}" if cfg.side == 1 else f"prior_low_{cfg.lookback}")
        direction = row["close"] > level if cfg.side == 1 else row["close"] < level
        regime = bull if cfg.side == 1 else bear
        if pd.notna(level) and regime and direction and row["atr_rank126"] <= cfg.aux:
            return float((cfg.aux - row["atr_rank126"] + 0.01) / max(row["atr_pct"], 1e-8))
    return None


def close(active: Trade, time_: pd.Timestamp, price: float, reason: str, cost: float) -> dict:
    gross = active.side * (price / active.entry_price - 1)
    return {
        "symbol": active.symbol,
        "side": "long" if active.side == 1 else "short",
        "entry_time": active.entry_time,
        "exit_time": time_,
        "entry_price": active.entry_price,
        "exit_price": price,
        "bars": active.bars,
        "gross_return": gross,
        "funding_cost": active.funding,
        "net_return": gross - 2 * cost - active.funding,
        "reason": reason,
    }


def simulate(cfg: Config, data: Dict[str, pd.DataFrame], regime_symbol: str, cost: float) -> pd.DataFrame:
    times = sorted(set().union(*(frame.index for frame in data.values())))
    active: Trade | None = None
    records: List[dict] = []
    for i, time_ in enumerate(times):
        if active is not None and time_ in data[active.symbol].index:
            row = data[active.symbol].loc[time_]
            active.bars += 1
            if cfg.market == "crypto":
                active.funding += active.side * float(row.get("funding_event", 0.0))
            stop_hit = row["low"] <= active.stop if active.side == 1 else row["high"] >= active.stop
            target_hit = row["high"] >= active.target if active.side == 1 else row["low"] <= active.target
            if stop_hit:
                records.append(close(active, time_, active.stop, "stop", cost)); active = None; continue
            if target_hit:
                records.append(close(active, time_, active.target, "target", cost)); active = None; continue
            if active.bars >= cfg.max_hold:
                records.append(close(active, time_, float(row["close"]), "time", cost)); active = None; continue
        if active is not None or i == 0:
            continue
        previous = times[i - 1]
        if previous not in data[regime_symbol].index:
            continue
        regime = data[regime_symbol].loc[previous]
        candidates: List[Tuple[float, str, pd.Series]] = []
        for symbol, frame in data.items():
            if cfg.market == "stock" and symbol == regime_symbol:
                continue
            if previous not in frame.index or time_ not in frame.index:
                continue
            score = stock_signal(cfg, frame.loc[previous], regime) if cfg.market == "stock" else crypto_signal(cfg, frame.loc[previous], regime)
            if score is not None and np.isfinite(score):
                candidates.append((score, symbol, frame.loc[time_]))
        if not candidates:
            continue
        score, symbol, entry_row = max(candidates, key=lambda item: item[0])
        entry = float(entry_row["open"])
        stop = entry * (1 - cfg.sl) if cfg.side == 1 else entry * (1 + cfg.sl)
        target = entry * (1 + cfg.tp) if cfg.side == 1 else entry * (1 - cfg.tp)
        active = Trade(symbol, cfg.side, time_, entry, stop, target, score)
    if active is not None:
        last = data[active.symbol].index[-1]
        records.append(close(active, last, float(data[active.symbol].iloc[-1]["close"]), "end", cost))
    output = pd.DataFrame(records)
    if not output.empty:
        output["entry_time"] = pd.to_datetime(output["entry_time"], utc=True)
        output["exit_time"] = pd.to_datetime(output["exit_time"], utc=True)
    return output


def bootstrap_lower(values: pd.Series, seed: int = 42) -> float:
    if len(values) < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    source = values.to_numpy(float)
    means = [rng.choice(source, len(source), replace=True).mean() for _ in range(3000)]
    return float(np.quantile(means, 0.05))


def metrics(trades: pd.DataFrame, extra_cost: float = 0.0) -> dict:
    if trades.empty:
        return {key: 0.0 for key in ["trades", "total_return", "avg_trade", "win_rate", "avg_win", "median_win", "avg_loss", "profit_factor", "max_drawdown", "target_hit_rate", "winner_ge_2_pct", "positive_year_pct"]} | {"bootstrap_mean_5pct": float("nan")}
    returns = trades["net_return"].astype(float) - extra_cost
    wins, losses = returns[returns > 0], returns[returns < 0]
    equity = (1 + returns).cumprod()
    yearly = pd.DataFrame({"year": trades["entry_time"].dt.year, "r": returns}).groupby("year")["r"].apply(lambda x: (1 + x).prod() - 1)
    return {
        "trades": int(len(trades)),
        "total_return": float(equity.iloc[-1] - 1),
        "avg_trade": float(returns.mean()),
        "win_rate": float((returns > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "median_win": float(wins.median()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "target_hit_rate": float((trades["reason"] == "target").mean()),
        "winner_ge_2_pct": float((wins >= 0.02).mean()) if len(wins) else 0.0,
        "bootstrap_mean_5pct": bootstrap_lower(returns),
        "positive_year_pct": float((yearly > 0).mean()),
    }


def grid() -> List[Config]:
    output: List[Config] = []
    for tp in (0.022, 0.030):
        for sl in (0.012, 0.018):
            for hold in (5, 10, 20):
                for lookback in (20, 55):
                    output.append(Config("stock", "breakout", 1, lookback, 0, tp, sl, hold))
                for threshold in (10, 20):
                    output.append(Config("stock", "pullback", 1, 0, threshold, tp, sl, hold))
                for dip in (0.01, 0.02):
                    output.append(Config("stock", "trend_pullback", 1, 0, dip, tp, sl, hold))
    for tp in (0.022, 0.030):
        for sl in (0.012, 0.018):
            for hold in (6, 12, 24):
                for side in (1, -1):
                    for lookback in (6, 18):
                        for volume in (1.0, 1.5):
                            output.append(Config("crypto", "breakout", side, lookback, volume, tp, sl, hold))
                    for threshold in (25, 35):
                        output.append(Config("crypto", "pullback", side, 0, threshold, tp, sl, hold))
                    for lookback in (6, 18):
                        for rank in (0.15, 0.25):
                            output.append(Config("crypto", "squeeze", side, lookback, rank, tp, sl, hold))
    return output


def train_score(row: pd.Series) -> float:
    if row["trades"] < 40 or row["profit_factor"] < 1.05 or row["total_return"] <= 0 or row["max_drawdown"] < -0.45:
        return -1e9
    return 800 * row["avg_trade"] + 1.5 * math.log(max(row["profit_factor"], 1e-8)) + row["target_hit_rate"] + row["winner_ge_2_pct"] + row["max_drawdown"]


def strict_pass(row: pd.Series) -> bool:
    minimum = 60 if row["market"] == "crypto" else 35
    return bool(
        row["trades"] >= minimum and row["total_return"] > 0 and row["avg_trade"] > 0
        and row["profit_factor"] >= 1.25 and row["max_drawdown"] >= -0.25
        and row["median_win"] >= 0.02 and row["winner_ge_2_pct"] >= 0.80
        and row["bootstrap_mean_5pct"] > 0 and row["positive_year_pct"] >= 0.67
        and row["stress_profit_factor"] >= 1.05 and row["stress_avg_trade"] > 0
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    stocks, crypto = stock_data(), crypto_data()
    rows, trade_map = [], {}
    configs = grid()
    for index, cfg in enumerate(configs, 1):
        if index == 1 or index % 50 == 0:
            print(f"Testing {index}/{len(configs)} {cfg.name}")
        data, regime, cost = (stocks, "SPY", STOCK_ONE_WAY_COST) if cfg.market == "stock" else (crypto, "BTCUSDT", CRYPTO_ONE_WAY_COST)
        trades = simulate(cfg, data, regime, cost)
        train_end, test_start = (STOCK_TRAIN_END, STOCK_TEST_START) if cfg.market == "stock" else (CRYPTO_TRAIN_END, CRYPTO_TEST_START)
        train = trades[trades["entry_time"] <= train_end] if not trades.empty else trades
        test = trades[trades["entry_time"] >= test_start] if not trades.empty else trades
        train_metrics, test_metrics = metrics(train), metrics(test)
        train_stress, test_stress = metrics(train, cost), metrics(test, cost)
        train_metrics.update({"stress_" + key: value for key, value in train_stress.items()})
        test_metrics.update({"stress_" + key: value for key, value in test_stress.items()})
        base = asdict(cfg) | {"name": cfg.name}
        rows += [base | {"split": "train"} | train_metrics, base | {"split": "test"} | test_metrics]
        trade_map[cfg.name] = trades

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "all_results.csv", index=False)
    selected = []
    for market in results["market"].unique():
        for family in results.loc[results["market"] == market, "family"].unique():
            train = results[(results["market"] == market) & (results["family"] == family) & (results["split"] == "train")].copy()
            train["score"] = train.apply(train_score, axis=1)
            best = train.sort_values("score", ascending=False).iloc[0]
            if best["score"] <= -1e8:
                continue
            selected.append(results[(results["name"] == best["name"]) & (results["split"] == "test")].iloc[0].to_dict())
    selected = pd.DataFrame(selected)
    selected["passes_strict"] = selected.apply(strict_pass, axis=1) if not selected.empty else False
    selected.to_csv(OUT / "selected_oos.csv", index=False)

    passing = selected[selected["passes_strict"] == True] if not selected.empty else selected
    columns = ["name", "market", "family", "trades", "total_return", "avg_trade", "win_rate", "avg_win", "median_win", "avg_loss", "profit_factor", "max_drawdown", "target_hit_rate", "winner_ge_2_pct", "positive_year_pct", "bootstrap_mean_5pct", "stress_profit_factor", "stress_avg_trade", "passes_strict"]
    report = [
        "# Wave 6 — net 2% winner-target strategy audit", "",
        "Literal certification that every future trade earns 2% is impossible. This audit tests fixed gross TP >=2.2%, median OOS winner >=2% net, at least 80% of OOS winners >=2% net, and positive expectancy after costs.", "",
        "- Stock train 2014–2022; OOS 2023–2026-08.",
        "- Crypto train 2022–2024; OOS 2025–2026-07.",
        "- Next-bar-open entries; same-bar stop before target; one position at a time.",
        f"- Stock one-way cost {STOCK_ONE_WAY_COST:.3%}; crypto one-way cost {CRYPTO_ONE_WAY_COST:.3%}; crypto funding included.",
        "- Strict pass also requires PF>=1.25, max DD>=-25%, bootstrap lower mean>0, >=67% positive years, and 1.5x-cost stress remains positive.", "",
        "## Train-selected models tested out of sample", "",
        selected[columns].sort_values(["passes_strict", "profit_factor"], ascending=[False, False]).to_markdown(index=False, floatfmt=".4f") if not selected.empty else "No selectable strategy.", "",
        f"Strict pass count: **{len(passing)}**.", "",
        "PASS is reproducible historical evidence, not a guarantee or third-party certification.",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps({"strict_pass_count": int(len(passing)), "passing": passing.to_dict(orient="records")}, indent=2, default=str), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
