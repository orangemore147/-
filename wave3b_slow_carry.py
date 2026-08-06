from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import wave3_funding_carry as w3

OUTDIR = Path("results_wave3b")
TRAIN_END = w3.TRAIN_END
TEST_START = w3.TEST_START
HOURS_YEAR = w3.HOURS_YEAR


def fixed_weights(index: pd.DatetimeIndex, names: List[str]) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=w3.SYMBOLS)
    weights.loc[:, names] = 1.0 / len(names)
    # Enter one hour after start to avoid assuming a pre-sample position.
    return weights.shift(1).fillna(0.0)


def slow_rank_weights(
    funding: pd.DataFrame,
    lookback_events: int,
    top_k: int,
    rebalance_days: int,
    threshold_bps: float,
) -> pd.DataFrame:
    scores = pd.DataFrame(index=funding.index, columns=funding.columns, dtype=float)
    for symbol in funding.columns:
        obs = funding[symbol].dropna()
        rolled = obs.rolling(lookback_events, min_periods=lookback_events).mean()
        scores.loc[rolled.index, symbol] = rolled
    # Carry last observed score so rebalancing can happen on a fixed calendar.
    scores = scores.ffill()

    weights = pd.DataFrame(np.nan, index=funding.index, columns=w3.SYMBOLS)
    start = funding.index[0].normalize()
    threshold = threshold_bps / 10000.0
    elapsed_days = ((funding.index.normalize() - start) / pd.Timedelta(days=1)).astype(int)
    rebalance_mask = np.asarray(elapsed_days % rebalance_days == 0) & (funding.index.hour == 0)

    for i in np.flatnonzero(rebalance_mask):
        row = scores.iloc[i].dropna().sort_values(ascending=False)
        selected = row[row >= threshold].head(top_k)
        current = pd.Series(0.0, index=w3.SYMBOLS)
        if len(selected):
            current.loc[selected.index] = 1.0 / len(selected)
        weights.iloc[i] = current
    return weights.ffill().fillna(0.0).shift(1).fillna(0.0)


def returns_with_final_exit(
    spot: Dict[str, pd.DataFrame],
    perp: Dict[str, pd.DataFrame],
    funding: pd.DataFrame,
    weights: pd.DataFrame,
    cost: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    net, basis, fund, turnover = w3.strategy_returns(spot, perp, funding, weights, cost)
    if len(net):
        final_exit = float(weights.iloc[-1].abs().sum()) * cost
        net.iloc[-1] -= final_exit
        turnover.iloc[-1] += float(weights.iloc[-1].abs().sum())
    return net, basis, fund, turnover


def evaluate(
    net: pd.Series,
    basis: pd.Series,
    fund: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
) -> Dict[str, float]:
    return w3.metrics(net, basis, fund, turnover, weights)


def train_score(row: pd.Series) -> float:
    if not np.isfinite(row.get("sharpe", np.nan)) or row["cagr"] <= 0:
        return -1e9
    if row["max_drawdown"] < -0.25:
        return -1e9
    return (
        2.0 * row["sharpe"]
        + row["positive_month_pct"]
        + 0.5 * row["cagr"]
        + row["max_drawdown"]
        - 0.0002 * row["annual_turnover"]
    )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    spot, perp, funding = w3.load_all()
    index = funding.index
    candidates: List[Tuple[str, str, pd.DataFrame]] = []

    fixed_sets = {
        "BTC": ["BTCUSDT"],
        "ETH": ["ETHUSDT"],
        "SOL": ["SOLUSDT"],
        "BTC_ETH": ["BTCUSDT", "ETHUSDT"],
        "MAJORS3": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "TOP5": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        "ALL10": w3.SYMBOLS,
    }
    for name, names in fixed_sets.items():
        candidates.append((f"fixed_{name}", "fixed", fixed_weights(index, names)))

    for lookback in [30, 90, 180]:
        for top_k in [1, 3, 5, 10]:
            for days in [7, 14, 30, 60, 90]:
                for threshold in [0.0, 0.5, 1.0]:
                    name = f"slowrank_lb{lookback}_k{top_k}_d{days}_th{threshold:g}"
                    weights = slow_rank_weights(funding, lookback, top_k, days, threshold)
                    candidates.append((name, "slow_rank", weights))

    rows: List[Dict[str, object]] = []
    for number, (name, family, weights) in enumerate(candidates, 1):
        print(f"[{number}/{len(candidates)}] {name}")
        for cost_name, cost in w3.COST_PROFILES.items():
            net, basis, fund, turnover = returns_with_final_exit(
                spot, perp, funding, weights, cost
            )
            for split, selector in [
                ("train", net.index <= TRAIN_END),
                ("test", net.index >= TEST_START),
                ("full", np.ones(len(net), dtype=bool)),
            ]:
                row: Dict[str, object] = {
                    "candidate": name,
                    "family": family,
                    "cost_profile": cost_name,
                    "split": split,
                }
                row.update(evaluate(
                    net.loc[selector], basis.loc[selector], fund.loc[selector],
                    turnover.loc[selector], weights.loc[selector]
                ))
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "all_candidates.csv", index=False)

    selected_rows: List[Dict[str, object]] = []
    for cost_name in w3.COST_PROFILES:
        train = results[(results.cost_profile == cost_name) & (results.split == "train")].copy()
        train["score"] = train.apply(train_score, axis=1)
        for family in sorted(train.family.unique()):
            best = train[train.family == family].sort_values("score", ascending=False).iloc[0]
            test = results[
                (results.candidate == best.candidate)
                & (results.cost_profile == cost_name)
                & (results.split == "test")
            ].iloc[0].to_dict()
            test["train_cagr"] = float(best.cagr)
            test["train_sharpe"] = float(best.sharpe)
            test["train_score"] = float(best.score)
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

    test = results[results.split == "test"].copy()
    top = test.sort_values(["sharpe", "cagr"], ascending=False).head(20)
    cols = [
        "candidate", "family", "cost_profile", "cagr", "sharpe", "max_drawdown",
        "avg_month", "worst_month", "positive_month_pct", "month_ge_10_pct",
        "funding_return_sum", "basis_return_sum", "annual_turnover", "active_time_pct",
    ]
    lines = [
        "# Wave 3B — static and slow funding carry",
        "",
        f"- Data: {index.min()} to {index.max()}",
        "- Long spot / short equal-notional perpetual",
        "- Includes initial entry and final exit costs",
        "- Train selection: 2023–2024; OOS: 2025–2026-07",
        "",
        "## Train-selected models — OOS",
        "",
        selected[cols + ["passes_oos"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top 20 OOS results (diagnostic only; not valid for model selection)",
        "",
        top[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pass rule",
        "",
        "CAGR > 0, Sharpe >= 1, max drawdown >= -20%, positive months >= 55%, average month > 0.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
