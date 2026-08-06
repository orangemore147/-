from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import wave4_equities as w4

# Long-history names that Bitget listed as supported stock perps in 2026.
TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "INTC", "ORCL", "IBM", "CSCO", "PEP", "MCD", "GE", "MA", "BABA", "LLY", "UNH", "ASML",
]
TRADE_TICKERS = [ticker for ticker in TICKERS if ticker not in {"SPY", "QQQ"}]
OUTDIR = Path("results_wave4b")
TRAIN_END = w4.TRAIN_END
TEST_START = w4.TEST_START
DAYS_YEAR = w4.DAYS_YEAR
ONE_WAY_COST = w4.ONE_WAY_COST

# Annualized drag scenarios applied to average gross notional.
# These are sensitivity tests, not claims about actual future funding.
FUNDING_DRAGS = [0.00, 0.05, 0.10, 0.20, 0.40]
PROFILES = {
    "unlevered": (None, 1.0),
    "base": (0.20, 1.5),
    "balanced": (0.35, 2.0),
}


def align_outer(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Use the benchmark timeline. Individual stocks may have occasional missing rows.
    index = data["SPY"].index.intersection(data["QQQ"].index).sort_values()
    result: Dict[str, pd.DataFrame] = {}
    for ticker, frame in data.items():
        result[ticker] = frame.reindex(index).ffill(limit=3)
    return result


def normalized(row: pd.Series, top_k: int) -> pd.Series:
    row = row.replace([np.inf, -np.inf], np.nan).dropna()
    row = row[row > 0].nlargest(top_k)
    out = pd.Series(0.0, index=TRADE_TICKERS)
    if len(row):
        out.loc[row.index] = 1.0 / len(row)
    return out


def top_momentum_weights(
    data: Dict[str, pd.DataFrame], lookback: int, top_k: int, rebalance: int, regime_days: int
) -> pd.DataFrame:
    index = data["SPY"].index
    close = pd.DataFrame({ticker: data[ticker]["adjclose"] for ticker in TICKERS}, index=index)
    scores = close[TRADE_TICKERS].pct_change(lookback)
    regime = close["SPY"] > close["SPY"].rolling(regime_days, min_periods=regime_days).mean()
    raw = pd.DataFrame(np.nan, index=index, columns=TRADE_TICKERS)
    for i in range(len(index)):
        if i % rebalance != 0:
            continue
        if bool(regime.iloc[i]):
            raw.iloc[i] = normalized(scores.iloc[i], top_k)
        else:
            raw.iloc[i] = 0.0
    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def inverse_vol_weights(
    data: Dict[str, pd.DataFrame], regime_days: int, vol_days: int, rebalance: int
) -> pd.DataFrame:
    index = data["SPY"].index
    close = pd.DataFrame({ticker: data[ticker]["adjclose"] for ticker in TICKERS}, index=index)
    regime = close["SPY"] > close["SPY"].rolling(regime_days, min_periods=regime_days).mean()
    vol = close[TRADE_TICKERS].pct_change().rolling(vol_days, min_periods=vol_days).std()
    raw = pd.DataFrame(np.nan, index=index, columns=TRADE_TICKERS)
    for i in range(len(index)):
        if i % rebalance != 0:
            continue
        signal = pd.Series(0.0, index=TRADE_TICKERS)
        if bool(regime.iloc[i]):
            inv = (1 / vol.iloc[i].replace(0, np.nan)).dropna()
            if len(inv):
                signal.loc[inv.index] = inv / inv.sum()
        raw.iloc[i] = signal
    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def dual_ema_weights(data: Dict[str, pd.DataFrame], fast: int, slow: int) -> pd.DataFrame:
    index = data["SPY"].index
    close = pd.DataFrame({ticker: data[ticker]["adjclose"] for ticker in TICKERS}, index=index)
    regime = close["QQQ"] > close["QQQ"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    weights = pd.DataFrame(0.0, index=index, columns=TRADE_TICKERS)
    for ticker in TRADE_TICKERS:
        ef = close[ticker].ewm(span=fast, adjust=False, min_periods=slow).mean()
        es = close[ticker].ewm(span=slow, adjust=False, min_periods=slow).mean()
        weights[ticker] = (regime & (ef > es)).astype(float) / len(TRADE_TICKERS)
    return weights.shift(1).fillna(0.0)


def compute_returns(
    data: Dict[str, pd.DataFrame], base: pd.DataFrame, target_vol: float | None, cap: float
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    returns = pd.DataFrame(
        {ticker: data[ticker]["adjclose"].pct_change().fillna(0.0) for ticker in TRADE_TICKERS},
        index=base.index,
    )
    unlevered = (base * returns).sum(axis=1)
    if target_vol is None:
        leverage = pd.Series(1.0, index=base.index)
    else:
        rolling = unlevered.rolling(63, min_periods=21).std() * math.sqrt(DAYS_YEAR)
        leverage = (target_vol / rolling.replace(0, np.nan)).clip(0, cap).shift(1).fillna(0.0)
    actual = base.mul(leverage, axis=0)
    gross = (actual * returns).sum(axis=1)
    turnover = actual.diff().abs().sum(axis=1).fillna(actual.abs().sum(axis=1))
    return gross - turnover * ONE_WAY_COST, actual, turnover


def metrics(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / DAYS_YEAR
    monthly = (1 + r).resample("ME").prod() - 1
    drawdown = eq / eq.cummax() - 1
    vol = r.std() * math.sqrt(DAYS_YEAR)
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else -1.0,
        "sharpe": float(r.mean() * DAYS_YEAR / vol) if vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "avg_month": float(monthly.mean()),
        "worst_month": float(monthly.min()),
        "positive_month_pct": float((monthly > 0).mean()),
        "month_ge_10_pct": float((monthly >= 0.10).mean()),
        "month_le_minus10_pct": float((monthly <= -0.10).mean()),
        "months": int(len(monthly)),
    }


def candidate_grid() -> List[Tuple[str, pd.DataFrame]]:
    raise RuntimeError("built after data load")


def train_score(m: Dict[str, float]) -> float:
    if not np.isfinite(m.get("sharpe", np.nan)) or m["cagr"] <= 0 or m["max_drawdown"] < -0.50:
        return -1e9
    return 1.8 * m["sharpe"] + 0.6 * m["positive_month_pct"] + 0.4 * m["cagr"] + 0.8 * m["max_drawdown"]


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    w4.TICKERS = TICKERS
    data = align_outer({ticker: w4.yahoo_chart(ticker) for ticker in TICKERS})
    index = data["SPY"].index
    print(f"Data {index.min()} to {index.max()}, {len(index)} sessions; universe {len(TRADE_TICKERS)}")

    candidates: List[Tuple[str, str, pd.DataFrame]] = []
    for lookback in [21, 63, 126, 252]:
        for top_k in [3, 5, 8, 10]:
            for rebalance in [5, 21, 63]:
                name = f"topmom_{lookback}_{top_k}_{rebalance}_200"
                candidates.append((name, "top_momentum", top_momentum_weights(data, lookback, top_k, rebalance, 200)))
    for regime in [100, 150, 200]:
        for vol_days in [21, 63]:
            for rebalance in [5, 21]:
                name = f"invvol_{regime}_{vol_days}_{rebalance}"
                candidates.append((name, "inverse_vol", inverse_vol_weights(data, regime, vol_days, rebalance)))
    for fast in [20, 50]:
        for slow in [100, 150, 200]:
            if fast < slow:
                name = f"dualema_{fast}_{slow}"
                candidates.append((name, "dual_ema", dual_ema_weights(data, fast, slow)))

    rows: List[Dict[str, object]] = []
    return_cache: Dict[Tuple[str, str], Tuple[pd.Series, pd.DataFrame, pd.Series]] = {}
    for number, (name, family, base) in enumerate(candidates, 1):
        print(f"[{number}/{len(candidates)}] {name}")
        for profile, (target, cap) in PROFILES.items():
            net, actual, turnover = compute_returns(data, base, target, cap)
            return_cache[(name, profile)] = (net, actual, turnover)
            train = net.loc[:TRAIN_END]
            row = {"candidate": name, "family": family, "profile": profile, "split": "train"}
            row.update(metrics(train))
            rows.append(row)
            test = net.loc[TEST_START:]
            row = {"candidate": name, "family": family, "profile": profile, "split": "test"}
            row.update(metrics(test))
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "all_candidates.csv", index=False)

    selected_rows: List[Dict[str, object]] = []
    selected_returns: Dict[Tuple[str, str], Tuple[pd.Series, pd.DataFrame]] = {}
    for profile in PROFILES:
        train = results[(results.profile == profile) & (results.split == "train")]
        for family in sorted(train.family.unique()):
            family_train = train[train.family == family].copy()
            family_train["score"] = family_train.apply(lambda row: train_score(row.to_dict()), axis=1)
            best = family_train.sort_values("score", ascending=False).iloc[0]
            test = results[
                (results.candidate == best.candidate)
                & (results.profile == profile)
                & (results.split == "test")
            ].iloc[0].to_dict()
            test["train_score"] = float(best.score)
            selected_rows.append(test)
            net, actual, _ = return_cache[(str(best.candidate), profile)]
            selected_returns[(family, profile)] = (net.loc[TEST_START:], actual.loc[TEST_START:])

    selected = pd.DataFrame(selected_rows)
    selected["passes_oos"] = (
        (selected.cagr > 0)
        & (selected.sharpe >= 0.75)
        & (selected.max_drawdown >= -0.35)
        & (selected.positive_month_pct >= 0.55)
    )
    selected.to_csv(OUTDIR / "selected_oos.csv", index=False)

    sensitivity_rows: List[Dict[str, object]] = []
    yearly_rows: List[Dict[str, object]] = []
    for (family, profile), (net, actual) in selected_returns.items():
        gross_exposure = actual.abs().sum(axis=1)
        for annual_drag in FUNDING_DRAGS:
            adjusted = net - gross_exposure * annual_drag / DAYS_YEAR
            m = metrics(adjusted)
            sensitivity_rows.append({
                "family": family,
                "profile": profile,
                "annual_funding_drag": annual_drag,
                "avg_gross_exposure": float(gross_exposure.mean()),
                **m,
            })
        for year, group in net.groupby(net.index.year):
            m = metrics(group)
            yearly_rows.append({"family": family, "profile": profile, "year": int(year), **m})

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUTDIR / "funding_sensitivity.csv", index=False)
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(OUTDIR / "yearly_oos.csv", index=False)

    # Leave-one-out robustness for the train-selected top-momentum configuration per profile.
    loo_rows: List[Dict[str, object]] = []
    for profile in PROFILES:
        best_row = selected[(selected.profile == profile) & (selected.family == "top_momentum")].iloc[0]
        parts = str(best_row.candidate).split("_")
        lookback, top_k, rebalance, regime = map(int, parts[1:])
        for excluded in TRADE_TICKERS:
            original = TRADE_TICKERS.copy()
            try:
                globals()["TRADE_TICKERS"] = [ticker for ticker in original if ticker != excluded]
                base = top_momentum_weights(data, lookback, min(top_k, len(TRADE_TICKERS)), rebalance, regime)
                target, cap = PROFILES[profile]
                net, _, _ = compute_returns(data, base, target, cap)
                m = metrics(net.loc[TEST_START:])
                loo_rows.append({"profile": profile, "excluded": excluded, **m})
            finally:
                globals()["TRADE_TICKERS"] = original
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(OUTDIR / "leave_one_out.csv", index=False)

    cols = [
        "candidate", "family", "profile", "passes_oos", "cagr", "sharpe",
        "max_drawdown", "avg_month", "worst_month", "positive_month_pct", "month_ge_10_pct",
    ]
    sens_display = sensitivity[
        (sensitivity.family == "top_momentum")
        & (sensitivity.annual_funding_drag.isin([0.0, 0.10, 0.20, 0.40]))
    ]
    loo_summary = loo.groupby("profile").agg(
        tests=("excluded", "count"),
        min_cagr=("cagr", "min"),
        median_cagr=("cagr", "median"),
        min_sharpe=("sharpe", "min"),
        worst_drawdown=("max_drawdown", "min"),
        positive_cases=("cagr", lambda x: int((x > 0).sum())),
    ).reset_index()

    lines = [
        "# Wave 4B — stock strategy robustness",
        "",
        f"- Data: {index.min()} to {index.max()}",
        f"- Broad Bitget-supported long-history universe: {', '.join(TRADE_TICKERS)}",
        "- Train: 2015–2022; strict OOS: 2023–2026-07",
        "- Trading cost: 0.08% per one-way notional change",
        "",
        "## Train-selected models — OOS",
        "",
        selected.sort_values(["passes_oos", "sharpe"], ascending=[False, False])[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top-momentum funding-drag sensitivity",
        "",
        sens_display[["family", "profile", "annual_funding_drag", "cagr", "sharpe", "max_drawdown", "avg_month", "positive_month_pct", "month_ge_10_pct"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top-momentum leave-one-stock-out summary",
        "",
        loo_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Caveats",
        "",
        "Funding drag scenarios are sensitivities, not observed Bitget funding history. The universe is broader but still uses securities that survived and remained listed through 2026. Corporate-action adjusted underlying prices may differ from stock-perp mark-price behavior outside U.S. market hours.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
