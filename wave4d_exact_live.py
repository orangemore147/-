from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from wave4c_live_signal import TICKERS, TRADE_TICKERS, fetch_market, yahoo_current

OUTDIR = Path("results_wave4d")
DAYS_YEAR = 252
LOOKBACK = 126
TOP_K = 3
REBALANCE = 5
REGIME_DAYS = 200
PROFILES = {
    "unlevered": {"target_vol": None, "max_leverage": 1.0},
    "base": {"target_vol": 0.20, "max_leverage": 1.5},
    "balanced": {"target_vol": 0.35, "max_leverage": 2.0},
}


def build_exact_weights(close: pd.DataFrame) -> pd.DataFrame:
    scores = close[TRADE_TICKERS].pct_change(LOOKBACK)
    regime = close["SPY"] > close["SPY"].rolling(REGIME_DAYS, min_periods=REGIME_DAYS).mean()
    raw = pd.DataFrame(np.nan, index=close.index, columns=TRADE_TICKERS)
    for i in range(len(close.index)):
        if i % REBALANCE != 0:
            continue
        signal = pd.Series(0.0, index=TRADE_TICKERS)
        if bool(regime.iloc[i]):
            row = scores.iloc[i].dropna()
            selected = row[row > 0].nlargest(TOP_K)
            if len(selected):
                signal.loc[selected.index] = 1.0 / len(selected)
        raw.iloc[i] = signal
    # Completed-close signal becomes effective from the next session.
    return raw.ffill().fillna(0.0).shift(1).fillna(0.0)


def profile_leverage(unlevered: pd.Series, target_vol: float | None, cap: float) -> pd.Series:
    if target_vol is None:
        return pd.Series(1.0, index=unlevered.index)
    rolling = unlevered.rolling(63, min_periods=21).std() * math.sqrt(DAYS_YEAR)
    return (target_vol / rolling.replace(0, np.nan)).clip(0, cap).shift(1).fillna(0.0)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    frames = {ticker: yahoo_current(ticker) for ticker in TICKERS}
    common = frames["SPY"].index.intersection(frames["QQQ"].index).sort_values()
    close = pd.DataFrame(index=common)
    for ticker, frame in frames.items():
        close[ticker] = frame["adjclose"].reindex(common).ffill(limit=3)

    weights = build_exact_weights(close)
    returns = close[TRADE_TICKERS].pct_change().fillna(0.0)
    unlevered = (weights * returns).sum(axis=1)
    latest = close.index[-1]
    active_weights = weights.loc[latest]
    selected = active_weights[active_weights > 0].index.tolist()

    effective_rows = weights.ne(weights.shift(1)).any(axis=1)
    change_dates = weights.index[effective_rows]
    last_effective_date = change_dates[change_dates <= latest][-1] if len(change_dates) else None
    next_rebalance_index = next((i for i in range(len(close.index), len(close.index) + REBALANCE + 1) if i % REBALANCE == 0), None)
    sessions_until_next_signal = (next_rebalance_index - len(close.index)) if next_rebalance_index is not None else None

    profiles: Dict[str, Dict] = {}
    for name, cfg in PROFILES.items():
        leverage_series = profile_leverage(unlevered, cfg["target_vol"], cfg["max_leverage"])
        lev = float(leverage_series.loc[latest])
        profiles[name] = {
            "gross_notional_as_account_equity": lev * float(active_weights.abs().sum()),
            "notional_per_selected_stock_as_account_equity": lev / len(selected) if selected else 0.0,
            "estimated_annualized_target_volatility": cfg["target_vol"],
            "maximum_leverage_cap": cfg["max_leverage"],
        }

    market_rows = []
    errors = []
    for ticker in selected:
        try:
            row = fetch_market(ticker + "USDT")
            row["ticker"] = ticker
            row["underlying_adjusted_close"] = float(close.loc[latest, ticker])
            row["momentum_126d_at_latest"] = float(close[ticker].pct_change(LOOKBACK).loc[latest])
            market_rows.append(row)
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})

    result = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "latest_underlying_close_date": latest.isoformat(),
        "last_effective_portfolio_change_date": last_effective_date.isoformat() if last_effective_date is not None else None,
        "sessions_until_next_scheduled_signal_calculation": sessions_until_next_signal,
        "selected_tickers": selected,
        "base_equal_weights": {ticker: float(active_weights[ticker]) for ticker in selected},
        "profiles": profiles,
        "market_data": market_rows,
        "errors": errors,
    }
    (OUTDIR / "exact_live.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    market = pd.DataFrame(market_rows)
    lines = [
        "# Wave 4D — exact live portfolio state",
        "",
        f"- Generated: {result['generated_at']}",
        f"- Latest underlying close: {result['latest_underlying_close_date']}",
        f"- Last effective portfolio change: {result['last_effective_portfolio_change_date']}",
        f"- Active names: {', '.join(selected) if selected else 'CASH'}",
        f"- Sessions until next scheduled signal: {sessions_until_next_signal}",
        "",
        "## Position profiles",
        "",
        pd.DataFrame(profiles).T.reset_index(names="profile").to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Bitget market and funding data",
        "",
        market.to_markdown(index=False, floatfmt=".6f") if len(market) else "No active positions.",
        "",
        "## Errors",
        "",
        f"```json\n{json.dumps(errors, indent=2)}\n```",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
