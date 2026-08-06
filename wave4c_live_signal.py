from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "INTC", "ORCL", "IBM", "CSCO", "PEP", "MCD", "GE", "MA", "BABA", "LLY", "UNH", "ASML",
]
TRADE_TICKERS = [ticker for ticker in TICKERS if ticker not in {"SPY", "QQQ"}]
START = "2015-01-01"
OUTDIR = Path("results_wave4c")
BITGET = "https://api.bitget.com"


def yahoo_current(ticker: str) -> pd.DataFrame:
    p1 = int(pd.Timestamp(START, tz="UTC").timestamp())
    p2 = int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits"
    )
    response = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0 live-signal/1.0"})
    response.raise_for_status()
    payload = response.json()["chart"]["result"][0]
    quote = payload["indicators"]["quote"][0]
    adj = payload["indicators"].get("adjclose", [{}])[0].get("adjclose")
    timestamps = pd.to_datetime(payload["timestamp"], unit="s", utc=True).normalize()
    frame = pd.DataFrame({
        "time": timestamps,
        "close": quote["close"],
        "adjclose": adj if adj is not None else quote["close"],
    }).dropna(subset=["adjclose"])
    return frame.drop_duplicates("time").set_index("time").sort_index()


def bitget_json(path: str, params: Dict[str, str]) -> Dict:
    response = requests.get(
        BITGET + path,
        params=params,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 live-signal/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "00000":
        raise RuntimeError(f"Bitget error {payload}")
    return payload


def fetch_market(symbol: str) -> Dict:
    ticker = bitget_json(
        "/api/v2/mix/market/ticker",
        {"symbol": symbol, "productType": "usdt-futures"},
    )["data"][0]
    current = bitget_json(
        "/api/v2/mix/market/current-fund-rate",
        {"symbol": symbol, "productType": "usdt-futures"},
    )["data"][0]
    history = bitget_json(
        "/api/v2/mix/market/history-fund-rate",
        {"symbol": symbol, "productType": "usdt-futures", "pageSize": "100", "pageNo": "1"},
    )["data"]

    rates = pd.Series([float(row["fundingRate"]) for row in history], dtype=float)
    interval_hours = float(current.get("fundingRateInterval") or 8)
    settlements_per_year = 24 / interval_hours * 365
    mean_rate = float(rates.mean()) if len(rates) else float("nan")
    median_rate = float(rates.median()) if len(rates) else float("nan")
    positive_pct = float((rates > 0).mean()) if len(rates) else float("nan")
    annualized_simple = mean_rate * settlements_per_year

    bid = float(ticker.get("bidPr") or 0)
    ask = float(ticker.get("askPr") or 0)
    mark = float(ticker.get("markPrice") or 0)
    spread_pct = (ask - bid) / mark if mark > 0 and ask > 0 and bid > 0 else float("nan")

    return {
        "symbol": symbol,
        "mark_price": mark,
        "last_price": float(ticker.get("lastPr") or 0),
        "bid": bid,
        "ask": ask,
        "spread_pct": spread_pct,
        "current_funding_rate": float(current["fundingRate"]),
        "funding_interval_hours": interval_hours,
        "recent_funding_observations": int(len(rates)),
        "recent_mean_funding_rate": mean_rate,
        "recent_median_funding_rate": median_rate,
        "recent_positive_funding_pct": positive_pct,
        "recent_simple_annualized_long_funding_drag": annualized_simple,
        "next_funding_time": pd.to_datetime(int(current["nextUpdate"]), unit="ms", utc=True).isoformat(),
        "min_funding_rate": float(current["minFundingRate"]),
        "max_funding_rate": float(current["maxFundingRate"]),
        "quote_volume_24h": float(ticker.get("usdtVolume") or ticker.get("quoteVolume") or 0),
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    frames = {ticker: yahoo_current(ticker) for ticker in TICKERS}
    common = frames["SPY"].index.intersection(frames["QQQ"].index)
    close = pd.DataFrame(index=common)
    for ticker, frame in frames.items():
        close[ticker] = frame["adjclose"].reindex(common).ffill(limit=3)

    latest = close.dropna(subset=["SPY", "QQQ"]).index[-1]
    spy_ma200 = close["SPY"].rolling(200, min_periods=200).mean()
    regime_on = bool(close.loc[latest, "SPY"] > spy_ma200.loc[latest])
    momentum = close[TRADE_TICKERS].pct_change(126).loc[latest].dropna().sort_values(ascending=False)
    positive = momentum[momentum > 0]
    selected = positive.head(3).index.tolist() if regime_on else []

    # Weekly-ish five-session rebalance anchor mirrors robustness backtest.
    latest_pos = int(close.index.get_loc(latest))
    sessions_since_rebalance = latest_pos % 5
    next_rebalance_sessions = 0 if sessions_since_rebalance == 0 else 5 - sessions_since_rebalance

    market_rows: List[Dict] = []
    errors: List[Dict] = []
    for ticker in selected:
        symbol = ticker + "USDT"
        try:
            row = fetch_market(symbol)
            row["ticker"] = ticker
            row["momentum_126d"] = float(momentum.loc[ticker])
            row["target_equal_weight"] = 1.0 / len(selected)
            market_rows.append(row)
        except Exception as exc:
            errors.append({"ticker": ticker, "symbol": symbol, "error": str(exc)})
        time.sleep(0.1)

    combined_funding_drag = (
        float(np.mean([row["recent_simple_annualized_long_funding_drag"] for row in market_rows]))
        if market_rows else float("nan")
    )
    max_spread = (
        float(np.nanmax([row["spread_pct"] for row in market_rows])) if market_rows else float("nan")
    )

    # Execution gate: market regime, all symbols resolvable, reasonable spread,
    # and recent simple annualized funding drag below 20%.
    execution_ok = bool(
        regime_on
        and len(selected) == 3
        and len(market_rows) == 3
        and np.isfinite(combined_funding_drag)
        and combined_funding_drag < 0.20
        and np.isfinite(max_spread)
        and max_spread <= 0.005
    )

    result = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "underlying_signal_date": latest.isoformat(),
        "strategy": "126-session momentum, top 3, 5-session rebalance, SPY above 200-day average",
        "regime_on": regime_on,
        "spy_adjusted_close": float(close.loc[latest, "SPY"]),
        "spy_ma200": float(spy_ma200.loc[latest]),
        "sessions_until_next_scheduled_rebalance": int(next_rebalance_sessions),
        "selected_tickers": selected,
        "combined_recent_simple_annualized_funding_drag": combined_funding_drag,
        "maximum_current_spread_pct": max_spread,
        "execution_gate": "PASS" if execution_ok else "NO_TRADE",
        "market_data": market_rows,
        "errors": errors,
        "notes": [
            "Annualized funding is a simple extrapolation of the latest 100 settlements and can change rapidly.",
            "Underlying signal uses adjusted U.S. stock closes; Bitget mark prices can diverge outside U.S. trading hours.",
            "This file does not place orders.",
        ],
    }
    (OUTDIR / "live_signal.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Wave 4C — current stock-perp signal",
        "",
        f"- Generated: {result['generated_at']}",
        f"- Underlying signal date: {result['underlying_signal_date']}",
        f"- Regime: {'ON' if regime_on else 'OFF'}",
        f"- Selected: {', '.join(selected) if selected else 'CASH'}",
        f"- Combined recent annualized funding drag: {combined_funding_drag:.2%}" if np.isfinite(combined_funding_drag) else "- Funding drag unavailable",
        f"- Maximum bid/ask spread: {max_spread:.3%}" if np.isfinite(max_spread) else "- Spread unavailable",
        f"- Execution gate: **{result['execution_gate']}**",
        "",
        "## Current market data",
        "",
        pd.DataFrame(market_rows).to_markdown(index=False, floatfmt=".6f") if market_rows else "No eligible markets resolved.",
        "",
        "## Errors",
        "",
        f"```json\n{json.dumps(errors, indent=2)}\n```",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
