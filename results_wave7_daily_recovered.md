# Wave 7 — derivatives crowding/OI edge

- Data: 2023-01-01 00:00:00+00:00 to 2026-07-31 20:00:00+00:00
- Train-only selection: 2023–2024; strict OOS: 2025–2026-07.
- Binance USD-M 4h; official kline/funding plus 5m futures metrics aggregated to 4h.
- One position at a time; next-bar-open entry; stop-first on ambiguous bars.
- Base one-way cost 0.08%; stress adds another 0.08% round trip-equivalent per trade.
- Reporting floor: CAGR >= 8%.

## Train-selected OOS models with CAGR >= 8%

None.

PASS count: **0**.

Recovered verbatim from successful strategy-computation output of GitHub Actions job 92753969897; that workflow subsequently failed only because a concurrent commit caused a non-fast-forward git push rejection.