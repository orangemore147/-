# Wave 12 — frozen squeeze-short external-universe confirmation

- Rule was frozen before this test: 4h downside break of prior 18 bars after ATR-rank compression <=25%, BTC below EMA200.
- Short only; TP 3.0%, SL 1.8%, maximum hold 24h; next-bar-open entry; stop-first on ambiguous bars.
- New universe was not used in Wave 6 parameter selection: DOTUSDT, AAVEUSDT, BCHUSDT, ETCUSDT, NEARUSDT, FILUSDT, XLMUSDT, UNIUSDT, ATOMUSDT, TRXUSDT
- Strict evaluation window: 2025–2026-07; Binance USD-M funding included.
- Base one-way execution cost 0.08%; stress uses 1.5x total execution cost.
- Confirmation requires CAGR>=8%, PF>=1.25, DD<=25%, bootstrap 5% mean>0, both calendar years positive, and stress CAGR>=8%.

|   trades |   cagr |   total_return |   avg_trade |   win_rate |   profit_factor |   max_drawdown |   median_win |   target_hit_rate |   bootstrap_mean_5pct |   positive_year_pct |   stress_cagr |   stress_profit_factor |   stress_max_drawdown | passes_external_confirmation   |
|---------:|-------:|---------------:|------------:|-----------:|----------------:|---------------:|-------------:|------------------:|----------------------:|--------------------:|--------------:|-----------------------:|----------------------:|:-------------------------------|
|       89 | 0.0319 |         0.0508 |      0.0008 |     0.4382 |          1.0811 |        -0.1761 |       0.0282 |            0.3371 |               -0.0029 |              0.5000 |       -0.0136 |                 0.9989 |               -0.2129 | False                          |

EXTERNAL CONFIRMATION PASS: **False**.