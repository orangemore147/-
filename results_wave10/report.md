# Wave 10 — cost-aware XGBoost BTC perpetual

- Binance BTCUSDT USD-M perpetual 1h data.
- Features are lag-safe technical/volume/funding variables; target is next 6h return.
- Hyperparameters and execution threshold selected on 2024 validation only after fitting on data through 2023.
- Strict OOS: monthly expanding/rolling retraining during 2025–2026-07; no OOS return used for configuration selection.
- Base execution cost: 0.08% per unit of one-way turnover; historical funding included.
- Stress: 1.5x execution cost.
- Research floor: OOS CAGR >=8%; Sharpe>=1; DD<=30%; positive months>=55%; stress CAGR>=8%.

## Selected model — strict OOS

| mode      |   threshold_mult |    cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   annual_turnover |   active_pct |   stress_cagr |   stress_sharpe |   stress_max_drawdown | passes_8pct   |
|:----------|-----------------:|--------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|------------------:|-------------:|--------------:|----------------:|----------------------:|:--------------|
| long_only |           3.0000 | -0.0762 |  -1.1781 |        -0.1404 |     -0.0064 |       -0.0531 |               0.1579 |            0.0000 |           44.3143 |       0.0077 |       -0.0924 |         -1.4415 |               -0.1582 | False         |

PASS: **False**.

This is an independent perpetual-futures replication inspired by Bysik & Ślepaczuk (2026), not an exact reproduction of their dataset or model pipeline.