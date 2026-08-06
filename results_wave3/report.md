# Wave 3 — delta-neutral funding carry

- Data: 2023-01-01 00:00:00+00:00 to 2026-07-31 23:00:00+00:00
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, AVAXUSDT
- Structure: long spot + short equal-notional perpetual
- Signal uses only funding events already settled; position starts next hour
- Parameter selection: 2023–2024; strict OOS: 2025–2026-07
- Taker pair cost: 0.20% per entry or exit; 0.40% complete round trip

## Train-selected model tested out of sample

| candidate                           | cost_profile     | passes_oos   |   cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   funding_return_sum |   basis_return_sum |   annual_turnover |   active_time_pct |
|:------------------------------------|:-----------------|:-------------|-------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|---------------------:|-------------------:|------------------:|------------------:|
| ranked_carry_lb12_k5_th2_reb6_keep1 | taker            | False        | 0.0000 |      nan |         0.0000 |      0.0000 |        0.0000 |               0.0000 |            0.0000 |               0.0000 |             0.0000 |            0.0000 |            0.0000 |
| ranked_carry_lb12_k5_th2_reb6_keep1 | optimistic_limit | False        | 0.0000 |      nan |         0.0000 |      0.0000 |        0.0000 |               0.0000 |            0.0000 |               0.0000 |             0.0000 |            0.0000 |            0.0000 |

## OOS parameter robustness

| cost_profile     |   candidates |   positive_cagr |   sharpe_ge_1 |   drawdown_better_20 |   median_cagr |   median_sharpe |
|:-----------------|-------------:|----------------:|--------------:|---------------------:|--------------:|----------------:|
| optimistic_limit |          240 |               0 |             0 |                  131 |       -0.0787 |         -6.3107 |
| taker            |          240 |               0 |             0 |                  122 |       -0.1234 |         -8.1380 |

## Pass rule

OOS CAGR > 0, Sharpe >= 1, max drawdown >= -20%, positive months >= 55%, average month > 0.
This backtest excludes exchange default, withdrawal freezes, spot custody risk, margin-liquidation mechanics, taxes and cross-exchange transfer delays.