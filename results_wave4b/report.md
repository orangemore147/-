# Wave 4B — stock strategy robustness

- Data: 2015-01-02 00:00:00+00:00 to 2026-07-31 00:00:00+00:00
- Broad Bitget-supported long-history universe: AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, INTC, ORCL, IBM, CSCO, PEP, MCD, GE, MA, BABA, LLY, UNH, ASML
- Train: 2015–2022; strict OOS: 2023–2026-07
- Trading cost: 0.08% per one-way notional change

## Train-selected models — OOS

| candidate          | family       | profile   | passes_oos   |   cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |
|:-------------------|:-------------|:----------|:-------------|-------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|
| invvol_150_21_21   | inverse_vol  | unlevered | True         | 0.2372 |   1.7681 |        -0.0894 |      0.0184 |       -0.0546 |               0.6977 |            0.0000 |
| invvol_150_21_21   | inverse_vol  | base      | True         | 0.3542 |   1.7400 |        -0.1294 |      0.0268 |       -0.0785 |               0.6977 |            0.0930 |
| invvol_100_21_21   | inverse_vol  | balanced  | True         | 0.4722 |   1.7125 |        -0.1736 |      0.0349 |       -0.1081 |               0.6977 |            0.1628 |
| dualema_50_150     | dual_ema     | unlevered | True         | 0.1959 |   1.5530 |        -0.0963 |      0.0155 |       -0.0525 |               0.6047 |            0.0233 |
| dualema_50_150     | dual_ema     | balanced  | True         | 0.4096 |   1.5530 |        -0.1868 |      0.0312 |       -0.1030 |               0.6047 |            0.1395 |
| dualema_50_150     | dual_ema     | base      | True         | 0.2872 |   1.5183 |        -0.1410 |      0.0224 |       -0.0775 |               0.6047 |            0.0465 |
| topmom_126_3_5_200 | top_momentum | balanced  | True         | 0.4956 |   1.2204 |        -0.2519 |      0.0395 |       -0.1324 |               0.5581 |            0.2791 |
| topmom_126_3_5_200 | top_momentum | base      | True         | 0.2775 |   1.1773 |        -0.1635 |      0.0224 |       -0.0770 |               0.5581 |            0.1628 |
| topmom_126_3_5_200 | top_momentum | unlevered | True         | 0.3534 |   1.1069 |        -0.2020 |      0.0292 |       -0.1879 |               0.5814 |            0.2093 |

## Top-momentum funding-drag sensitivity

| family       | profile   |   annual_funding_drag |    cagr |   sharpe |   max_drawdown |   avg_month |   positive_month_pct |   month_ge_10_pct |
|:-------------|:----------|----------------------:|--------:|---------:|---------------:|------------:|---------------------:|------------------:|
| top_momentum | unlevered |                0.0000 |  0.3534 |   1.1069 |        -0.2020 |      0.0292 |               0.5814 |            0.2093 |
| top_momentum | unlevered |                0.1000 |  0.2353 |   0.8206 |        -0.2191 |      0.0215 |               0.5349 |            0.1860 |
| top_momentum | unlevered |                0.2000 |  0.1274 |   0.5341 |        -0.2528 |      0.0138 |               0.5349 |            0.1860 |
| top_momentum | unlevered |                0.4000 | -0.0610 |  -0.0388 |        -0.4691 |     -0.0015 |               0.4651 |            0.1860 |
| top_momentum | base      |                0.0000 |  0.2775 |   1.1773 |        -0.1635 |      0.0224 |               0.5581 |            0.1628 |
| top_momentum | base      |                0.1000 |  0.1948 |   0.8866 |        -0.1828 |      0.0167 |               0.5349 |            0.1395 |
| top_momentum | base      |                0.2000 |  0.1174 |   0.5957 |        -0.2055 |      0.0111 |               0.5349 |            0.0930 |
| top_momentum | base      |                0.4000 | -0.0228 |   0.0136 |        -0.3089 |     -0.0001 |               0.4651 |            0.0930 |
| top_momentum | balanced  |                0.0000 |  0.4956 |   1.2204 |        -0.2519 |      0.0395 |               0.5581 |            0.2791 |
| top_momentum | balanced  |                0.1000 |  0.3351 |   0.9302 |        -0.2831 |      0.0297 |               0.5349 |            0.2326 |
| top_momentum | balanced  |                0.2000 |  0.1918 |   0.6398 |        -0.3166 |      0.0200 |               0.5349 |            0.1860 |
| top_momentum | balanced  |                0.4000 | -0.0506 |   0.0586 |        -0.4951 |      0.0009 |               0.4651 |            0.1628 |

## Top-momentum leave-one-stock-out summary

| profile   |   tests |   min_cagr |   median_cagr |   min_sharpe |   worst_drawdown |   positive_cases |
|:----------|--------:|-----------:|--------------:|-------------:|-----------------:|-----------------:|
| balanced  |      19 |     0.3183 |        0.4956 |       0.8996 |          -0.3130 |               19 |
| base      |      19 |     0.1754 |        0.2775 |       0.8188 |          -0.2253 |               19 |
| unlevered |      19 |     0.2724 |        0.3534 |       0.9219 |          -0.2585 |               19 |

## Caveats

Funding drag scenarios are sensitivities, not observed Bitget funding history. The universe is broader but still uses securities that survived and remained listed through 2026. Corporate-action adjusted underlying prices may differ from stock-perp mark-price behavior outside U.S. market hours.