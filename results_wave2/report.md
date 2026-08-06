# Wave 2 — long/cash and low-frequency crypto strategy scan

- Data: 2023-01-01 00:00:00+00:00 to 2026-07-31 23:00:00+00:00
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, AVAXUSDT
- One-way trading cost: 0.08%
- Parameter selection: 2023–2024 only
- Strict out-of-sample: 2025–2026-07

## Train-selected models, tested out of sample

| candidate                   | family             | profile      | passes_oos   |    cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   month_le_minus10_pct |   annual_turnover |
|:----------------------------|:-------------------|:-------------|:-------------|--------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|-----------------------:|------------------:|
| btc_rotation_30-1-200       | btc_rotation       | aggressive   | False        |  0.8269 |   1.3100 |        -0.4499 |      0.0710 |       -0.1493 |               0.2632 |            0.2105 |                 0.0526 |           64.0832 |
| btc_rotation_30-7-200       | btc_rotation       | balanced     | False        |  0.2774 |   0.8178 |        -0.3721 |      0.0306 |       -0.2282 |               0.2632 |            0.1579 |                 0.0526 |           19.4923 |
| btc_rotation_30-7-200       | btc_rotation       | conservative | False        |  0.1501 |   0.7976 |        -0.2053 |      0.0142 |       -0.1197 |               0.2632 |            0.0526 |                 0.0526 |           10.0624 |
| top_momentum_30-1-1-200     | top_momentum       | aggressive   | False        |  0.1602 |   0.5496 |        -0.6794 |      0.0341 |       -0.3538 |               0.2632 |            0.1579 |                 0.1053 |           73.8148 |
| inverse_vol_basket_200-60   | inverse_vol_basket | balanced     | False        |  0.0544 |   0.3330 |        -0.3520 |      0.0081 |       -0.1806 |               0.3158 |            0.1053 |                 0.1053 |            8.3603 |
| inverse_vol_basket_200-60   | inverse_vol_basket | conservative | False        |  0.0426 |   0.3073 |        -0.1936 |      0.0044 |       -0.0922 |               0.3158 |            0.0526 |                 0.0000 |            4.5099 |
| inverse_vol_basket_100-20   | inverse_vol_basket | aggressive   | False        | -0.1307 |   0.0888 |        -0.5823 |      0.0033 |       -0.3324 |               0.3158 |            0.2105 |                 0.1579 |           20.2311 |
| dual_ema_long_20-100        | dual_ema_long      | aggressive   | False        | -0.1744 |   0.0502 |        -0.5910 |     -0.0028 |       -0.2420 |               0.3158 |            0.2105 |                 0.2632 |           35.1430 |
| top_momentum_30-3-1-200     | top_momentum       | balanced     | False        | -0.0897 |  -0.0417 |        -0.5175 |     -0.0006 |       -0.2539 |               0.2632 |            0.0526 |                 0.1579 |           56.3860 |
| top_momentum_30-3-1-200     | top_momentum       | conservative | False        | -0.0317 |  -0.0638 |        -0.3055 |     -0.0008 |       -0.1345 |               0.2632 |            0.0526 |                 0.0526 |           28.5093 |
| dual_ema_long_50-200        | dual_ema_long      | conservative | False        | -0.0586 |  -0.1779 |        -0.2778 |     -0.0036 |       -0.0912 |               0.3684 |            0.0526 |                 0.0000 |            6.3782 |
| dual_ema_long_50-200        | dual_ema_long      | balanced     | False        | -0.1530 |  -0.1779 |        -0.4899 |     -0.0082 |       -0.1788 |               0.3684 |            0.0526 |                 0.2105 |           12.7564 |
| pullback_long_200-40-55     | pullback_long      | conservative | False        | -0.0569 |  -0.2362 |        -0.2565 |     -0.0040 |       -0.0790 |               0.2105 |            0.0526 |                 0.0000 |           16.1819 |
| pullback_long_200-40-55     | pullback_long      | balanced     | False        | -0.1295 |  -0.2402 |        -0.4573 |     -0.0085 |       -0.1536 |               0.2105 |            0.1053 |                 0.1579 |           28.8846 |
| donchian_long_50-10         | donchian_long      | conservative | False        | -0.1506 |  -0.4586 |        -0.4266 |     -0.0105 |       -0.1493 |               0.3684 |            0.0526 |                 0.1053 |          107.1824 |
| ema_long_200                | ema_long           | balanced     | False        | -0.2598 |  -0.5077 |        -0.5424 |     -0.0194 |       -0.1731 |               0.2632 |            0.0526 |                 0.2632 |           52.2113 |
| ema_long_200                | ema_long           | conservative | False        | -0.1242 |  -0.5230 |        -0.3209 |     -0.0096 |       -0.0890 |               0.3158 |            0.0526 |                 0.0000 |           26.4220 |
| bollinger_bull_10-15--5-200 | bollinger_bull     | balanced     | False        | -0.2695 |  -0.5927 |        -0.5113 |     -0.0175 |       -0.3139 |               0.2105 |            0.1053 |                 0.1579 |           74.5778 |
| bollinger_bull_10-15--5-200 | bollinger_bull     | aggressive   | False        | -0.4198 |  -0.5973 |        -0.7000 |     -0.0244 |       -0.4615 |               0.2105 |            0.1579 |                 0.2105 |          111.0547 |
| ema_long_50                 | ema_long           | aggressive   | False        | -0.5581 |  -0.6287 |        -0.8236 |     -0.0458 |       -0.3238 |               0.3684 |            0.2632 |                 0.4737 |          192.5517 |
| bollinger_bull_10-15--5-200 | bollinger_bull     | conservative | False        | -0.1411 |  -0.6423 |        -0.2969 |     -0.0105 |       -0.1683 |               0.2105 |            0.0526 |                 0.1053 |           38.9563 |
| donchian_long_20-10         | donchian_long      | balanced     | False        | -0.5419 |  -1.2109 |        -0.7723 |     -0.0565 |       -0.2378 |               0.2632 |            0.0526 |                 0.3684 |          276.7051 |
| donchian_long_20-10         | donchian_long      | aggressive   | False        | -0.7513 |  -1.2180 |        -0.9225 |     -0.0934 |       -0.3601 |               0.2105 |            0.0526 |                 0.5789 |          442.2972 |
| pullback_long_100-30-55     | pullback_long      | aggressive   | False        |  0.0000 | nan      |         0.0000 |      0.0000 |        0.0000 |               0.0000 |            0.0000 |                 0.0000 |            0.0000 |

## Family ensemble

| profile      | split   |   total_return |    cagr |   ann_vol |   sharpe |   max_drawdown |   avg_month |   median_month |   best_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   month_le_minus10_pct |   months |   avg_gross_exposure |   annual_turnover |
|:-------------|:--------|---------------:|--------:|----------:|---------:|---------------:|------------:|---------------:|-------------:|--------------:|---------------------:|------------------:|-----------------------:|---------:|---------------------:|------------------:|
| conservative | test    |        -0.0644 | -0.0412 |    0.1572 |  -0.1890 |        -0.2021 |     -0.0026 |        -0.0032 |       0.1191 |       -0.0863 |               0.4211 |            0.0526 |                 0.0000 |       19 |               0.8333 |            0.0000 |
| balanced     | test    |        -0.2110 | -0.1392 |    0.3133 |  -0.3214 |        -0.3959 |     -0.0089 |        -0.0137 |       0.2358 |       -0.1644 |               0.2632 |            0.0526 |                 0.1053 |       19 |               0.9007 |            0.0000 |
| aggressive   | test    |        -0.2637 | -0.1761 |    0.4647 |  -0.1833 |        -0.5416 |     -0.0066 |        -0.0279 |       0.4271 |       -0.2276 |               0.3158 |            0.1579 |                 0.2105 |       19 |               0.8843 |            0.0000 |

## OOS pass rule

CAGR > 0; Sharpe >= 0.75; max drawdown >= -35%; positive months >= 55%; average month > 0.
A 10% month is reported as a frequency, not treated as a guarantee.