# Wave 4 — US mega-cap stock-contract strategies

- Data: Yahoo adjusted daily prices, 2015-01-02 00:00:00+00:00 to 2026-07-31 00:00:00+00:00
- Trade universe: AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AMD
- One-way assumed execution cost: 0.08%
- Parameter selection: 2015–2022; strict OOS: 2023–2026-07
- Signals use completed close and are applied to the next close-to-close return

## Train-selected family models — OOS

| candidate                  | family          | profile    | passes_oos   |   cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   month_le_minus10_pct |   annual_turnover |
|:---------------------------|:----------------|:-----------|:-------------|-------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|-----------------------:|------------------:|
| top_momentum_63-5-21-200   | top_momentum    | balanced   | True         | 0.7543 |   1.6230 |        -0.3293 |      0.0562 |       -0.1804 |               0.5581 |            0.3256 |                 0.0698 |           13.8493 |
| top_momentum_63-5-21-200   | top_momentum    | base       | True         | 0.4037 |   1.5830 |        -0.2003 |      0.0314 |       -0.1065 |               0.5581 |            0.1860 |                 0.0233 |            8.5534 |
| dual_ema_20-150            | dual_ema        | base       | True         | 0.3367 |   1.4952 |        -0.1835 |      0.0260 |       -0.0725 |               0.5581 |            0.1395 |                 0.0000 |            7.7451 |
| inverse_vol_200-21-21      | inverse_vol     | balanced   | True         | 0.5714 |   1.4372 |        -0.3330 |      0.0438 |       -0.1318 |               0.6047 |            0.3023 |                 0.1395 |            8.7492 |
| dual_ema_20-150            | dual_ema        | balanced   | True         | 0.5436 |   1.4326 |        -0.3049 |      0.0412 |       -0.1220 |               0.5581 |            0.3488 |                 0.0465 |           11.5816 |
| breakout_100-50            | breakout        | base       | True         | 0.3053 |   1.4322 |        -0.1854 |      0.0239 |       -0.0588 |               0.5581 |            0.1163 |                 0.0000 |            5.5320 |
| sma_long_200               | sma_long        | base       | True         | 0.3196 |   1.4288 |        -0.1801 |      0.0249 |       -0.0730 |               0.6047 |            0.0930 |                 0.0000 |           11.2324 |
| inverse_vol_200-21-21      | inverse_vol     | base       | True         | 0.3209 |   1.4180 |        -0.2054 |      0.0253 |       -0.0879 |               0.6047 |            0.1395 |                 0.0000 |            5.8069 |
| sma_long_200               | sma_long        | balanced   | True         | 0.5226 |   1.3873 |        -0.2996 |      0.0400 |       -0.1257 |               0.5814 |            0.3023 |                 0.0465 |           17.0355 |
| breakout_100-50            | breakout        | balanced   | True         | 0.4246 |   1.3168 |        -0.2733 |      0.0334 |       -0.1014 |               0.5581 |            0.2791 |                 0.0233 |            6.9922 |
| pullback_200-40-60         | pullback        | base       | True         | 0.1161 |   1.0535 |        -0.0906 |      0.0094 |       -0.0426 |               0.5581 |            0.0000 |                 0.0000 |            7.1562 |
| single_rotation_126-21-200 | single_rotation | balanced   | False        | 0.7164 |   1.5213 |        -0.3612 |      0.0546 |       -0.1733 |               0.5116 |            0.3488 |                 0.0930 |           10.8586 |
| inverse_vol_100-21-5       | inverse_vol     | aggressive | False        | 0.9368 |   1.5032 |        -0.4168 |      0.0671 |       -0.2355 |               0.6279 |            0.3721 |                 0.0930 |           21.5320 |
| single_rotation_126-21-200 | single_rotation | base       | False        | 0.3770 |   1.4644 |        -0.2224 |      0.0298 |       -0.1298 |               0.5116 |            0.2093 |                 0.0233 |            6.5450 |
| sma_long_200               | sma_long        | aggressive | False        | 0.8034 |   1.3700 |        -0.4382 |      0.0610 |       -0.1933 |               0.5814 |            0.3953 |                 0.1395 |           25.7568 |
| single_rotation_126-5-200  | single_rotation | aggressive | False        | 0.9515 |   1.3346 |        -0.5082 |      0.0737 |       -0.2594 |               0.6047 |            0.3721 |                 0.1860 |           27.3337 |
| breakout_100-50            | breakout        | aggressive | False        | 0.6427 |   1.3037 |        -0.3925 |      0.0505 |       -0.1560 |               0.5581 |            0.3721 |                 0.0698 |           10.4157 |
| dual_ema_50-100            | dual_ema        | aggressive | False        | 0.7234 |   1.3030 |        -0.4367 |      0.0589 |       -0.2323 |               0.5814 |            0.4186 |                 0.1860 |           25.2952 |
| top_momentum_126-1-5-200   | top_momentum    | aggressive | False        | 0.8683 |   1.2679 |        -0.5396 |      0.0695 |       -0.2683 |               0.6279 |            0.3488 |                 0.1860 |           30.5002 |
| pullback_100-40-60         | pullback        | balanced   | False        | 0.0849 |   1.1127 |        -0.0709 |      0.0070 |       -0.0397 |               0.4884 |            0.0000 |                 0.0000 |            7.7960 |
| pullback_100-40-60         | pullback        | aggressive | False        | 0.1277 |   1.1127 |        -0.1053 |      0.0104 |       -0.0594 |               0.4651 |            0.0000 |                 0.0000 |           11.6940 |

## Family ensemble — OOS

| profile    | split   |   total_return |   cagr |   ann_vol |   sharpe |   max_drawdown |   avg_month |   median_month |   best_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   month_le_minus10_pct |   months |   avg_gross_exposure |   annual_turnover |
|:-----------|:--------|---------------:|-------:|----------:|---------:|---------------:|------------:|---------------:|-------------:|--------------:|---------------------:|------------------:|-----------------------:|---------:|---------------------:|------------------:|
| base       | test    |         1.6525 | 0.3153 |    0.1770 |   1.6377 |        -0.1514 |      0.0243 |         0.0103 |       0.1455 |       -0.0746 |               0.6047 |            0.0930 |                 0.0000 |       43 |               0.9487 |            0.0000 |
| balanced   | test    |         3.4878 | 0.5247 |    0.2848 |   1.6238 |        -0.2430 |      0.0394 |         0.0172 |       0.2472 |       -0.1233 |               0.5814 |            0.2791 |                 0.0233 |       43 |               0.9487 |            0.0000 |
| aggressive | test    |         6.6448 | 0.7708 |    0.4361 |   1.5282 |        -0.3569 |      0.0561 |         0.0324 |       0.3409 |       -0.1974 |               0.6047 |            0.4186 |                 0.0930 |       43 |               0.9398 |            0.0000 |

## OOS pass rule

CAGR > 0; Sharpe >= 0.75; max drawdown >= -35%; positive months >= 55%; average month > 0.
A 10% month is measured as a historical frequency, never treated as a guaranteed target.