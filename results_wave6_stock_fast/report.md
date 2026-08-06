# Fast stock trade-level 2% target audit

- Adjusted OHLC; train 2014–2022; strict OOS 2023–2026-08.
- One-way execution cost 0.072%; stress test adds the same cost again.
- Entry next open; same-day stop assumed before target; one position at a time.
- A pass means historical positive expectancy with net winner target, never every future trade winning.

| name                               | signal         |   trades |   total_return |   avg_trade |   win_rate |   avg_win |   median_win |   avg_loss |   profit_factor |   max_drawdown |   target_hit_rate |   winner_ge_2_pct |   positive_year_pct |   bootstrap_mean_5pct |   stress_profit_factor |   stress_avg_trade | passes_strict   |
|:-----------------------------------|:---------------|---------:|---------------:|------------:|-----------:|----------:|-------------:|-----------:|----------------:|---------------:|------------------:|------------------:|--------------------:|----------------------:|-----------------------:|-------------------:|:----------------|
| pullback_rsi20_tp0.030_sl0.012_h20 | pullback_rsi20 |      203 |         0.2301 |      0.0012 |     0.3498 |    0.0285 |       0.0286 |    -0.0134 |          1.1394 |        -0.1741 |            0.3448 |            1.0000 |              0.7500 |               -0.0011 |                 1.0542 |             0.0005 | False           |
| trendpullback1_tp0.030_sl0.012_h20 | trendpullback1 |      261 |         0.1927 |      0.0009 |     0.3410 |    0.0285 |       0.0286 |    -0.0134 |          1.0984 |        -0.2818 |            0.3372 |            1.0000 |              0.7500 |               -0.0012 |                 1.0162 |             0.0002 | False           |
| trendpullback2_tp0.030_sl0.012_h20 | trendpullback2 |      253 |         0.1210 |      0.0006 |     0.3360 |    0.0285 |       0.0286 |    -0.0134 |          1.0724 |        -0.2476 |            0.3320 |            1.0000 |              0.7500 |               -0.0014 |                 0.9921 |            -0.0001 | False           |
| breakout55_tp0.030_sl0.012_h20     | breakout55     |      201 |         0.0501 |      0.0004 |     0.3333 |    0.0282 |       0.0286 |    -0.0134 |          1.0486 |        -0.2812 |            0.3284 |            0.9851 |              0.5000 |               -0.0019 |                 0.9698 |            -0.0003 | False           |
| pullback_rsi10_tp0.030_sl0.018_h20 | pullback_rsi10 |       93 |        -0.0481 |     -0.0003 |     0.3978 |    0.0285 |       0.0286 |    -0.0193 |          0.9778 |        -0.2627 |            0.3871 |            1.0000 |              0.7500 |               -0.0043 |                 0.9187 |            -0.0010 | False           |
| breakout20_tp0.030_sl0.012_h20     | breakout20     |      209 |        -0.1684 |     -0.0007 |     0.3062 |    0.0282 |       0.0286 |    -0.0134 |          0.9251 |        -0.3998 |            0.3014 |            0.9844 |              0.2500 |               -0.0029 |                 0.8556 |            -0.0014 | False           |

Strict pass count: **0**.