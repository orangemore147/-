# Wave 6 — net 2% winner-target strategy audit

Literal certification that every future trade earns 2% is impossible. This audit tests fixed gross TP >=2.2%, median OOS winner >=2% net, at least 80% of OOS winners >=2% net, and positive expectancy after costs.

- Stock train 2014–2022; OOS 2023–2026-08.
- Crypto train 2022–2024; OOS 2025–2026-07.
- Next-bar-open entries; same-bar stop before target; one position at a time.
- Stock one-way cost 0.072%; crypto one-way cost 0.080%; crypto funding included.
- Strict pass also requires PF>=1.25, max DD>=-25%, bootstrap lower mean>0, >=67% positive years, and 1.5x-cost stress remains positive.

## Train-selected models tested out of sample

| name                                                 | market   | family         |   trades |   total_return |   avg_trade |   win_rate |   avg_win |   median_win |   avg_loss |   profit_factor |   max_drawdown |   target_hit_rate |   winner_ge_2_pct |   positive_year_pct |   bootstrap_mean_5pct |   stress_profit_factor |   stress_avg_trade | passes_strict   |
|:-----------------------------------------------------|:---------|:---------------|---------:|---------------:|------------:|-----------:|----------:|-------------:|-----------:|----------------:|---------------:|------------------:|------------------:|--------------------:|----------------------:|-----------------------:|-------------------:|:----------------|
| stock_pullback_L_lb0_a20_tp0.030_sl0.012_h20         | stock    | pullback       |      203 |         0.2301 |      0.0012 |     0.3498 |    0.0285 |       0.0286 |    -0.0134 |          1.1394 |        -0.1741 |            0.3448 |            1.0000 |              0.7500 |               -0.0011 |                 1.0542 |             0.0005 | False           |
| stock_trend_pullback_L_lb0_a0.02_tp0.030_sl0.012_h20 | stock    | trend_pullback |      253 |         0.1210 |      0.0006 |     0.3360 |    0.0285 |       0.0286 |    -0.0134 |          1.0724 |        -0.2476 |            0.3320 |            1.0000 |              0.7500 |               -0.0014 |                 0.9921 |            -0.0001 | False           |
| stock_breakout_L_lb20_a0_tp0.030_sl0.012_h20         | stock    | breakout       |      209 |        -0.1684 |     -0.0007 |     0.3062 |    0.0282 |       0.0286 |    -0.0134 |          0.9251 |        -0.3998 |            0.3014 |            0.9844 |              0.2500 |               -0.0029 |                 0.8556 |            -0.0014 | False           |

Strict pass count: **0**.

PASS is reproducible historical evidence, not a guarantee or third-party certification.