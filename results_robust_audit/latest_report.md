# Robustness audit — frozen development candidates

- Completed: 2026-08-08T13:25:37.439188+00:00
- New development candidates audited: 20
- Robust-development survivors: 20
- One-shot holdout evaluations: 20
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family              | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:--------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| b27f93d3d39c | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   2.4324 |              5 |                 1.0000 |             44 |     1.9034 |           25 |   2.6715 | False          |               40 |       1.0622 |         0.0083 |      0.8814 |       -0.0390 |
| 928c3d12f864 | settlement_reversal | BTCUSDT  | long   | True         |                0.9231 |                   1.8037 |              4 |                 1.0000 |             35 |     2.2643 |           30 |   1.8739 | False          |               79 |       1.1971 |         0.0874 |      1.0062 |       -0.0109 |
| 5abc1586011c | compression_break   | BTCUSDT  | long   | True         |                0.9167 |                   2.3198 |              5 |                 1.0000 |             34 |     2.9324 |           24 |   2.5488 | False          |               30 |       1.2981 |         0.0447 |      1.0698 |        0.0078 |
| 94e8e622875d | climax_reversal     | SOLUSDT  | long   | True         |                0.9167 |                   2.4652 |              5 |                 0.8000 |             39 |     1.8400 |           44 |   2.4666 | False          |               48 |       0.6472 |        -0.2253 |      0.5661 |       -0.2690 |
| 5c125eb0087e | climax_reversal     | SOLUSDT  | long   | True         |                0.9167 |                   2.2423 |              5 |                 1.0000 |             52 |     1.8508 |           50 |   2.1891 | False          |               62 |       0.9219 |        -0.0516 |      0.7768 |       -0.1198 |
| ad4e1f20273d | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   2.4313 |              5 |                 1.0000 |             53 |     1.7818 |           39 |   2.5948 | False          |               48 |       0.5287 |        -0.3042 |      0.4611 |       -0.3436 |
| 92a92ba37e3d | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   2.0676 |              5 |                 1.0000 |             79 |     1.7867 |           48 |   2.2197 | False          |               63 |       0.4103 |        -0.4653 |      0.3567 |       -0.5048 |
| bb24dd315499 | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   1.8868 |              5 |                 1.0000 |            143 |     1.6879 |           95 |   2.0890 | False          |              158 |       0.8935 |        -0.1359 |      0.7454 |       -0.2854 |
| c3cf19840046 | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   1.6703 |              5 |                 0.6000 |             71 |     2.1496 |           46 |   1.6942 | False          |               72 |       0.8461 |        -0.1101 |      0.7266 |       -0.1840 |
| 0355b5f7fc16 | climax_reversal     | SOLUSDT  | long   | True         |                0.9000 |                   2.0369 |              5 |                 1.0000 |            104 |     2.3077 |           70 |   2.5103 | False          |              132 |       0.8063 |        -0.1843 |      0.6685 |       -0.3041 |