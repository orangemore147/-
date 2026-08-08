# Robustness audit — frozen development candidates

- Completed: 2026-08-08T14:56:38.190662+00:00
- New development candidates audited: 8
- Robust-development survivors: 8
- One-shot holdout evaluations: 8
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family              | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:--------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| bb0ee9365452 | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   2.3161 |              5 |                 1.0000 |             44 |     1.9034 |           25 |   2.6715 | False          |               40 |       1.0622 |         0.0083 |      0.8814 |       -0.0390 |
| a40b4dfa21c5 | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   1.7870 |              5 |                 0.6000 |             42 |     1.7225 |           23 |   2.0582 | False          |               46 |       0.8039 |        -0.0717 |      0.6739 |       -0.1218 |
| 7f08c6e31131 | settlement_reversal | BTCUSDT  | long   | True         |                0.9231 |                   1.8037 |              4 |                 1.0000 |             35 |     2.2643 |           30 |   1.8739 | False          |               79 |       1.1971 |         0.0874 |      1.0062 |       -0.0109 |
| 9b77f4d20db4 | settlement_reversal | BTCUSDT  | long   | True         |                0.9231 |                   1.5896 |              4 |                 1.0000 |             38 |     2.2285 |           31 |   1.7320 | False          |               88 |       1.1857 |         0.0916 |      0.9970 |       -0.0177 |
| df53ef8ccb9e | compression_break   | BTCUSDT  | long   | True         |                0.9167 |                   1.9429 |              5 |                 1.0000 |             42 |     1.7589 |           28 |   2.1255 | False          |               33 |       1.1174 |         0.0186 |      0.9369 |       -0.0210 |
| 7e96ac16ff6d | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   2.5541 |              5 |                 1.0000 |             86 |     1.8197 |           63 |   2.6268 | False          |               75 |       0.6090 |        -0.3511 |      0.5315 |       -0.4074 |
| 88a4a89a3bac | compression_break   | BTCUSDT  | long   | True         |                0.9091 |                   2.0613 |              5 |                 0.8000 |             31 |     2.5639 |           24 |   2.0613 | False          |               29 |       1.2890 |         0.0442 |      1.0747 |        0.0085 |
| 0c429aa4f728 | climax_reversal     | SOLUSDT  | long   | True         |                0.9000 |                   1.9575 |              5 |                 1.0000 |             95 |     1.7410 |           55 |   2.1639 | False          |               94 |       0.6017 |        -0.3961 |      0.5286 |       -0.4610 |