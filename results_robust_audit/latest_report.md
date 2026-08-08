# Robustness audit — frozen development candidates

- Completed: 2026-08-08T11:52:20.103275+00:00
- New development candidates audited: 20
- Robust-development survivors: 20
- One-shot holdout evaluations: 20
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family              | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:--------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| 9b46a232619e | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   2.4324 |              5 |                 1.0000 |             44 |     1.9034 |           25 |   2.6715 | False          |               40 |       1.0622 |         0.0083 |      0.8814 |       -0.0390 |
| 8733957bd798 | settlement_reversal | BTCUSDT  | long   | True         |                0.9231 |                   1.8037 |              4 |                 1.0000 |             35 |     2.2643 |           30 |   1.8739 | False          |               79 |       1.1971 |         0.0874 |      1.0062 |       -0.0109 |
| 8b31fa5db43a | climax_reversal     | SOLUSDT  | long   | True         |                0.9167 |                   2.7860 |              5 |                 0.8000 |             60 |     1.9079 |           52 |   3.4292 | False          |               66 |       0.6371 |        -0.2942 |      0.5563 |       -0.3484 |
| fa64e5037861 | climax_reversal     | SOLUSDT  | long   | True         |                0.9167 |                   2.3286 |              5 |                 1.0000 |             55 |     1.8518 |           50 |   2.5734 | False          |               62 |       0.8692 |        -0.0996 |      0.7580 |       -0.1644 |
| e8f8e4ed2a3f | compression_break   | BTCUSDT  | long   | True         |                0.9167 |                   2.3198 |              5 |                 1.0000 |             34 |     2.9324 |           24 |   2.5488 | False          |               30 |       1.2981 |         0.0447 |      1.0698 |        0.0078 |
| 5932df144b18 | climax_reversal     | SOLUSDT  | long   | True         |                0.9167 |                   2.1231 |              5 |                 1.0000 |             82 |     1.8894 |           55 |   2.3030 | False          |               97 |       0.7196 |        -0.2952 |      0.6324 |       -0.3731 |
| 131c9a08983d | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   2.4666 |              5 |                 1.0000 |             49 |     2.1865 |           38 |   2.6404 | False          |               46 |       0.5276 |        -0.2970 |      0.4604 |       -0.3352 |
| 184730d2ff4f | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   2.4797 |              5 |                 1.0000 |             52 |     1.8919 |           38 |   2.6401 | False          |               47 |       0.5404 |        -0.2937 |      0.4718 |       -0.3328 |
| 2d4777c48479 | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   2.2067 |              5 |                 1.0000 |             69 |     1.9061 |           57 |   2.4483 | False          |               81 |       0.9897 |        -0.0213 |      0.8303 |       -0.1121 |
| 0c88d8e32ff3 | climax_reversal     | SOLUSDT  | long   | True         |                0.8889 |                   1.7496 |              5 |                 1.0000 |            109 |     2.2721 |           68 |   1.7777 | False          |              118 |       0.7723 |        -0.1985 |      0.6435 |       -0.3046 |