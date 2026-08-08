# Robustness audit — frozen development candidates

- Completed: 2026-08-08T09:35:08.036929+00:00
- New development candidates audited: 20
- Robust-development survivors: 20
- One-shot holdout evaluations: 20
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family            | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| 0eddfaf210d5 | climax_reversal   | SOLUSDT  | long   | True         |                0.9167 |                   3.2372 |              5 |                 1.0000 |             57 |     2.1645 |           45 |   4.0668 | False          |               59 |       0.8825 |        -0.0921 |      0.7758 |       -0.1544 |
| 5ae860a2a138 | climax_reversal   | SOLUSDT  | long   | True         |                0.9167 |                   2.5134 |              5 |                 0.8000 |             38 |     2.5998 |           45 |   3.6425 | False          |               46 |       0.6609 |        -0.2097 |      0.5783 |       -0.2525 |
| df6e7d1b133d | climax_reversal   | SOLUSDT  | long   | True         |                0.9167 |                   3.0810 |              5 |                 0.8000 |             57 |     2.2091 |           49 |   3.5500 | False          |               61 |       0.5645 |        -0.3361 |      0.4925 |       -0.3834 |
| 4c1e8ab25687 | climax_reversal   | SOLUSDT  | long   | True         |                0.9167 |                   2.2917 |              5 |                 1.0000 |             64 |     2.1215 |           50 |   2.5731 | False          |               65 |       0.9080 |        -0.0828 |      0.7969 |       -0.1519 |
| 01b0c7127018 | compression_break | BTCUSDT  | long   | True         |                0.9167 |                   2.3198 |              5 |                 1.0000 |             34 |     2.9324 |           24 |   2.5488 | False          |               30 |       1.2981 |         0.0447 |      1.0698 |        0.0078 |
| 7d29c8de0b9c | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.7199 |              5 |                 1.0000 |             75 |     2.4379 |           46 |   3.2777 | False          |               59 |       0.5240 |        -0.3478 |      0.4544 |       -0.3929 |
| bb7b8a1e57ac | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.3976 |              5 |                 1.0000 |             75 |     2.2801 |           47 |   3.2204 | False          |               64 |       0.5352 |        -0.3671 |      0.4657 |       -0.4144 |
| 38f1611b5b57 | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.3998 |              5 |                 1.0000 |             48 |     2.3683 |           37 |   2.5602 | False          |               45 |       0.5397 |        -0.2863 |      0.4715 |       -0.3242 |
| da7c1d8632a7 | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.0978 |              5 |                 1.0000 |            100 |     2.3979 |           68 |   2.5007 | False          |              127 |       0.7933 |        -0.1890 |      0.6577 |       -0.3039 |
| 7ccf6e37442c | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.0978 |              5 |                 1.0000 |             96 |     2.2200 |           68 |   2.5007 | False          |              122 |       0.7507 |        -0.2186 |      0.6234 |       -0.3254 |