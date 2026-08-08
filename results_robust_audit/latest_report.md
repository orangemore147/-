# Robustness audit — frozen development candidates

- Completed: 2026-08-08T14:06:37.602882+00:00
- New development candidates audited: 20
- Robust-development survivors: 19
- One-shot holdout evaluations: 19
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family              | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:--------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| a2b5c2a25240 | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   2.4324 |              5 |                 1.0000 |             44 |     1.9034 |           25 |   2.6715 | False          |          40.0000 |       1.0622 |         0.0083 |      0.8814 |       -0.0390 |
| aaa350454fd5 | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   1.7870 |              5 |                 0.6000 |             42 |     1.7225 |           23 |   2.0582 | False          |          46.0000 |       0.8039 |        -0.0717 |      0.6739 |       -0.1218 |
| 283309775531 | settlement_reversal | BTCUSDT  | long   | True         |                0.9231 |                   1.5896 |              4 |                 1.0000 |             38 |     2.2285 |           31 |   1.7320 | False          |          88.0000 |       1.1857 |         0.0916 |      0.9970 |       -0.0177 |
| f02f243855f3 | compression_break   | BTCUSDT  | long   | True         |                0.9167 |                   2.3198 |              5 |                 1.0000 |             34 |     2.9324 |           24 |   2.5488 | False          |          30.0000 |       1.2981 |         0.0447 |      1.0698 |        0.0078 |
| 00053aed289c | compression_break   | BTCUSDT  | long   | True         |                0.9167 |                   1.9429 |              5 |                 1.0000 |             42 |     1.7589 |           28 |   2.1255 | False          |          33.0000 |       1.1174 |         0.0186 |      0.9369 |       -0.0210 |
| b3587778c152 | settlement_reversal | BTCUSDT  | long   | True         |                0.9167 |                   1.7692 |              4 |                 1.0000 |             35 |     2.2643 |           30 |   1.8739 | False          |          79.0000 |       1.1971 |         0.0874 |      1.0062 |       -0.0109 |
| 5913cb5fce5e | compression_break   | BTCUSDT  | both   | True         |                0.9167 |                   1.5591 |              5 |                 1.0000 |             53 |     1.7975 |           29 |   1.7373 | False          |          55.0000 |       0.6709 |        -0.1179 |      0.5480 |       -0.1744 |
| e4b935959be1 | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   1.9798 |              5 |                 1.0000 |            124 |     1.6762 |           89 |   2.0985 | False          |         140.0000 |       0.8734 |        -0.1385 |      0.7276 |       -0.2720 |
| f962707b6bb4 | compression_break   | SOLUSDT  | both   | True         |                0.9091 |                   1.7468 |              5 |                 0.8000 |            326 |     1.5091 |          149 |   1.7468 | False          |         293.0000 |       0.8955 |        -0.2239 |      0.7420 |       -0.4544 |
| 3d5f59c4f9a1 | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   1.6350 |              5 |                 1.0000 |             98 |     1.8093 |           55 |   1.6768 | False          |         101.0000 |       0.7426 |        -0.2380 |      0.6375 |       -0.3254 |