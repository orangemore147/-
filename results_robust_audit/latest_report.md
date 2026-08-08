# Robustness audit — frozen development candidates

- Completed: 2026-08-08T15:54:35.895599+00:00
- New development candidates audited: 9
- Robust-development survivors: 9
- One-shot holdout evaluations: 9
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family              | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:--------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| c62359fc5eee | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   2.4324 |              5 |                 1.0000 |             44 |     1.9034 |           25 |   2.6715 | False          |               40 |       1.0622 |         0.0083 |      0.8814 |       -0.0390 |
| 7a18af37c3c1 | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   1.9602 |              5 |                 0.8000 |             46 |     1.7857 |           31 |   1.9602 | False          |               36 |       0.8185 |        -0.0586 |      0.6978 |       -0.0986 |
| 648436b4e0f1 | settlement_reversal | BTCUSDT  | long   | True         |                1.0000 |                   1.6445 |              4 |                 1.0000 |             34 |     2.3956 |           30 |   1.7346 | False          |               76 |       1.2123 |         0.0990 |      1.0338 |        0.0033 |
| 4c8814623dda | settlement_reversal | BTCUSDT  | long   | True         |                0.9231 |                   1.5896 |              4 |                 1.0000 |             38 |     2.2285 |           31 |   1.7320 | False          |               88 |       1.1857 |         0.0916 |      0.9970 |       -0.0177 |
| 6be1d777c971 | compression_break   | BTCUSDT  | long   | True         |                0.9167 |                   1.9429 |              5 |                 1.0000 |             42 |     1.7589 |           28 |   2.1255 | False          |               33 |       1.1174 |         0.0186 |      0.9369 |       -0.0210 |
| 54c30eaa21bd | compression_break   | SOLUSDT  | long   | True         |                0.9091 |                   2.7833 |              4 |                 0.7500 |             60 |     1.8225 |           21 |   2.7833 | False          |               49 |       0.8218 |        -0.1079 |      0.7331 |       -0.1590 |
| af5c61429e47 | compression_break   | BTCUSDT  | long   | True         |                0.9091 |                   2.1255 |              5 |                 1.0000 |             34 |     2.9324 |           24 |   2.5488 | False          |               30 |       1.2981 |         0.0447 |      1.0698 |        0.0078 |
| 97167f9cbe86 | climax_reversal     | SOLUSDT  | long   | True         |                0.9091 |                   2.1639 |              5 |                 1.0000 |             89 |     1.8373 |           57 |   2.1983 | False          |              102 |       0.7301 |        -0.2998 |      0.6419 |       -0.3809 |
| 23f02ce45321 | climax_reversal     | SOLUSDT  | long   | True         |                0.8182 |                   1.9884 |              5 |                 0.8000 |             40 |     2.3051 |           43 |   1.9884 | False          |               44 |       0.8050 |        -0.1023 |      0.7010 |       -0.1487 |