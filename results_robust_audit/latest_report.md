# Robustness audit — frozen development candidates

- Completed: 2026-08-08T17:00:45.511625+00:00
- New development candidates audited: 5
- Robust-development survivors: 5
- One-shot holdout evaluations: 5
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family              | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:--------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| 4688d768db18 | compression_break   | BTCUSDT  | long   | True         |                1.0000 |                   2.4324 |              5 |                 1.0000 |             44 |     1.9034 |           25 |   2.6715 | False          |               40 |       1.0622 |         0.0083 |      0.8814 |       -0.0390 |
| 54d97e2e2192 | settlement_reversal | BTCUSDT  | long   | True         |                0.9231 |                   1.5896 |              4 |                 1.0000 |             38 |     2.2285 |           31 |   1.7320 | False          |               88 |       1.1857 |         0.0916 |      0.9970 |       -0.0177 |
| 279bbfc38131 | compression_break   | BTCUSDT  | long   | True         |                0.9167 |                   1.9429 |              5 |                 1.0000 |             42 |     1.7589 |           28 |   2.1255 | False          |               33 |       1.1174 |         0.0186 |      0.9369 |       -0.0210 |
| 7bb8a914d6e1 | compression_break   | BTCUSDT  | long   | True         |                0.9091 |                   2.0613 |              5 |                 0.8000 |             31 |     2.5639 |           24 |   2.0613 | False          |               29 |       1.2890 |         0.0442 |      1.0747 |        0.0085 |
| a18168957b15 | climax_reversal     | SOLUSDT  | long   | True         |                0.8182 |                   1.9284 |              5 |                 1.0000 |             75 |     2.9060 |           58 |   1.9338 | False          |               57 |       0.4350 |        -0.4487 |      0.3814 |       -0.4857 |