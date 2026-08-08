# Robustness audit — frozen development candidates

- Completed: 2026-08-08T10:58:38.472072+00:00
- New development candidates audited: 20
- Robust-development survivors: 20
- One-shot holdout evaluations: 20
- External confirmations: 0
- Strict passes: 0

Holdout results are never used to mutate or rank future candidates. Signal-equivalent development variants are collapsed before holdout.

| key          | family            | symbol   | side   | robust_dev   |   neighbor_pass_ratio |   neighbor_median_val_pf |   valid_blocks |   positive_block_ratio |   train_trades |   train_pf |   val_trades |   val_pf | holdout_gate   |   holdout_trades |   holdout_pf |   holdout_cagr |   stress_pf |   stress_cagr |
|:-------------|:------------------|:---------|:-------|:-------------|----------------------:|-------------------------:|---------------:|-----------------------:|---------------:|-----------:|-------------:|---------:|:---------------|-----------------:|-------------:|---------------:|------------:|--------------:|
| c4239dbafd66 | climax_reversal   | SOLUSDT  | long   | True         |                0.9167 |                   2.2917 |              5 |                 1.0000 |             66 |     2.1496 |           50 |   2.5731 | False          |               68 |       0.9308 |        -0.0697 |      0.8138 |       -0.1428 |
| 1a2bf855a42c | compression_break | BTCUSDT  | long   | True         |                0.9167 |                   2.3198 |              5 |                 1.0000 |             34 |     2.9324 |           24 |   2.5488 | False          |               30 |       1.2981 |         0.0447 |      1.0698 |        0.0078 |
| a8ae517919d3 | climax_reversal   | SOLUSDT  | long   | True         |                0.9167 |                   2.1967 |              5 |                 1.0000 |             76 |     1.9247 |           54 |   2.4120 | False          |               90 |       0.7110 |        -0.2895 |      0.6256 |       -0.3627 |
| 9897b84a2d8b | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.3248 |              5 |                 1.0000 |             67 |     2.1380 |           43 |   3.0408 | False          |               55 |       0.4542 |        -0.3927 |      0.3936 |       -0.4320 |
| 360dd0bcc0e7 | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.4483 |              5 |                 1.0000 |             64 |     1.9558 |           56 |   2.7660 | False          |               78 |       1.0405 |         0.0064 |      0.8654 |       -0.0836 |
| 2ff779a5e67a | climax_reversal   | SOLUSDT  | long   | True         |                0.9091 |                   2.3301 |              5 |                 1.0000 |             71 |     2.1314 |           51 |   2.5388 | False          |               70 |       0.9521 |        -0.0576 |      0.8321 |       -0.1337 |
| aecea451b906 | climax_reversal   | SOLUSDT  | long   | True         |                0.9000 |                   2.1126 |              5 |                 1.0000 |             66 |     2.0354 |           56 |   2.6762 | False          |               79 |       1.0237 |        -0.0026 |      0.8506 |       -0.0929 |
| e52325e6de4f | climax_reversal   | SOLUSDT  | long   | True         |                0.9000 |                   2.0525 |              5 |                 1.0000 |             72 |     2.4683 |           45 |   2.2672 | False          |               56 |       0.5246 |        -0.3485 |      0.4593 |       -0.3914 |
| 5a17e351507a | climax_reversal   | SOLUSDT  | long   | True         |                0.9000 |                   1.9955 |              5 |                 1.0000 |             93 |     1.9354 |           56 |   1.9561 | False          |               72 |       0.5062 |        -0.4496 |      0.4437 |       -0.4957 |
| 091e9d256624 | climax_reversal   | SOLUSDT  | long   | True         |                0.8333 |                   2.2359 |              5 |                 0.8000 |             36 |     2.1385 |           43 |   2.2359 | False          |               43 |       0.7541 |        -0.1118 |      0.6531 |       -0.1566 |