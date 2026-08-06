# Crypto futures strategy scan

- Data: Binance USD-M perpetual 1h klines, 2023-01-01 00:00:00+00:00 to 2026-07-31 23:00:00+00:00
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
- Conservative one-way execution cost: 0.08%
- Training period ends: 2024-12-31 23:00:00+00:00
- Strict out-of-sample test begins: 2025-01-01 00:00:00+00:00

## Selected family models — out of sample

| candidate        | family    | profile    |    cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |
|:-----------------|:----------|:-----------|--------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|
| donchian_72      | donchian  | base       | -0.1114 |  -0.1536 |        -0.3759 |     -0.0071 |       -0.1086 |               0.4211 |            0.1053 |
| ema_48-168       | ema       | base       | -0.0816 |  -0.0610 |        -0.3062 |     -0.0014 |       -0.1988 |               0.4737 |            0.1579 |
| flowtrend_24-168 | flowtrend | base       | -0.5396 |  -2.0236 |        -0.7085 |     -0.0569 |       -0.2369 |               0.3684 |            0.0000 |
| tsmom_720        | tsmom     | base       | -0.1147 |  -0.1350 |        -0.3316 |     -0.0037 |       -0.1581 |               0.4737 |            0.2632 |
| xsmom_336-24     | xsmom     | base       | -0.1269 |  -0.2276 |        -0.4570 |     -0.0081 |       -0.1441 |               0.5263 |            0.1053 |
| donchian_72      | donchian  | aggressive | -0.3041 |  -0.1536 |        -0.6307 |     -0.0196 |       -0.2108 |               0.3684 |            0.2105 |
| ema_48-168       | ema       | aggressive | -0.2568 |  -0.0609 |        -0.5330 |     -0.0025 |       -0.3663 |               0.4737 |            0.2632 |
| flowtrend_24-168 | flowtrend | aggressive | -0.7626 |  -1.9679 |        -0.8984 |     -0.0977 |       -0.4226 |               0.3684 |            0.2105 |
| tsmom_72         | tsmom     | aggressive | -0.7564 |  -1.6188 |        -0.9119 |     -0.0944 |       -0.3741 |               0.3684 |            0.1053 |
| xsmom_72-8       | xsmom     | aggressive | -0.8428 |  -2.5336 |        -0.9735 |     -0.1217 |       -0.4516 |               0.2632 |            0.1579 |

## Selected-family ensemble

| candidate                | profile    |    cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |
|:-------------------------|:-----------|--------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|
| selected_family_ensemble | base       | -0.1957 |  -0.7032 |        -0.3118 |     -0.0160 |       -0.1077 |               0.4737 |            0.0000 |
| selected_family_ensemble | aggressive | -0.6281 |  -1.5244 |        -0.8204 |     -0.0684 |       -0.3081 |               0.3158 |            0.1053 |

## Decision rule

A model is not considered deployable merely because average monthly return exceeds 10%. It must also remain profitable out of sample, have Sharpe above 1.0, maximum drawdown better than -30%, positive months above 55%, and no single month dominating the result.

## Files

- `all_candidates.csv`: every parameter set and split
- `selected_out_of_sample.csv`: one train-selected model per family
- `ensemble.csv`: ensemble results
- `monthly_returns.csv`: monthly return series