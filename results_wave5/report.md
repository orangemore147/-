# Wave 5 — academic strategy replication

- Data: 2014-01-02 00:00:00+00:00 to 2026-08-05 00:00:00+00:00
- Universe: AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, INTC, ORCL, IBM, CSCO, PEP, MCD, GE, MA, BABA, LLY, UNH, ASML
- Train selection: 2014–2022; strict OOS: 2023–2026-08
- One-way execution cost: 0.08%
- Signals use completed daily closes and affect the next daily return.

## Train-selected family models — OOS

| candidate                             | family                |   cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct | passes_sharpe_1_5   |
|:--------------------------------------|:----------------------|-------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|:--------------------|
| inverse_vol_150-21-21                 | inverse_vol           | 0.2635 |   1.7062 |        -0.1273 |      0.0200 |       -0.0792 |               0.6364 |            0.0227 | True                |
| ensemble_20-150                       | ensemble              | 0.3624 |   1.5582 |        -0.1380 |      0.0273 |       -0.0874 |               0.6136 |            0.1364 | True                |
| atr_trend_126-3.0-10-5                | atr_trend             | 0.4319 |   1.5279 |        -0.1877 |      0.0327 |       -0.1319 |               0.6364 |            0.0909 | True                |
| dual_ema_20-200                       | dual_ema              | 0.2600 |   1.5039 |        -0.1258 |      0.0200 |       -0.0804 |               0.6136 |            0.0455 | True                |
| risk_managed_momentum_126-3-21-35-200 | risk_managed_momentum | 0.6374 |   1.4662 |        -0.3278 |      0.0466 |       -0.1716 |               0.6364 |            0.3409 | False               |
| lowvol_momentum_126-5-5               | lowvol_momentum       | 0.2101 |   1.1180 |        -0.1519 |      0.0171 |       -0.0763 |               0.5909 |            0.0909 | False               |

## Selected-family ensemble funding-drag sensitivity

|   annual_funding_drag |   cagr |   sharpe |   max_drawdown |   avg_month |   median_month |   worst_month |   best_month |   positive_month_pct |   month_ge_10_pct |   months |
|----------------------:|-------:|---------:|---------------:|------------:|---------------:|--------------:|-------------:|---------------------:|------------------:|---------:|
|                0.0000 | 0.3644 |   1.6416 |        -0.1379 |      0.0273 |         0.0347 |       -0.0895 |       0.2038 |               0.5909 |            0.0909 |  44.0000 |
|                0.1000 | 0.2347 |   1.1459 |        -0.1527 |      0.0191 |         0.0302 |       -0.0975 |       0.1939 |               0.5682 |            0.0682 |  44.0000 |
|                0.2000 | 0.1172 |   0.6502 |        -0.1972 |      0.0108 |         0.0218 |       -0.1054 |       0.1841 |               0.5682 |            0.0682 |  44.0000 |

## Interpretation

Sharpe >= 1.5 is a screening threshold, not proof of future profitability. Survivorship bias, stock-perpetual funding, overnight mark-price deviations, and model-selection risk remain.