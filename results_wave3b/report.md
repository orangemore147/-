# Wave 3B — static and slow funding carry

- Data: 2023-01-01 00:00:00+00:00 to 2026-07-31 23:00:00+00:00
- Long spot / short equal-notional perpetual
- Includes initial entry and final exit costs
- Train selection: 2023–2024; OOS: 2025–2026-07

## Train-selected models — OOS

| candidate                 | family    | cost_profile     |   cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   funding_return_sum |   basis_return_sum |   annual_turnover |   active_time_pct | passes_oos   |
|:--------------------------|:----------|:-----------------|-------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|---------------------:|-------------------:|------------------:|------------------:|:-------------|
| fixed_ALL10               | fixed     | taker            | 0.0225 |   2.0192 |        -0.0046 |      0.0019 |       -0.0027 |               0.7895 |            0.0000 |               0.0363 |             0.0009 |            0.6326 |            1.0000 | True         |
| slowrank_lb90_k10_d90_th0 | slow_rank | taker            | 0.0252 |   1.4102 |        -0.0069 |      0.0021 |       -0.0032 |               0.7895 |            0.0000 |               0.0512 |             0.0012 |            4.0596 |            1.0000 | True         |
| fixed_ALL10               | fixed     | optimistic_limit | 0.0228 |   2.0651 |        -0.0046 |      0.0019 |       -0.0027 |               0.7895 |            0.0000 |               0.0363 |             0.0009 |            0.6326 |            1.0000 | True         |
| slowrank_lb90_k10_d90_th0 | slow_rank | optimistic_limit | 0.0277 |   1.5724 |        -0.0060 |      0.0023 |       -0.0023 |               0.7895 |            0.0000 |               0.0512 |             0.0012 |            4.0596 |            1.0000 | True         |

## Top 20 OOS results (diagnostic only; not valid for model selection)

| candidate                   | family    | cost_profile     |   cagr |   sharpe |   max_drawdown |   avg_month |   worst_month |   positive_month_pct |   month_ge_10_pct |   funding_return_sum |   basis_return_sum |   annual_turnover |   active_time_pct |
|:----------------------------|:----------|:-----------------|-------:|---------:|---------------:|------------:|--------------:|---------------------:|------------------:|---------------------:|-------------------:|------------------:|------------------:|
| fixed_BTC_ETH               | fixed     | optimistic_limit | 0.0369 |   4.5043 |        -0.0054 |      0.0030 |       -0.0019 |               0.8421 |            0.0000 |               0.0587 |             0.0000 |            0.6326 |            1.0000 |
| fixed_BTC_ETH               | fixed     | taker            | 0.0365 |   4.4127 |        -0.0054 |      0.0030 |       -0.0019 |               0.8421 |            0.0000 |               0.0587 |             0.0000 |            0.6326 |            1.0000 |
| fixed_BTC                   | fixed     | optimistic_limit | 0.0396 |   3.9075 |        -0.0044 |      0.0032 |       -0.0018 |               0.8421 |            0.0000 |               0.0625 |             0.0004 |            0.6326 |            1.0000 |
| fixed_BTC                   | fixed     | taker            | 0.0393 |   3.8432 |        -0.0044 |      0.0032 |       -0.0018 |               0.8421 |            0.0000 |               0.0625 |             0.0004 |            0.6326 |            1.0000 |
| fixed_ETH                   | fixed     | optimistic_limit | 0.0341 |   3.1130 |        -0.0067 |      0.0028 |       -0.0033 |               0.8421 |            0.0000 |               0.0549 |            -0.0004 |            0.6326 |            1.0000 |
| fixed_ETH                   | fixed     | taker            | 0.0337 |   3.0612 |        -0.0067 |      0.0028 |       -0.0033 |               0.8421 |            0.0000 |               0.0549 |            -0.0004 |            0.6326 |            1.0000 |
| slowrank_lb30_k3_d90_th0.5  | slow_rank | optimistic_limit | 0.0210 |   2.6767 |        -0.0036 |      0.0017 |       -0.0014 |               0.4211 |            0.0000 |               0.0380 |             0.0010 |            2.7412 |            0.4489 |
| slowrank_lb30_k3_d90_th0    | slow_rank | optimistic_limit | 0.0369 |   2.6225 |        -0.0040 |      0.0030 |       -0.0018 |               0.8421 |            0.0000 |               0.0660 |             0.0011 |            4.4281 |            1.0000 |
| slowrank_lb30_k3_d90_th0    | slow_rank | taker            | 0.0341 |   2.3547 |        -0.0048 |      0.0028 |       -0.0026 |               0.8421 |            0.0000 |               0.0660 |             0.0011 |            4.4281 |            1.0000 |
| slowrank_lb30_k3_d90_th0.5  | slow_rank | taker            | 0.0193 |   2.3480 |        -0.0044 |      0.0016 |       -0.0020 |               0.3684 |            0.0000 |               0.0380 |             0.0010 |            2.7412 |            0.4489 |
| slowrank_lb180_k5_d90_th0.5 | slow_rank | optimistic_limit | 0.0298 |   2.3431 |        -0.0029 |      0.0024 |       -0.0005 |               0.5263 |            0.0000 |               0.0504 |             0.0020 |            2.6568 |            0.6049 |
| slowrank_lb180_k5_d90_th0   | slow_rank | optimistic_limit | 0.0352 |   2.3074 |        -0.0045 |      0.0029 |       -0.0023 |               0.8421 |            0.0000 |               0.0602 |             0.0018 |            3.1629 |            1.0000 |
| slowrank_lb90_k5_d60_th0    | slow_rank | optimistic_limit | 0.0336 |   2.2387 |        -0.0034 |      0.0028 |       -0.0023 |               0.8947 |            0.0000 |               0.0618 |             0.0015 |            4.9341 |            1.0000 |
| slowrank_lb180_k3_d90_th0.5 | slow_rank | optimistic_limit | 0.0298 |   2.2346 |        -0.0034 |      0.0025 |       -0.0008 |               0.5789 |            0.0000 |               0.0498 |             0.0019 |            2.3195 |            0.6049 |
| fixed_MAJORS3               | fixed     | optimistic_limit | 0.0216 |   2.2180 |        -0.0092 |      0.0018 |       -0.0049 |               0.7368 |            0.0000 |               0.0353 |            -0.0000 |            0.6326 |            1.0000 |
| slowrank_lb180_k5_d90_th0.5 | slow_rank | taker            | 0.0281 |   2.1639 |        -0.0038 |      0.0023 |       -0.0013 |               0.5263 |            0.0000 |               0.0504 |             0.0020 |            2.6568 |            0.6049 |
| fixed_MAJORS3               | fixed     | taker            | 0.0212 |   2.1636 |        -0.0092 |      0.0018 |       -0.0049 |               0.7368 |            0.0000 |               0.0353 |            -0.0000 |            0.6326 |            1.0000 |
| slowrank_lb180_k5_d90_th0   | slow_rank | taker            | 0.0333 |   2.1549 |        -0.0052 |      0.0027 |       -0.0030 |               0.8421 |            0.0000 |               0.0602 |             0.0018 |            3.1629 |            1.0000 |
| slowrank_lb180_k5_d60_th0   | slow_rank | optimistic_limit | 0.0324 |   2.0974 |        -0.0033 |      0.0027 |       -0.0021 |               0.8947 |            0.0000 |               0.0581 |             0.0017 |            4.1750 |            1.0000 |
| slowrank_lb180_k3_d90_th0.5 | slow_rank | taker            | 0.0284 |   2.0953 |        -0.0042 |      0.0023 |       -0.0014 |               0.5263 |            0.0000 |               0.0498 |             0.0019 |            2.3195 |            0.6049 |

## Pass rule

CAGR > 0, Sharpe >= 1, max drawdown >= -20%, positive months >= 55%, average month > 0.