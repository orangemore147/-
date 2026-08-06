from pathlib import Path

import numpy as np

import wave6_trade_level_2pct as audit
from wave6_adjusted_stock_data import stock_data


original_grid = audit.grid


def fast_bootstrap_lower(values, seed=42):
    if len(values) < 10:
        return float('nan')
    source = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = rng.choice(source, size=(1000, len(source)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05))


audit.OUT = Path('results_wave6_stock')
audit.stock_data = stock_data
audit.crypto_data = lambda: {}
audit.grid = lambda: [config for config in original_grid() if config.market == 'stock']
audit.bootstrap_lower = fast_bootstrap_lower

audit.main()
