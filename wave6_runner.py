import numpy as np

import wave6_trade_level_2pct as audit
from wave6_adjusted_stock_data import stock_data


def fast_bootstrap_lower(values, seed=42):
    if len(values) < 10:
        return float('nan')
    source = values.to_numpy(float)
    rng = np.random.default_rng(seed)
    means = rng.choice(source, size=(500, len(source)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05))


audit.stock_data = stock_data
audit.bootstrap_lower = fast_bootstrap_lower

audit.main()
