import pandas as pd
import wave8_adaptive_trend as w8


def normalize_70_30_fixed(pos: pd.DataFrame) -> pd.DataFrame:
    longs = pos.gt(0).astype(float)
    shorts = pos.lt(0).astype(float)
    nl = longs.sum(axis=1).replace(0, pd.NA)
    ns = shorts.sum(axis=1).replace(0, pd.NA)
    long_w = longs.div(nl, axis=0).mul(0.70)
    short_w = shorts.div(ns, axis=0).mul(-0.30)
    return long_w.fillna(0.0).add(short_w.fillna(0.0), fill_value=0.0)


w8.normalize_70_30 = normalize_70_30_fixed

if __name__ == '__main__':
    w8.main()
