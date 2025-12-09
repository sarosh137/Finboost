import numpy as np
import pandas as pd

def pct_change_fillna(df, cols=['close']):
    df = df.copy()
    for c in cols:
        df[c+'_ret'] = df[c].pct_change().fillna(0)
    return df

def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)
