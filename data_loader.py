import os
import pandas as pd
import glob

def load_csv_symbol(path):
    # Expect CSV with 'timestamp,open,high,low,close,volume' header
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_all_symbols(data_dir):
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    symbols = {}
    for f in files:
        symbols[os.path.basename(f).replace('.csv','')] = load_csv_symbol(f)
    return symbols
