import time
import pandas as pd
from configs.settings import SETTINGS
from feature_engineering import process_pair


def generate_simple_signal(df):
    """Simple RSI strategy placeholder."""
    last = df.iloc[-1]

    if last["rsi"] < 30:
        return "BUY"
    elif last["rsi"] > 70:
        return "SELL"
    else:
        return "HOLD"


def process_live_pair(pair):
    print(f"\n[LIVE] Processing {pair} ...")

    # 🔥 Regenerate latest engineered features
    process_pair(pair)

    path = f"data/processed_{pair}.parquet"
    df = pd.read_parquet(path)

    # 🔥 SAME FEATURES AS TRAINING + BACKTEST
    required = [
        "close",
        "rsi",
        "ma_10",
        "ma_20",
        "ma_50",
        "volatility",
        "return",
        "hour",
        "minute",
        "dayofweek"
    ]

    for col in required:
        if col not in df.columns:
            print(f"[ERROR] Missing feature {col}. Run feature_engineering again.")
            return None

    signal = generate_simple_signal(df)
    print(f"[SIGNAL] {pair} → {signal}")
    return signal


def main_loop():
    while True:
        for pair in SETTINGS["pairs"]:
            process_live_pair(pair)

        print("\nSleeping 60 seconds...\n")
        time.sleep(60)


if __name__ == "__main__":
    main_loop()
