import ccxt
import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Load settings.json
# --------------------------------------------------------------------------
with open("configs/settings.json", "r") as f:
    SETTINGS = json.load(f)

PAIRS = SETTINGS["pairs"]
MONTHS = SETTINGS["history_months"]
TIMEFRAME = SETTINGS["timeframe"]  # '5m'

# --------------------------------------------------------------------------
# Create data directory
# --------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)

# --------------------------------------------------------------------------
# Configure CCXT Binance
# --------------------------------------------------------------------------
exchange = ccxt.binance({
    "enableRateLimit": True
})

# Map timeframe string to ccxt format
TF_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
}
TF = TF_MAP.get(TIMEFRAME, "5m")

# --------------------------------------------------------------------------
# Helper function: generate 'since' timestamp for last X months
# --------------------------------------------------------------------------
def months_to_timestamp(months):
    end = datetime.utcnow()
    start = end - timedelta(days=30 * months)
    return int(start.timestamp() * 1000)


# --------------------------------------------------------------------------
# Fetch one pair from Binance with chunking
# --------------------------------------------------------------------------
def fetch_crypto(pair):
    print(f"\n-----------------------------------------")
    print(f"Fetching {pair} from Binance ({MONTHS} months, {TF})")
    print("-----------------------------------------")

    since = months_to_timestamp(MONTHS)
    all_rows = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=pair,
                timeframe=TF,
                since=since,
                limit=1000,       # Binance max per request
            )

            if not ohlcv:
                break

            all_rows.extend(ohlcv)

            # Move cursor forward (last candle timestamp)
            since = ohlcv[-1][0] + 1

            # Respect API limits
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"❌ Error fetching {pair}: {e}")
            break

    # If no data → return None
    if len(all_rows) == 0:
        print(f"⚠ No data for {pair}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Save Parquet
    file_path = f"data/{pair}.parquet"
    df.to_parquet(file_path, index=False)

    print(f"✔ Saved {file_path} ({len(df)} rows)")
    return df


# --------------------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------------------
def fetch_all_crypto():
    print(f"🚀 Binance Crypto Fetcher Starting")
    print(f"Pairs: {PAIRS}")
    print(f"History: {MONTHS} months")
    print(f"Timeframe: {TF}\n")

    success = []
    failed = []

    for pair in PAIRS:
        try:
            df = fetch_crypto(pair)
            if df is not None and len(df) > 0:
                success.append(pair)
            else:
                failed.append(pair)
        except Exception as e:
            print(f"❌ Critical error for {pair}: {e}")
            failed.append(pair)

    # Save results
    with open("data/fetch_crypto_success.json", "w") as f:
        json.dump(success, f, indent=4)

    with open("data/fetch_crypto_fail.json", "w") as f:
        json.dump(failed, f, indent=4)

    print("\n================ RESULTS ================")
    print("✔ Successful:", success)
    print("❌ Failed:", failed)
    print("=========================================\n")


# --------------------------------------------------------------------------
# RUN SCRIPT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    fetch_all_crypto()
