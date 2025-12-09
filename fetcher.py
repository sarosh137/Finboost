"""
fetcher.py — Smart TwelveData Fetcher for Finboost (FULL FILE)

★ Automatically handles TwelveData FREE tier API rate limits
★ Waits 65 seconds when limit reached
★ Retries same chunk until success
★ Downloads full 12 months of intraday data reliably
★ Supports Forex, Metals, Crypto

Requirements:
    pip install pandas requests pyarrow
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

# ---------- Load settings ----------
SETTINGS_PATH = "configs/settings.json"
if not os.path.exists(SETTINGS_PATH):
    raise RuntimeError("configs/settings.json not found.")

with open(SETTINGS_PATH, "r") as f:
    SETTINGS = json.load(f)

PAIRS = SETTINGS.get("pairs", [])
TF = SETTINGS.get("timeframe", "5m")
MONTHS = int(SETTINGS.get("history_months", 12))

TD = SETTINGS.get("twelvedata", {})
TD_KEY = TD.get("api_key", "")
TD_ENABLED = TD.get("enabled", False)

if not TD_ENABLED or TD_KEY == "":
    raise RuntimeError("Add TwelveData API key in settings.json")


# ---------- Paths ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE = "https://api.twelvedata.com/time_series"

# ---------- Helpers ----------
def map_symbol(pair):
    """Convert Finboost symbol to TwelveData format."""
    pair = pair.upper()
    if pair.endswith("USDT"):
        return f"{pair[:-4]}/USDT"
    return f"{pair[:3]}/{pair[3:]}"


def interval_to_td(tf):
    tf = tf.lower()
    if tf.endswith("m"):
        return f"{tf[:-1]}min"
    if tf.endswith("h"):
        return f"{tf[:-1]}h"
    if tf in ("1d", "d"):
        return "1day"
    return "5min"


INTERVAL = interval_to_td(TF)
CHUNK_DAYS = 30


# ---------- Candle parser ----------
def parse_twelvedata(resp):
    """Parse TwelveData response into DataFrame."""
    if resp is None or "values" not in resp:
        return None

    df = pd.DataFrame(resp["values"])

    if "datetime" not in df.columns:
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Required OHLC
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forex/Metals might not have volume
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ---------- Fetch single chunk with auto rate-limit wait ----------
def fetch_chunk(symbol, interval, start_dt, end_dt):
    """
    Fetch one chunk of data.
    If rate-limit occurs → auto wait 65 seconds → retry.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TD_KEY,
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "outputsize": 5000
    }

    while True:
        try:
            r = requests.get(BASE, params=params, timeout=30)
            data = r.json()

            # Detect rate limit
            if "message" in data and "run out of API credits" in data["message"]:
                print("  ⚠ Rate limit reached — waiting 65 seconds...")
                time.sleep(65)
                continue  # retry same chunk

            # HTTP error but retryable
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"  ⚠ HTTP {r.status_code}, retrying in 10s...")
                time.sleep(10)
                continue

            # Non-retryable error
            if r.status_code != 200:
                print(f"  ❌ HTTP {r.status_code}: {r.text}")
                return None

            return parse_twelvedata(data)

        except Exception as e:
            print(f"  ⚠ Error: {e}, retrying in 10s...")
            time.sleep(10)


# ---------- Fetch all chunks for one pair ----------
def fetch_pair(pair):
    print(f"\n------------------------------------")
    print(f"Processing {pair}")
    print("------------------------------------")

    symbol = map_symbol(pair)
    print(f"TwelveData symbol: {symbol}")

    end = datetime.utcnow()
    start = end - timedelta(days=MONTHS * 30)

    current_start = start
    all_chunks = []

    while current_start < end:
        chunk_end = min(end, current_start + timedelta(days=CHUNK_DAYS))

        print(f"  Fetching {symbol} from {current_start} → {chunk_end}")

        df_chunk = fetch_chunk(symbol, INTERVAL, current_start, chunk_end)

        if df_chunk is not None and not df_chunk.empty:
            print(f"    Retrieved rows: {len(df_chunk)}")
            all_chunks.append(df_chunk)
        else:
            print(f"    ⚠ No data in this chunk.")

        current_start = chunk_end

    if not all_chunks:
        print(f"  ❌ No data could be fetched for {pair}")
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df.reset_index(drop=True)

    return df


# ---------- Save ----------
def save_pair(df, pair):
    path = f"{DATA_DIR}/{pair}.parquet"
    df.to_parquet(path, index=False)
    print(f"  ✔ Saved: {path} ({len(df)} rows)")


# ---------- Main ----------
def fetch_all():
    print(f"TwelveData fetcher starting: {MONTHS} months, timeframe={TF}")

    for pair in PAIRS:
        try:
            df = fetch_pair(pair)
            if df is not None:
                save_pair(df, pair)
        except Exception as e:
            print(f"  ❌ Error processing {pair}: {e}")


if __name__ == "__main__":
    fetch_all()
