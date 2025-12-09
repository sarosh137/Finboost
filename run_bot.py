"""Run-time wrapper for Finboost.

This script orchestrates:
- fetching latest data (fetcher.fetch_all)
- feature engineering for latest data
- loading model(s) from models/
- generating signals
- executing via MT5 or Binance (if enabled)
- logging trades and decisions

This is a high-level starter; do not use for live trading without extensive testing.
"""

import os
import time
import json
from utils.logger import get_logger

logger = get_logger()

with open('configs/settings.json','r') as f:
    SETTINGS = json.load(f)

# Import local modules (these are scripts, not packages)
import fetcher
import feature_engineering
import model_training  # contains simple model in this template
import backtest

def generate_signals_for_pair(pair):
    # Example: load processed data and use simple rule or model to create a signal
    ppath = f'data/processed_{pair}.parquet'
    if not os.path.exists(ppath):
        logger.warning('No processed data for %s', pair)
        return None
    df = feature_engineering.process_pair(pair) if True else None
    # Placeholder: simple momentum signal, replace with model inference
    try:
        df = pd.read_parquet(ppath)
        last = df.iloc[-1]
        if last['rsi'] < 30:
            return {'pair':pair, 'signal':'buy', 'confidence':0.6}
        if last['rsi'] > 70:
            return {'pair':pair, 'signal':'sell', 'confidence':0.6}
    except Exception as e:
        logger.exception('Signal generation failed for %s', pair)
    return None

def main_loop():
    logger.info('Finboost main loop starting')
    while True:
        try:
            # fetch latest (non-blocking in template)
            # fetcher.fetch_all()
            # process
            # generate signals
            signals = []
            for pair in SETTINGS['pairs']:
                sig = generate_signals_for_pair(pair)
                if sig:
                    signals.append(sig)
                    logger.info('Signal: %s', sig)
            # TODO: execute signals via MT5/Binance
            logger.info('Sleeping for 60 seconds (template)')
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info('Stopping Finboost')
            break
        except Exception as e:
            logger.exception('Main loop error')
            time.sleep(5)

if __name__ == '__main__':
    main_loop()
