import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name='Finboost', log_file='logs/finboost.log', level='INFO'):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))

    # Rotating file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    fh.setLevel(numeric_level)
    fh.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(levelname)s %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
