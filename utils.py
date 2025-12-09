import os, yaml, logging

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_logger(name='finboost'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    return logging.getLogger(name)
