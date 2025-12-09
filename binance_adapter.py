# Binance adapter (stub). Uses python-binance if installed.
try:
    from binance.client import Client
except Exception:
    Client = None

client = None

def init_binance(api_key=None, api_secret=None):
    global client
    if Client is None:
        raise RuntimeError('python-binance not installed')
    client = Client(api_key, api_secret)

def place_order(symbol, side, quantity, order_type='MARKET'):
    if client is None:
        raise RuntimeError('Binance client not initialized')
    # Note: error handling, position sizing, and risk checks needed in production.
    return client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
