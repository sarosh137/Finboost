# MT5 adapter (stub). This file shows how to initialize MT5 and place an order.
# WARNING: This is a simplified example. Test extensively on demo servers.

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

def init_mt5(path=None):
    if mt5 is None:
        raise RuntimeError('MetaTrader5 package not available.')
    if path:
        mt5.initialize(path)
    else:
        mt5.initialize()

def place_order(symbol, volume, order_type='buy', sl=None, tp=None):
    # Very simplified order wrapper. Replace with production-quality logic.
    if mt5 is None:
        raise RuntimeError('MT5 not available.')
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type=='buy' else tick.bid
    # Build request...
    # For safety: not implemented in starter.
    return {'status':'stub','symbol':symbol,'price':price}
