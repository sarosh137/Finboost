"""
Finboost Backtest Engine
-----------------------------------------
✓ Loads trained Finboost model
✓ Evaluates on processed_<pair>.parquet files
✓ Simulates SL/TP, reversal exit, 5m scalping
✓ Outputs metrics | PnL | Plot-ready logs
-----------------------------------------
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from utils.logger import get_logger

# Load settings
with open("configs/settings.json", "r") as f:
    SETTINGS = json.load(f)

SEQ_LEN = SETTINGS["model"]["seq_len"]
SL_TP_RATIO = SETTINGS["execution"]["sl_tp_ratio"]
MAX_RISK = SETTINGS["execution"]["max_trade_risk_pct"] / 100
TIMEFRAME = SETTINGS["timeframe"]

logger = get_logger("Backtest", SETTINGS["logging"]["log_file"], SETTINGS["logging"]["level"])


# ----------------------------
# Load Model (PyTorch)
# ----------------------------
def load_model(model_path, input_dim):
    """Loads a trained PyTorch model."""
    from models import FinboostModel  # your TCN/Attention model

    model = FinboostModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ----------------------------
# Build input sequences
# ----------------------------
def create_sequences(df, feature_cols):
    X = []
    indices = []

    for i in range(len(df) - SEQ_LEN):
        X.append(df[feature_cols].iloc[i : i + SEQ_LEN].values)
        indices.append(i + SEQ_LEN)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    return X, indices


# ----------------------------
# Run model predictions
# ----------------------------
def predict(model, X):
    """Run neural network predictions for:
       next_candle, reversal, regime.
    """
    with torch.no_grad():
        preds = model(X)
    return preds


# ----------------------------
# Trading Logic
# ----------------------------
def run_strategy(df, preds, idx_list):
    balance = 10000
    equity_curve = []
    trades = []

    df["pred_ret"] = np.nan
    df["pred_rev"] = np.nan
    df["signal"] = 0

    next_candle, reversal_prob, regime = preds

    next_candle = next_candle.squeeze().numpy()
    reversal_prob = reversal_prob.squeeze().numpy()

    for pred, rev, idx in zip(next_candle, reversal_prob, idx_list):

        df.loc[idx, "pred_ret"] = pred
        df.loc[idx, "pred_rev"] = rev

        # Signal logic
        if pred > 0 and rev < 0.4:
            df.loc[idx, "signal"] = 1  # buy
        elif pred < 0 and rev < 0.4:
            df.loc[idx, "signal"] = -1  # sell

    # ----------------------------
    # Simulated Trading
    # ----------------------------
    position = 0
    entry_price = 0

    for i in range(SEQ_LEN, len(df)):
        sig = df.loc[i, "signal"]
        price = df.loc[i, "close"]

        # Open a new trade
        if position == 0 and sig != 0:

            risk_amount = balance * MAX_RISK
            qty = risk_amount / price

            position = sig
            entry_price = price

            sl = entry_price * (0.99 if sig == 1 else 1.01)
            tp = entry_price * (1 + SL_TP_RATIO * 0.01) if sig == 1 else entry_price * (1 - SL_TP_RATIO * 0.01)

            trades.append(["ENTRY", i, price, sig])

        # Manage open trade
        if position != 0:

            # reversal exit
            if df.loc[i, "pred_rev"] > 0.65:
                pnl = (price - entry_price) * position * qty
                balance += pnl
                trades.append(["REV_EXIT", i, price, pnl])
                position = 0
                continue

            # take profit
            if (position == 1 and price >= tp) or (position == -1 and price <= tp):
                pnl = (tp - entry_price) * position * qty
                balance += pnl
                trades.append(["TP", i, tp, pnl])
                position = 0
                continue

            # stop loss
            if (position == 1 and price <= sl) or (position == -1 and price >= sl):
                pnl = (sl - entry_price) * position * qty
                balance += pnl
                trades.append(["SL", i, sl, pnl])
                position = 0
                continue

        equity_curve.append(balance)

    return balance, trades, equity_curve


# ----------------------------
# Full Backtest Runner
# ----------------------------
def backtest_pair(pair):
    logger.info(f"▶ Backtesting {pair}")

    df_path = f"data/processed_{pair}.parquet"

    if not os.path.exists(df_path):
        logger.error(f"Missing processed file: {df_path}")
        return None

    df = pd.read_parquet(df_path)

    feature_cols = ["open", "high", "low", "close", "volume", "rsi", "atr", "macd"]

    model_path = f"models/{pair}_model.pt"
    if not os.path.exists(model_path):
        logger.error(f"Missing model for {pair}")
        return None

    model = load_model(model_path, input_dim=len(feature_cols))

    X, idx_list = create_sequences(df, feature_cols)
    preds = predict(model, X)

    final_balance, trades, equity_curve = run_strategy(df, preds, idx_list)

    results = {
        "pair": pair,
        "final_balance": final_balance,
        "total_return_%": round(((final_balance - 10000) / 10000) * 100, 2),
        "num_trades": len(trades),
        "trades": trades
    }

    # Save result file
    save_path = f"backtest_results/{pair}_results.json"
    os.makedirs("backtest_results", exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"✓ Backtest complete for {pair}")
    logger.info(f"Final Balance: {final_balance:.2f}")
    logger.info(f"Saved results: {save_path}")

    return results


# ----------------------------
# Run all pairs
# ----------------------------
def backtest_all():
    all_results = []

    for pair in SETTINGS["pairs"]:
        res = backtest_pair(pair)
        if res:
            all_results.append(res)

    logger.info("ALL BACKTESTS DONE")
    return all_results


if __name__ == "__main__":
    backtest_all()
