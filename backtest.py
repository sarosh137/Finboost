import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from configs.settings import SETTINGS
from models import FinboostModel


def load_processed_data(pair):
    path = f"data/processed_{pair}.parquet"
    if not os.path.exists(path):
        print(f"[ERROR] Processed data not found for {pair}: {path}")
        return None
    return pd.read_parquet(path)


def build_sequences(df, seq_len, feature_cols):
    X = []
    for i in range(len(df) - seq_len):
        seq = df[feature_cols].iloc[i:i+seq_len].values
        X.append(seq)
    return torch.tensor(X, dtype=torch.float32)


def run_backtest_for_pair(pair):
    print(f"\n=== BACKTESTING {pair} ===")

    df = load_processed_data(pair)
    if df is None:
        return {"pair": pair, "status": "no_data"}

    # 🔥 THE CORRECT FEATURES (match feature_engineering.py)
    feature_cols = [
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

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"[ERROR] Missing features: {missing}")
        return {"pair": pair, "status": "missing_features", "missing": missing}

    seq_len = SETTINGS["model"]["seq_len"]
    X = build_sequences(df, seq_len, feature_cols)

    if len(X) == 0:
        print(f"[ERROR] Not enough data to build sequences for {pair}")
        return {"pair": pair, "status": "no_sequences"}

    # 🔥 LOAD THE CORRECT MODEL FILE
    model_path = f"models/{pair}_best.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model checkpoint not found: {model_path}")
        return {"pair": pair, "status": "missing_model"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FinboostModel(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            out = model(batch)
            next_candle = out["next_candle"].cpu().numpy().flatten()
            preds.extend(next_candle)

    # Save results
    out_path = f"backtest_results/{pair}_results.json"
    os.makedirs("backtest_results", exist_ok=True)
    json.dump({"pair": pair, "predictions": preds}, open(out_path, "w"), indent=4)

    print(f"[OK] Backtest saved → {out_path}")

    return {"pair": pair, "status": "ok", "predictions": len(preds)}


def main():
    pairs = SETTINGS["pairs"]
    results = {}

    for pair in pairs:
        results[pair] = run_backtest_for_pair(pair)

    json.dump(results, open("backtest_results/summary.json", "w"), indent=4)
    print("\n=== BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()
