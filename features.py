#!/usr/bin/env python3
"""
model_training.py

Uses your models.py FinboostModel (TCN+Attention) if present.
Trains per-pair with time-based train/val split, checkpointing, early stopping.

Outputs:
  - models/{pair}_best.pt        (best validation checkpoint)
  - models/{pair}_final.pt       (final model after training)
  - models/{pair}_history.json   (train/val loss history)
"""

import os
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load settings
with open("configs/settings.json", "r") as f:
    SETTINGS = json.load(f)

SEQ_LEN = int(SETTINGS["model"].get("seq_len", 120))
BATCH_SIZE = int(SETTINGS["model"].get("batch_size", 64))
EPOCHS = int(SETTINGS["model"].get("epochs", 20))
LR = float(SETTINGS["model"].get("learning_rate", 1e-4))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data")
PROCESSED_PREFIX = "processed_"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VAL_RATIO = float(SETTINGS.get("validation_ratio", 0.1))  # portion at end reserved for val
PATIENCE = int(SETTINGS.get("early_stopping_patience", 5))

# Try import FinboostModel from models.py, fallback to a small conv model if not available
try:
    from models import FinboostModel  # your provided file
    MODEL_CLASS = FinboostModel
    print("Using FinboostModel from models.py")
except Exception:
    print("FinboostModel not found in models.py; falling back to a small Conv model (toy).")

    class MODEL_CLASS(nn.Module):
        def __init__(self, in_ch, hidden=64):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=3, padding=2, dilation=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU())
            self.head_ret = nn.Linear(64, 1)
            self.head_rev = nn.Linear(64, 1)
            self.head_reg = nn.Linear(64, 3)
        def forward(self, x):
            # x: B, seq_len, features
            x = x.transpose(1, 2)  # B, C, L
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            z = self.fc(x)
            return self.head_ret(z), torch.sigmoid(self.head_rev(z)), torch.softmax(self.head_reg(z), dim=-1)


# ---------- Dataset ----------
class SlidingDataset(Dataset):
    """
    Builds sliding windows from processed parquet files.
    Expects processed files with numeric feature columns and a 'close' column and a 'ret' or 'return' column for regression target.
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = SEQ_LEN):
        # df must be indexed by datetime or have a datetime column already set as index
        self.seq_len = seq_len
        # pick numeric feature columns automatically
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove columns that are purely target-like if present (we still include 'close' and indicators)
        # We'll look for a return column name
        self.return_col = None
        for candidate in ("ret", "return", "returns", "rtn"):
            if candidate in numeric_cols:
                self.return_col = candidate
                break
        # ensure 'close' present
        if "close" in numeric_cols:
            pass
        else:
            raise ValueError("Processed data must contain 'close' numeric column.")

        # remove any index-based numeric columns if present
        # select features: all numeric columns except 'return' target (we'll predict next-step return), and exclude 'volume' optionally
        features = [c for c in numeric_cols if c not in (self.return_col,)]
        # keep deterministic order
        features = sorted(features)
        self.features = features

        self.X = df[self.features].values.astype(np.float32)
        # build target arrays
        if self.return_col is not None:
            ret = df[self.return_col].values.astype(np.float32)
        else:
            # create next-step return from close if not present
            close = df["close"].values.astype(np.float32)
            ret = np.concatenate([np.diff(close) / close[:-1], [0.0]])
        # shift next-step return
        self.y_ret = np.roll(ret, -1).astype(np.float32)

        # placeholder binary reversal and regime labels (0)
        self.y_rev = np.zeros_like(self.y_ret, dtype=np.float32)
        self.y_reg = np.zeros(len(self.y_ret), dtype=np.int64)

        self.len = max(0, len(self.X) - seq_len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]           # (seq_len, feats)
        y_ret = self.y_ret[idx + self.seq_len]        # scalar
        y_rev = self.y_rev[idx + self.seq_len]        # scalar
        y_reg = self.y_reg[idx + self.seq_len]        # int
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_ret, dtype=torch.float32).unsqueeze(0), torch.tensor(y_rev, dtype=torch.float32).unsqueeze(0), torch.tensor(y_reg, dtype=torch.long)


# ---------- Utilities ----------
def train_epoch(model, loader, optimizer, loss_fns):
    model.train()
    total_loss = 0.0
    count = 0
    for X, y_ret, y_rev, y_reg in loader:
        X = X.to(DEVICE); y_ret = y_ret.to(DEVICE); y_rev = y_rev.to(DEVICE); y_reg = y_reg.to(DEVICE)
        optimizer.zero_grad()
        pred_ret, pred_rev, pred_reg = model(X)
        l = loss_fns['ret'](pred_ret, y_ret) + 0.5 * loss_fns['rev'](pred_rev, y_rev) + 0.2 * loss_fns['reg'](pred_reg, y_reg)
        l.backward()
        optimizer.step()
        total_loss += l.item()
        count += 1
    return total_loss / max(1, count)


def val_epoch(model, loader, loss_fns):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for X, y_ret, y_rev, y_reg in loader:
            X = X.to(DEVICE); y_ret = y_ret.to(DEVICE); y_rev = y_rev.to(DEVICE); y_reg = y_reg.to(DEVICE)
            pred_ret, pred_rev, pred_reg = model(X)
            l = loss_fns['ret'](pred_ret, y_ret) + 0.5 * loss_fns['rev'](pred_rev, y_rev) + 0.2 * loss_fns['reg'](pred_reg, y_reg)
            total_loss += l.item()
            count += 1
    return total_loss / max(1, count)


def scale_train_val(train_X, val_X):
    # compute mean/std on train and apply to both
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std
    return train_X, val_X, mean, std


# ---------- Main training flow for one pair ----------
def train_pair(pair):
    pfile = DATA_DIR / f"{PROCESSED_PREFIX}{pair}.parquet"
    if not pfile.exists():
        print(f"No processed data for {pair}, skipping.")
        return

    print(f"\n=== TRAIN {pair} ===")
    df = pd.read_parquet(pfile)

    # Ensure datetime index (if there is a datetime column, set it)
    if "datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    # if index is unnamed datetime-like, it's fine.

    # Drop NaNs early
    df = df.dropna().reset_index(drop=False)
    # keep datetime column separate for time-based split
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        # if index was datetime and reset_index removed it to column
        if df.columns[0].lower().startswith("time"):
            df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
        else:
            # fallback create incremental datetime (not ideal)
            df["datetime"] = pd.date_range(end=datetime.utcnow(), periods=len(df), freq='5T')

    # sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # time-based split
    n = len(df)
    if n < SEQ_LEN + 10:
        print(f"Not enough rows for {pair} (need > {SEQ_LEN + 10}), found {n}. Skipping.")
        return
    split_idx = int(n * (1.0 - VAL_RATIO))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Build sliding datasets to fit scaler values and shapes
    # select numeric columns for features detection
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    # choose features: all numeric except next-return if present
    # We'll remove possible target column names
    for bad in ("y", "target"):
        if bad in num_cols:
            num_cols.remove(bad)
    # if 'ret' or 'return' exists remove from feature list (we compute next-step ret separately)
    for candidate in ("ret", "return", "returns", "rtn"):
        if candidate in num_cols:
            num_cols.remove(candidate)
    # We must keep 'close' and indicator columns (they remain)
    features = [c for c in sorted(num_cols) if c != "index"]  # deterministic order
    if "close" not in features:
        print(f"Processed file for {pair} must contain 'close' column. Skipping.")
        return

    # Build X arrays for scaling: we will scale on flattened windows to compute per-feature mean/std
    # But for simplicity compute mean/std on raw features columns in train_df
    train_X_vals = train_df[features].values.astype(np.float32)
    val_X_vals = val_df[features].values.astype(np.float32)
    train_X_vals, val_X_vals, mean, std = scale_train_val(train_X_vals, val_X_vals)

    # Now reconstruct train/val DataFrames with scaled features
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    train_df_scaled[features] = train_X_vals
    val_df_scaled[features] = val_X_vals

    # build sliding datasets
    train_ds = SlidingDataset(train_df_scaled.set_index("datetime"), seq_len=SEQ_LEN)
    val_ds = SlidingDataset(val_df_scaled.set_index("datetime"), seq_len=SEQ_LEN)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # model init
    model = MODEL_CLASS(in_ch=len(train_ds.features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    loss_fns = {
        "ret": nn.MSELoss(),
        "rev": nn.BCELoss(),
        "reg": nn.CrossEntropyLoss()
    }

    best_val = float("inf")
    best_epoch = -1
    history = {"train_loss": [], "val_loss": []}
    stall = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fns)
        val_loss = val_epoch(model, val_loader, loss_fns)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[{pair}] Epoch {epoch}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}  time={(time.time()-t0):.1f}s")

        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_path = MODEL_DIR / f"{pair}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "features": train_ds.features
            }, best_path)
            print(f"  -> Saved best model to {best_path}")
            stall = 0
        else:
            stall += 1

        # early stopping
        if stall >= PATIENCE:
            print(f"  Early stopping after {stall} epochs without improvement.")
            break

    # final save
    final_path = MODEL_DIR / f"{pair}_final.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "features": train_ds.features
    }, final_path)
    print(f"Saved final model to {final_path}. Best val {best_val:.6f} at epoch {best_epoch}.")

    # save history
    hist_path = MODEL_DIR / f"{pair}_history.json"
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved training history to {hist_path}")


def train_all():
    pairs = SETTINGS.get("pairs", [])
    for pair in pairs:
        try:
            train_pair(pair)
        except Exception as e:
            print(f"ERROR training {pair}: {e}")


if __name__ == "__main__":
    train_all()
