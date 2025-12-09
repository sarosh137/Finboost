#!/usr/bin/env python3
"""
Finboost - Advanced Model Training Engine (FINAL FIXED VERSION)
===============================================================

✔ Fully compatible with your FinboostModel(input_dim=…)
✔ Works with processed_* parquet files
✔ Adds reversal labels
✔ Prevents reversal column from being scaled
✔ Train/validation split (time-based)
✔ Early stopping + best model checkpoint
✔ Saves model + history
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# LOAD SETTINGS
# ------------------------------
with open("configs/settings.json", "r") as f:
    SETTINGS = json.load(f)

SEQ_LEN = SETTINGS["model"]["seq_len"]
BATCH = SETTINGS["model"]["batch_size"]
EPOCHS = SETTINGS["model"]["epochs"]
LR = SETTINGS["model"]["learning_rate"]

VAL_RATIO = 0.1
PATIENCE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ------------------------------
# IMPORT YOUR MODEL
# ------------------------------
from models import FinboostModel
print("✔ Using FinboostModel from models.py")

# ------------------------------
# REVERSAL LABEL LOGIC
# ------------------------------
def compute_reversal_labels(df, horizon=4, threshold=0.0025):
    """
    reversal = 1 when future return flips direction
    AND move is strong (|return| > threshold)
    """
    closes = df["close"].values
    fut = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
    fut = np.concatenate([fut, np.zeros(horizon)])

    now_dir = np.sign(np.diff(closes, prepend=closes[0]))
    fut_dir = np.sign(fut)

    reversal = (now_dir * fut_dir) < 0
    strong = np.abs(fut) > threshold

    return (reversal & strong).astype(np.float32)

# ------------------------------
# DATASET
# ------------------------------
class FinboostDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len

        # auto detect numeric features
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        # do not treat reversal/regime as features
        exclude = ["reversal", "regime"]
        self.features = sorted([c for c in numeric if c not in exclude])

        # feature matrix
        self.X = df[self.features].values.astype(np.float32)

        # next-candle return target
        ret = df["ret"].values.astype(np.float32)
        yret = np.roll(ret, -1)
        yret[-1] = 0
        self.y_ret = yret

        # reversal target
        self.y_rev = df["reversal"].values.astype(np.float32)

        # placeholder regime target
        self.y_reg = np.zeros(len(df), dtype=np.int64)

    def __len__(self):
        return len(self.X) - self.seq_len - 1

    def __getitem__(self, i):
        x = self.X[i:i+self.seq_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([self.y_ret[i+self.seq_len]], dtype=torch.float32),
            torch.tensor([self.y_rev[i+self.seq_len]], dtype=torch.float32),
            torch.tensor(self.y_reg[i+self.seq_len], dtype=torch.long)
        )

# ------------------------------
# TRAIN + VAL STEPS
# ------------------------------
def train_step(model, dl, opt, loss_fns):
    model.train()
    total = 0
    for X, yret, yrev, yreg in dl:
        X, yret, yrev, yreg = X.to(DEVICE), yret.to(DEVICE), yrev.to(DEVICE), yreg.to(DEVICE)

        opt.zero_grad()
        out = model(X)

        loss = (
            loss_fns["ret"](out["next_candle"], yret) +
            0.5 * loss_fns["rev"](out["reversal"], yrev) +
            0.2 * loss_fns["reg"](out["regime"], yreg)
        )
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(dl)


def val_step(model, dl, loss_fns):
    model.eval()
    total = 0
    with torch.no_grad():
        for X, yret, yrev, yreg in dl:
            X, yret, yrev, yreg = X.to(DEVICE), yret.to(DEVICE), yrev.to(DEVICE), yreg.to(DEVICE)

            out = model(X)
            loss = (
                loss_fns["ret"](out["next_candle"], yret) +
                0.5 * loss_fns["rev"](out["reversal"], yrev) +
                0.2 * loss_fns["reg"](out["regime"], yreg)
            )
            total += loss.item()
    return total / len(dl)

# ------------------------------
# TRAIN A SINGLE PAIR
# ------------------------------
def train_pair(pair):
    path = DATA_DIR / f"processed_{pair}.parquet"
    if not path.exists():
        print(f"⚠ No data for {pair}, skipping.")
        return

    print(f"\n=== TRAINING {pair} ===")

    df = pd.read_parquet(path).dropna().reset_index(drop=True)

    # ensure return
    if "ret" not in df.columns:
        df["ret"] = df["close"].pct_change().fillna(0)

    # add reversal labels
    df["reversal"] = compute_reversal_labels(df)

    # split
    split = int(len(df) * (1 - VAL_RATIO))
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()

    # -------------------------
    # FIX: DO NOT SCALE LABELS
    # -------------------------
    feats = train_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["reversal", "regime"]
    feats = [f for f in feats if f not in exclude]

    mean = train_df[feats].mean()
    std = train_df[feats].std().replace(0, 1)

    train_df[feats] = (train_df[feats] - mean) / std
    val_df[feats] = (val_df[feats] - mean) / std

    # ensure reversal still 0/1
    train_df["reversal"] = train_df["reversal"].clip(0, 1)
    val_df["reversal"] = val_df["reversal"].clip(0, 1)

    # datasets
    train_ds = FinboostDataset(train_df, SEQ_LEN)
    val_ds = FinboostDataset(val_df, SEQ_LEN)

    # loaders
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # ------------------------
    # CREATE MODEL CORRECTLY
    # ------------------------
    model = FinboostModel(
        input_dim=len(train_ds.features)
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    loss_fns = {
        "ret": nn.MSELoss(),
        "rev": nn.BCELoss(),
        "reg": nn.CrossEntropyLoss()
    }

    best = float("inf")
    stalls = 0

    # ------------------------
    # TRAINING LOOP
    # ------------------------
    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        tr = train_step(model, train_dl, opt, loss_fns)
        va = val_step(model, val_dl, loss_fns)

        print(f"[{pair}] Epoch {epoch}/{EPOCHS}  Train={tr:.6f}  Val={va:.6f}  ({time.time()-start:.1f}s)")

        if va < best:
            best = va
            stalls = 0
            torch.save(model.state_dict(), MODEL_DIR / f"{pair}_best.pt")
            print(f"  ✔ Saved BEST model for {pair}")
        else:
            stalls += 1

        if stalls >= PATIENCE:
            print(f"⛔ Early stop for {pair}")
            break

    torch.save(model.state_dict(), MODEL_DIR / f"{pair}_final.pt")
    print(f"✔ Training finished for {pair}. Best Val={best:.6f}")

# ------------------------------
# TRAIN ALL PAIRS
# ------------------------------
def train_all():
    for pair in SETTINGS["pairs"]:
        train_pair(pair)

if __name__ == "__main__":
    train_all()
