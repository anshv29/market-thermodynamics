"""
Transformer Training and Inference Runner
==========================================
1. Loads latent states z(t) and regime labels from database
2. Builds sequence dataset (sliding window)
3. Trains the transformer to predict z(t+1) and r(t+1)
4. Evaluates regime prediction accuracy
5. Saves model for use by RL agent
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import text
from tqdm import tqdm

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from db.connection import get_engine
from models.transformer import MarketTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_START  = "2006-01-01"
TRAIN_END    = "2018-12-31"
VAL_START    = "2019-01-01"
VAL_END      = "2019-12-31"
SEQ_LEN      = 60     # days of history fed to transformer
PATCH_SIZE   = 5      # days per patch token
LATENT_DIM   = 16
N_REGIMES    = 6
EPOCHS       = 40
LR           = 3e-4
BATCH_SIZE   = 32
D_MODEL      = 128
N_HEADS      = 8
N_LAYERS     = 4
LAMBDA_REG   = 0.5    # weight on regression loss
LAMBDA_CLS   = 1.0    # weight on classification loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class MarketSequenceDataset(Dataset):
    """
    Sliding window dataset.
    Each sample: last SEQ_LEN days of z(t) → predict z(t+1) and r(t+1)
    """

    def __init__(self, z_array: np.ndarray, regime_array: np.ndarray,
                 seq_len: int = SEQ_LEN):
        self.z       = torch.tensor(z_array,      dtype=torch.float32)
        self.regimes = torch.tensor(regime_array, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.z) - self.seq_len - 1

    def __getitem__(self, idx):
        x        = self.z[idx : idx + self.seq_len]
        z_target = self.z[idx + self.seq_len]
        r_target = self.regimes[idx + self.seq_len]
        return x, z_target, r_target


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_latent_data() -> tuple:
    """Load latent states and regime labels from database."""
    engine = get_engine()

    print("Loading latent states and regime labels...")
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT l.date, l.z_vector, r.regime_label
            FROM latent_state_daily l
            JOIN regime_daily r ON l.date = r.date
            ORDER BY l.date
        """), conn)

    df["date"] = pd.to_datetime(df["date"])
    df["z"] = df["z_vector"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )

    z_array      = np.array(df["z"].tolist(), dtype=np.float32)
    regime_array = df["regime_label"].values.astype(np.int64)
    dates        = df["date"].values

    print(f"Loaded {len(df)} days of latent states.")
    print(f"Regime distribution: {np.bincount(regime_array)}")

    return z_array, regime_array, dates, df["date"]


# ── Training ──────────────────────────────────────────────────────────────────

def train_transformer(z_array, regime_array, dates):
    """Train the transformer on the training period."""

    # Split by date
    date_series = pd.to_datetime(dates)
    train_mask  = (date_series >= TRAIN_START) & (date_series <= TRAIN_END)
    val_mask    = (date_series >= VAL_START)   & (date_series <= VAL_END)

    # Need SEQ_LEN buffer before train start
    train_start_idx = np.where(train_mask)[0][0]
    train_end_idx   = np.where(train_mask)[0][-1]
    val_start_idx   = max(np.where(val_mask)[0][0] - SEQ_LEN, 0)
    val_end_idx     = np.where(val_mask)[0][-1]

    z_train      = z_array[max(0, train_start_idx - SEQ_LEN):train_end_idx + 1]
    regime_train = regime_array[max(0, train_start_idx - SEQ_LEN):train_end_idx + 1]
    z_val        = z_array[val_start_idx:val_end_idx + 1]
    regime_val   = regime_array[val_start_idx:val_end_idx + 1]

    train_dataset = MarketSequenceDataset(z_train, regime_train)
    val_dataset   = MarketSequenceDataset(z_val,   regime_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = MarketTransformer(
        latent_dim=LATENT_DIM,
        n_regimes=N_REGIMES,
        seq_len=SEQ_LEN,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    reg_loss   = nn.MSELoss()
    cls_loss   = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for x, z_target, r_target in train_loader:
            optimizer.zero_grad()
            z_pred, regime_logits = model(x)

            loss_reg = reg_loss(z_pred, z_target)
            loss_cls = cls_loss(regime_logits, r_target)
            loss     = LAMBDA_REG * loss_reg + LAMBDA_CLS * loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss    = 0.0

        with torch.no_grad():
            for x, z_target, r_target in val_loader:
                z_pred, regime_logits = model(x)
                loss_reg = reg_loss(z_pred, z_target)
                loss_cls = cls_loss(regime_logits, r_target)
                loss     = LAMBDA_REG * loss_reg + LAMBDA_CLS * loss_cls
                val_loss += loss.item()

                preds       = regime_logits.argmax(dim=1)
                val_correct += (preds == r_target).sum().item()
                val_total   += len(r_target)

        val_acc = val_correct / max(val_total, 1)
        avg_train = train_loss / max(n_batches, 1)
        avg_val   = val_loss / max(len(val_loader), 1)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} "
                  f"Train={avg_train:.4f} "
                  f"Val={avg_val:.4f} "
                  f"ValAcc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/transformer_best.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    return model, best_val_acc


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_on_test(z_array, regime_array, dates):
    """Evaluate on the unseen 2020-2023 test period."""
    date_series = pd.to_datetime(dates)
    test_mask   = date_series >= "2020-01-01"
    test_start  = max(np.where(test_mask)[0][0] - SEQ_LEN, 0)

    z_test      = z_array[test_start:]
    regime_test = regime_array[test_start:]

    test_dataset = MarketSequenceDataset(z_test, regime_test)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MarketTransformer(
        latent_dim=LATENT_DIM, n_regimes=N_REGIMES,
        seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    )
    model.load_state_dict(torch.load("models/transformer_best.pt",
                                      weights_only=True))
    model.eval()

    correct = 0
    total   = 0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, z_target, r_target in test_loader:
            _, regime_logits = model(x)
            preds = regime_logits.argmax(dim=1)
            correct += (preds == r_target).sum().item()
            total   += len(r_target)
            all_preds.extend(preds.numpy())
            all_targets.extend(r_target.numpy())

    test_acc = correct / max(total, 1)
    print(f"\nTest accuracy (2020-2026): {test_acc:.3f}")

    # Per-regime accuracy
    print("\nPer-regime accuracy:")
    for r in range(N_REGIMES):
        mask = np.array(all_targets) == r
        if mask.sum() > 0:
            acc = (np.array(all_preds)[mask] == r).mean()
            print(f"  Regime {r}: {acc:.3f} ({mask.sum()} samples)")

    return test_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def run_transformer_pipeline():
    print("\n── Phase 5: Temporal Transformer ───────────────────\n")

    z_array, regime_array, dates, date_series = load_latent_data()

    if os.path.exists("models/transformer_best.pt"):
        print("Found saved transformer. Skipping training.")
        print("Evaluating on test set...")
        evaluate_on_test(z_array, regime_array, dates)
    else:
        model, best_val_acc = train_transformer(z_array, regime_array, dates)
        print("\nEvaluating on test set...")
        evaluate_on_test(z_array, regime_array, dates)

    print("\n── Phase 5 Complete ─────────────────────────────────\n")


if __name__ == "__main__":
    run_transformer_pipeline()