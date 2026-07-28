"""
VAE + HMM Runner
=================
1. Builds daily feature vectors (physics + GNN embeddings)
2. Trains the VAE to compress them into 16-dim latent states
3. Fits an HMM on the latent sequence to discover market regimes
4. Stores latent states and regime labels in the database
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sqlalchemy import text
import random

# Fix all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from db.connection import get_engine
from models.vae import MarketVAE


# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_START  = "2006-01-01"
TRAIN_END    = "2018-12-31"
LATENT_DIM   = 16
EPOCHS       = 50
LR           = 1e-3
BATCH_SIZE   = 64
BETA         = 1.0
N_REGIMES    = 6    # number of HMM hidden states


# ── Feature Construction ──────────────────────────────────────────────────────

def build_daily_feature_vectors() -> pd.DataFrame:
    """
    For each trading day, build a single feature vector combining:
    - Market-level physics features (7 features)
    - Sector-level temperatures (11 features)
    - Aggregated GNN embeddings (mean across all stocks = 32 features)

    Total input dim: 50 features per day
    """
    engine = get_engine()
    print("Loading market physics features...")
    with engine.connect() as conn:
        physics = pd.read_sql(text("""
            SELECT date, temperature, entropy, order_parameter,
                   lambda_1, lambda_ratio, d_order_parameter, pe_market
            FROM market_physics_daily
            WHERE temperature IS NOT NULL
            ORDER BY date
        """), conn)

    physics["date"] = pd.to_datetime(physics["date"])
    physics = physics.set_index("date")

    print("Loading sector temperatures...")
    with engine.connect() as conn:
        sector_raw = pd.read_sql(text("""
            SELECT date, sector, temperature
            FROM sector_physics_daily
            WHERE temperature IS NOT NULL
            ORDER BY date, sector
        """), conn)

    sector_raw["date"] = pd.to_datetime(sector_raw["date"])
    sector_pivot = sector_raw.pivot(
        index="date", columns="sector", values="temperature"
    ).fillna(0)

    print("Loading GNN embeddings (mean pooled)...")
    with engine.connect() as conn:
        embeddings_raw = pd.read_sql(text("""
            SELECT date, embedding
            FROM gnn_embeddings_daily
            ORDER BY date, ticker
        """), conn)

    embeddings_raw["date"] = pd.to_datetime(embeddings_raw["date"])

    # Mean pool embeddings across all stocks per day
    print("Mean pooling embeddings by day...")
    def parse_embedding(e):
        if isinstance(e, list):
            return e
        return json.loads(e)

    embeddings_raw["emb_array"] = embeddings_raw["embedding"].apply(parse_embedding)

    emb_by_date = embeddings_raw.groupby("date")["emb_array"].apply(
        lambda x: np.mean(np.array(list(x)), axis=0)
    )

    emb_df = pd.DataFrame(
        emb_by_date.tolist(),
        index=emb_by_date.index,
        columns=[f"gnn_{i}" for i in range(32)]
    )

    print("Combining all features...")
    print("Combining all features...")
    combined = physics.join(sector_pivot, how="inner")
    combined = combined.join(emb_df, how="inner")
    combined = combined.fillna(0)

    print(f"Feature matrix: {combined.shape[0]} days × {combined.shape[1]} features")
    return combined


# ── Training ──────────────────────────────────────────────────────────────────

def train_vae(feature_df: pd.DataFrame):
    """Train the VAE on the feature matrix."""

    # Normalize features
    train_data = feature_df.loc[TRAIN_START:TRAIN_END]
    mean = train_data.mean()
    std  = train_data.std().replace(0, 1)

    normalized = (feature_df - mean) / std
    normalized = normalized.fillna(0)

    input_dim = normalized.shape[1]
    print(f"\nTraining VAE | input_dim={input_dim} | latent_dim={LATENT_DIM}")

    # Build dataset
    X_train = torch.tensor(
        normalized.loc[TRAIN_START:TRAIN_END].values,
        dtype=torch.float32
    )
    X_all = torch.tensor(normalized.values, dtype=torch.float32)

    dataset    = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = MarketVAE(input_dim=input_dim, latent_dim=LATENT_DIM, beta=BETA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss  = 0.0
        epoch_recon = 0.0
        epoch_kl    = 0.0

        for (batch,) in dataloader:
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(batch)
            loss, recon, kl = model.vae_loss(batch, x_recon, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()

        scheduler.step()
        n = len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} "
                  f"Loss={epoch_loss/n:.4f} "
                  f"Recon={epoch_recon/n:.4f} "
                  f"KL={epoch_kl/n:.4f}")

    return model, normalized, mean, std


# ── Latent State Generation ───────────────────────────────────────────────────

def generate_latent_states(model: MarketVAE,
                            normalized: pd.DataFrame) -> pd.DataFrame:
    """Run the full dataset through the VAE encoder to get z(t) for every day."""
    model.eval()
    X_all = torch.tensor(normalized.values, dtype=torch.float32)

    with torch.no_grad():
        _, mu, logvar, z = model(X_all)

    z_df = pd.DataFrame(
        z.numpy(),
        index=normalized.index,
        columns=[f"z_{i}" for i in range(LATENT_DIM)]
    )
    mu_df = pd.DataFrame(
        mu.numpy(),
        index=normalized.index,
        columns=[f"mu_{i}" for i in range(LATENT_DIM)]
    )
    logvar_df = pd.DataFrame(
        logvar.numpy(),
        index=normalized.index,
        columns=[f"logvar_{i}" for i in range(LATENT_DIM)]
    )

    return z_df, mu_df, logvar_df


# ── HMM Regime Detection ──────────────────────────────────────────────────────

def fit_hmm(z_df: pd.DataFrame, n_regimes: int = N_REGIMES):
    """
    Fit a Gaussian HMM on the latent state sequence.
    Discovers recurring market regimes from data.
    """
    from hmmlearn import hmm

    print(f"\nFitting HMM with {n_regimes} hidden states...")
    Z = z_df.values

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
        verbose=False,
    )
    model.fit(Z)

    # Viterbi decoding — most likely regime sequence
    regime_labels = model.predict(Z)

    # Posterior probabilities
    regime_probs = model.predict_proba(Z)

    print(f"HMM converged. Regime distribution:")
    for k in range(n_regimes):
        count = (regime_labels == k).sum()
        pct   = count / len(regime_labels) * 100
        print(f"  Regime {k}: {count} days ({pct:.1f}%)")

    return model, regime_labels, regime_probs


# ── Database Storage ──────────────────────────────────────────────────────────

def store_latent_states(z_df, mu_df, logvar_df,
                         regime_labels, regime_probs):
    """Store VAE latent states and HMM regime labels in the database."""
    engine = get_engine()
    records = []

    for i, date in enumerate(z_df.index):
        records.append({
            "date":    date.date(),
            "z_vector": json.dumps(z_df.iloc[i].tolist()),
            "z_mu":     json.dumps(mu_df.iloc[i].tolist()),
            "z_logvar": json.dumps(logvar_df.iloc[i].tolist()),
        })

    print(f"\nStoring {len(records)} latent states...")
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO latent_state_daily (date, z_vector, z_mu, z_logvar)
                VALUES (CAST(:date AS DATE),
                        CAST(:z_vector AS jsonb),
                        CAST(:z_mu AS jsonb),
                        CAST(:z_logvar AS jsonb))
                ON CONFLICT (date) DO UPDATE SET
                    z_vector = EXCLUDED.z_vector,
                    z_mu     = EXCLUDED.z_mu,
                    z_logvar = EXCLUDED.z_logvar
            """),
            records
        )
        conn.commit()
    print("Latent states stored.")

    # Store regime labels
    regime_records = []
    for i, date in enumerate(z_df.index):
        regime_records.append({
            "date":         date.date(),
            "regime_label": int(regime_labels[i]),
            "regime_probs": json.dumps(regime_probs[i].tolist()),
            "regime_name":  f"regime_{regime_labels[i]}",
        })

    print(f"Storing {len(regime_records)} regime labels...")
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO regime_daily
                    (date, regime_label, regime_probs, regime_name)
                VALUES (CAST(:date AS DATE), :regime_label,
                        CAST(:regime_probs AS jsonb), :regime_name)
                ON CONFLICT (date) DO UPDATE SET
                    regime_label = EXCLUDED.regime_label,
                    regime_probs = EXCLUDED.regime_probs,
                    regime_name  = EXCLUDED.regime_name
            """),
            regime_records
        )
        conn.commit()
    print("Regime labels stored.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_vae_pipeline():
    print("\n── Phase 4: VAE + HMM ───────────────────────────────\n")

    # 1. Build feature vectors
    feature_df = build_daily_feature_vectors()

    # 2. Train VAE (or load if exists)
    if os.path.exists("models/vae_weights.pt"):
        print("Loading saved VAE weights...")
        mean = pd.read_csv("models/vae_mean.csv", index_col=0).squeeze()
        std  = pd.read_csv("models/vae_std.csv",  index_col=0).squeeze()
        normalized = (feature_df - mean) / std.replace(0, 1)
        normalized = normalized.fillna(0)
        input_dim  = normalized.shape[1]
        model = MarketVAE(input_dim=input_dim, latent_dim=LATENT_DIM)
        model.load_state_dict(torch.load("models/vae_weights.pt",
                                          weights_only=True))
        print("VAE loaded.")
        # Force regenerate all latent states including new dates
        print("Regenerating latent states for full history including new dates...")
        z_df, mu_df, logvar_df = generate_latent_states(model, normalized)
        hmm_model, regime_labels, regime_probs = fit_hmm(z_df)
        store_latent_states(z_df, mu_df, logvar_df, regime_labels, regime_probs)
        print("\n── Phase 4 Complete ─────────────────────────────────\n")
        return
    else:
        model, normalized, mean, std = train_vae(feature_df)
        torch.save(model.state_dict(), "models/vae_weights.pt")
        mean.to_csv("models/vae_mean.csv")
        std.to_csv("models/vae_std.csv")
        print("VAE saved.")

    # 3. Generate latent states
    print("\nGenerating latent states for full history...")
    z_df, mu_df, logvar_df = generate_latent_states(model, normalized)
    print(f"Generated {len(z_df)} latent state vectors.")

    # 4. Fit HMM
    hmm_model, regime_labels, regime_probs = fit_hmm(z_df)

    # 5. Store everything
    store_latent_states(z_df, mu_df, logvar_df, regime_labels, regime_probs)

    print("\n── Phase 4 Complete ─────────────────────────────────\n")
    return z_df, regime_labels, regime_probs


if __name__ == "__main__":
    run_vae_pipeline()