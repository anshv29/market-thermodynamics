"""
GNN Training and Embedding Generation
======================================
1. Loads physics features from the database
2. Builds daily market graphs
3. Trains the GNN to predict next-day volatility
4. Generates and stores 32-dim embeddings for every stock every day
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
from sqlalchemy import text

from db.connection import get_engine, get_session
from db.schema import Universe, GNNEmbeddingsDaily
from models.gnn import MarketGNN, build_graph

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_START    = "2006-01-01"
TRAIN_END      = "2018-12-31"
EMBED_START    = "2006-01-01"
CORR_WINDOW    = 60
CORR_THRESHOLD = 0.3
EMBEDDING_DIM  = 32
EPOCHS         = 30
LR             = 1e-3


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_training_data() -> tuple:
    engine = get_engine()
    print("Loading prices...")

    with engine.connect() as conn:
        prices_df = pd.read_sql(text("""
            SELECT date, ticker, adj_close
            FROM ohlcv_daily
            WHERE date >= :start AND adj_close IS NOT NULL
            ORDER BY date, ticker
        """), conn, params={"start": EMBED_START})

    prices = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    prices.index = pd.to_datetime(prices.index)
    returns = prices.pct_change().fillna(0)

    print("Loading stock physics features...")
    with engine.connect() as conn:
        physics_df = pd.read_sql(text("""
            SELECT date, ticker, potential_energy, acceleration,
                   vol_10d, vol_60d
            FROM stock_physics_daily
            WHERE date >= :start
            ORDER BY date, ticker
        """), conn, params={"start": EMBED_START})

    physics_df["date"] = pd.to_datetime(physics_df["date"])

    session = get_session()
    universe = session.query(Universe).filter_by(is_active=1).all()
    session.close()

    sectors     = {u.ticker: u.sector for u in universe}
    sector_list = sorted(set(sectors.values()))
    sector_map  = {s: i for i, s in enumerate(sector_list)}

    tickers = sorted(set(prices.columns) &
                     set(physics_df["ticker"].unique()) &
                     set(sectors.keys()))

    print(f"Common tickers: {len(tickers)}")
    print(f"Date range: {prices.index.min().date()} → {prices.index.max().date()}")

    return prices[tickers], returns[tickers], physics_df, sectors, sector_map, tickers


def build_node_features(date, tickers, physics_df, sectors, sector_map):
    day_physics = physics_df[physics_df["date"] == date].set_index("ticker")
    n_sectors   = len(sector_map)
    features    = []

    for ticker in tickers:
        if ticker in day_physics.index:
            row   = day_physics.loc[ticker]
            pe    = float(row["potential_energy"]) if pd.notna(row["potential_energy"]) else 0.0
            accel = float(row["acceleration"])     if pd.notna(row["acceleration"])     else 0.0
            v10   = float(row["vol_10d"])          if pd.notna(row["vol_10d"])          else 0.0
            v60   = float(row["vol_60d"])          if pd.notna(row["vol_60d"])          else 0.0
        else:
            pe, accel, v10, v60 = 0.0, 0.0, 0.0, 0.0

        sector_vec = [0.0] * n_sectors
        if ticker in sectors and sectors[ticker] in sector_map:
            sector_vec[sector_map[sectors[ticker]]] = 1.0

        features.append([pe, accel, v10, v60] + sector_vec)

    return torch.tensor(features, dtype=torch.float)


def build_correlation_matrix(returns_window):
    corr = returns_window.corr().fillna(0).values
    return torch.tensor(corr, dtype=torch.float)


# ── Training ──────────────────────────────────────────────────────────────────

def train_gnn(prices, returns, physics_df, sectors, sector_map, tickers):
    n_sectors        = len(sector_map)
    node_feature_dim = 4 + n_sectors

    model     = MarketGNN(node_feature_dim=node_feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_dates = returns.loc[TRAIN_START:TRAIN_END].index.tolist()
    train_dates = [d for d in train_dates
                   if d >= pd.Timestamp(TRAIN_START) + pd.Timedelta(days=CORR_WINDOW)]

    print(f"\nTraining GNN on {len(train_dates)} days...")
    print(f"Node feature dim: {node_feature_dim}")
    print(f"Tickers: {len(tickers)}")

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches  = 0
        idx = np.random.permutation(len(train_dates) - 1)

        for i in tqdm(idx[:200], desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            date     = train_dates[i]
            next_day = train_dates[i + 1] if i + 1 < len(train_dates) else None
            if next_day is None:
                continue

            window_start = date - pd.Timedelta(days=CORR_WINDOW * 2)
            ret_window   = returns.loc[window_start:date][tickers].fillna(0)
            corr_matrix  = build_correlation_matrix(ret_window)
            node_feats   = build_node_features(date, tickers, physics_df, sectors, sector_map)
            graph        = build_graph(node_feats, corr_matrix, CORR_THRESHOLD)

            if next_day in returns.index:
                target = torch.tensor(
                    returns.loc[next_day, tickers].fillna(0).abs().values,
                    dtype=torch.float
                )
            else:
                continue

            optimizer.zero_grad()
            _, predictions = model(graph)
            loss = criterion(predictions, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.6f}")

    return model


# ── Embedding Generation ──────────────────────────────────────────────────────

def generate_and_store_embeddings(model, prices, returns,
                                   physics_df, sectors, sector_map, tickers):
    model.eval()
    engine    = get_engine()
    all_dates = returns.loc[EMBED_START:].index.tolist()
    all_dates = [d for d in all_dates
                 if d >= pd.Timestamp(EMBED_START) + pd.Timedelta(days=CORR_WINDOW)]

    print(f"\nGenerating embeddings for {len(all_dates)} days...")

    with torch.no_grad():
        for date in tqdm(all_dates, desc="Generating embeddings"):
            window_start = date - pd.Timedelta(days=CORR_WINDOW * 2)
            ret_window   = returns.loc[window_start:date][tickers].fillna(0)
            corr_matrix  = build_correlation_matrix(ret_window)
            node_feats   = build_node_features(date, tickers, physics_df, sectors, sector_map)
            graph        = build_graph(node_feats, corr_matrix, CORR_THRESHOLD)

            embeddings, _ = model(graph)
            embeddings_np = embeddings.numpy()

            records = []
            for idx, ticker in enumerate(tickers):
                records.append({
                    "ticker":    ticker,
                    "date":      date.date(),
                    "embedding": json.dumps(embeddings_np[idx].tolist()),
                })

            with engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO gnn_embeddings_daily (ticker, date, embedding)
                        VALUES (:ticker, CAST(:date AS DATE), CAST(:embedding AS jsonb))
                        ON CONFLICT (ticker, date) DO NOTHING
                    """),
                    records
                )
                conn.commit()

    print("Embeddings stored.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_gnn_pipeline():
    print("\n── Phase 3: Graph Neural Network ───────────────────\n")

    prices, returns, physics_df, sectors, sector_map, tickers = load_training_data()

    n_sectors        = len(sector_map)
    node_feature_dim = 4 + n_sectors
    model = MarketGNN(node_feature_dim=node_feature_dim)

    if os.path.exists("models/gnn_weights.pt"):
        print("Loading saved model weights...")
        model.load_state_dict(torch.load("models/gnn_weights.pt", weights_only=True))
        print("Model loaded. Skipping training.")
    else:
        model = train_gnn(prices, returns, physics_df, sectors, sector_map, tickers)
        torch.save(model.state_dict(), "models/gnn_weights.pt")
        print("Model saved.")

    generate_and_store_embeddings(
        model, prices, returns, physics_df, sectors, sector_map, tickers
    )

    print("\n── Phase 3 Complete ─────────────────────────────────\n")


if __name__ == "__main__":
    run_gnn_pipeline()