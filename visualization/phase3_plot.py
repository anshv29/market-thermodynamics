"""
Phase 3 Visualization
=====================
Shows what the GNN learned by visualizing stock embeddings.

1. UMAP: reduces 32-dim embeddings to 2D — stocks that behave
   similarly in the network cluster together
2. Sector clustering check: did the GNN learn sectors without
   being told sector labels explicitly?
3. Embedding evolution: how did a stock's network position
   change through the 2008 and 2020 crises?
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import text
from db.connection import get_engine, get_session
from db.schema import Universe

engine = get_engine()

# ── Load embeddings for a single recent date ──────────────────────────────────

print("Loading embeddings...")
with engine.connect() as conn:
    df = pd.read_sql(text("""
        SELECT ticker, embedding
        FROM gnn_embeddings_daily
        WHERE date = (SELECT MAX(date) FROM gnn_embeddings_daily)
    """), conn)

print(f"Loaded {len(df)} stock embeddings.")

# Parse embeddings
embeddings = np.array([e if isinstance(e, list) else json.loads(e) for e in df["embedding"]])

# Load sector info
session = get_session()
universe = session.query(Universe).filter_by(is_active=1).all()
session.close()
sector_map = {u.ticker: u.sector for u in universe}
name_map   = {u.ticker: u.name   for u in universe}

df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
df["name"]   = df["ticker"].map(name_map).fillna(df["ticker"])

# ── UMAP dimensionality reduction ────────────────────────────────────────────

print("Running UMAP...")
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)
except ImportError:
    print("UMAP not installed, using PCA instead...")
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    embedding_2d = reducer.fit_transform(embeddings)

df["x"] = embedding_2d[:, 0]
df["y"] = embedding_2d[:, 1]

# ── Plot 1: Sector clustering ─────────────────────────────────────────────────

sector_colors = {
    "Information Technology":  "#4C9BE8",
    "Health Care":             "#E84C9B",
    "Financials":              "#4CE8A0",
    "Consumer Discretionary":  "#E8A04C",
    "Industrials":             "#9B4CE8",
    "Communication Services":  "#E8E84C",
    "Consumer Staples":        "#4CE8E8",
    "Energy":                  "#E84C4C",
    "Real Estate":             "#A0E84C",
    "Materials":               "#E84CA0",
    "Utilities":               "#4C4CE8",
    "Unknown":                 "#888888",
}

fig1 = go.Figure()

for sector in sorted(df["sector"].unique()):
    mask = df["sector"] == sector
    subset = df[mask]
    fig1.add_trace(go.Scatter(
        x=subset["x"],
        y=subset["y"],
        mode="markers",
        name=sector,
        marker=dict(
            color=sector_colors.get(sector, "#888888"),
            size=8,
            opacity=0.8,
            line=dict(width=0.5, color="white"),
        ),
        text=subset.apply(lambda r: f"{r['ticker']}<br>{r['name']}<br>{r['sector']}", axis=1),
        hovertemplate="%{text}<extra></extra>",
    ))

fig1.update_layout(
    title=dict(
        text="GNN Stock Embeddings — Sector Clustering<br>"
             "<sup>Each dot = one stock. Color = sector. "
             "Clusters = stocks with similar network roles. "
             "GNN was NOT told sector labels.</sup>",
        font=dict(size=16),
    ),
    height=700,
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

fig1.write_html("visualization/phase3_sector_clusters.html")
print("Saved: visualization/phase3_sector_clusters.html")

# ── Plot 2: Embedding evolution through crises ────────────────────────────────

print("\nLoading embedding timeline for selected stocks...")

# Pick a few representative stocks across sectors
focus_tickers = ["JPM", "GS", "AAPL", "MSFT", "XOM", "JNJ", "WMT", "BA"]
focus_tickers = [t for t in focus_tickers if t in df["ticker"].values]

with engine.connect() as conn:
    timeline_df = pd.read_sql(text("""
        SELECT ticker, date, embedding
        FROM gnn_embeddings_daily
        WHERE ticker = ANY(:tickers)
        ORDER BY date
    """), conn, params={"tickers": focus_tickers})

timeline_df["date"] = pd.to_datetime(timeline_df["date"])

# Compute embedding norm (magnitude) as a proxy for "network stress"
timeline_df["embedding_norm"] = timeline_df["embedding"].apply(
    lambda e: float(np.linalg.norm(e if isinstance(e, list) else json.loads(e)))
)

# Plot embedding norm over time per stock
fig2 = go.Figure()

colors_line = px.colors.qualitative.Set2

for i, ticker in enumerate(focus_tickers):
    subset = timeline_df[timeline_df["ticker"] == ticker]
    fig2.add_trace(go.Scatter(
        x=subset["date"],
        y=subset["embedding_norm"],
        name=ticker,
        line=dict(color=colors_line[i % len(colors_line)], width=1.5),
        hovertemplate=f"{ticker}<br>%{{x|%Y-%m-%d}}<br>Norm=%{{y:.3f}}<extra></extra>",
    ))

# Mark crises
crises = [
    ("2008-09-01", "2009-03-31", "GFC 2008"),
    ("2020-02-15", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "Rate Hikes"),
]
for start, end, label in crises:
    fig2.add_vrect(
        x0=start, x1=end,
        fillcolor="rgba(255,0,0,0.1)",
        layer="below", line_width=0,
    )
    fig2.add_annotation(
        x=pd.Timestamp(start),
        y=timeline_df["embedding_norm"].max() * 0.95,
        text=label,
        showarrow=False,
        font=dict(size=9, color="red"),
    )

fig2.update_layout(
    title=dict(
        text="GNN Embedding Magnitude Over Time<br>"
             "<sup>How each stock's network role intensity changed "
             "through market crises</sup>",
        font=dict(size=16),
    ),
    height=500,
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
    xaxis=dict(showgrid=True, gridcolor="#2A2A3A"),
    yaxis=dict(showgrid=True, gridcolor="#2A2A3A", title="Embedding Magnitude"),
    hovermode="x unified",
)

fig2.write_html("visualization/phase3_embedding_timeline.html")
print("Saved: visualization/phase3_embedding_timeline.html")

fig1.show()
fig2.show()
print("\nDone.")