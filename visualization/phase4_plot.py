"""
Phase 4 Visualization
=====================
Shows what the VAE + HMM discovered:
1. Regime timeline — which regime was the market in each day
2. UMAP of latent states — market trajectory through z-space
3. Regime characteristics — what each regime looks like
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()

REGIME_COLORS = {
    0: "#E05A2B",
    1: "#E8C84C",
    2: "#4C9BE8",
    3: "#E84C9B",
    4: "#4CE8A0",
    5: "#9B59B6",
}

print("Loading regime labels...")
with engine.connect() as conn:
    regime_df = pd.read_sql(text("""
        SELECT r.date, r.regime_label, r.regime_probs,
               m.temperature, m.entropy, m.order_parameter, m.pe_market
        FROM regime_daily r
        JOIN market_physics_daily m ON r.date = m.date
        WHERE m.temperature IS NOT NULL
        ORDER BY r.date
    """), conn)

regime_df["date"] = pd.to_datetime(regime_df["date"])

print("Loading latent states...")
with engine.connect() as conn:
    latent_df = pd.read_sql(text("""
        SELECT date, z_vector FROM latent_state_daily ORDER BY date
    """), conn)

latent_df["date"] = pd.to_datetime(latent_df["date"])
latent_df["z"] = latent_df["z_vector"].apply(
    lambda x: x if isinstance(x, list) else json.loads(x)
)

# ── UMAP of latent states ─────────────────────────────────────────────────────

print("Running UMAP on latent states...")
Z = np.array(latent_df["z"].tolist())

try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    Z_2d = reducer.fit_transform(Z)
except ImportError:
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    Z_2d = reducer.fit_transform(Z)

latent_df["umap_x"] = Z_2d[:, 0]
latent_df["umap_y"] = Z_2d[:, 1]

# Merge with regime labels
latent_df = latent_df.merge(
    regime_df[["date", "regime_label"]],
    on="date", how="left"
)
latent_df["regime_label"] = latent_df["regime_label"].fillna(-1).astype(int)

# ── Plot 1: Regime Timeline ───────────────────────────────────────────────────

fig1 = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Market Regime — discovered by HMM (no labels given)",
        "Market Temperature",
        "Order Parameter Ψ",
        "Market Entropy",
    ],
    vertical_spacing=0.06,
    row_heights=[0.4, 0.2, 0.2, 0.2],
)

# Regime as colored scatter
for regime in sorted(regime_df["regime_label"].unique()):
    mask = regime_df["regime_label"] == regime
    subset = regime_df[mask]
    fig1.add_trace(go.Scatter(
        x=subset["date"],
        y=[regime] * len(subset),
        mode="markers",
        name=f"Regime {regime}",
        marker=dict(
            color=REGIME_COLORS.get(regime, "#888888"),
            size=4,
            symbol="square",
        ),
        hovertemplate=f"Regime {regime}<br>%{{x|%Y-%m-%d}}<extra></extra>",
    ), row=1, col=1)

# Physics features
fig1.add_trace(go.Scatter(
    x=regime_df["date"], y=regime_df["temperature"],
    name="Temperature", line=dict(color="#E05A2B", width=1),
    showlegend=False,
), row=2, col=1)

fig1.add_trace(go.Scatter(
    x=regime_df["date"], y=regime_df["order_parameter"],
    name="Order Parameter", line=dict(color="#9B59B6", width=1),
    showlegend=False,
), row=3, col=1)

fig1.add_trace(go.Scatter(
    x=regime_df["date"], y=regime_df["entropy"],
    name="Entropy", line=dict(color="#4C9BE8", width=1),
    showlegend=False,
), row=4, col=1)

# Crisis shading
crises = [
    ("2008-09-01", "2009-03-31", "GFC"),
    ("2020-02-15", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "Rate Hikes"),
]
for start, end, label in crises:
    for row in range(1, 5):
        fig1.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255,0,0,0.08)",
            layer="below", line_width=0,
            row=row, col=1,
        )

fig1.update_layout(
    title=dict(
        text="Market Regime Timeline — VAE + HMM Discovery<br>"
             "<sup>Regimes discovered purely from data. "
             "Red shading = known crisis periods.</sup>",
        font=dict(size=16),
    ),
    height=800,
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
    hovermode="x unified",
)
fig1.update_xaxes(showgrid=True, gridcolor="#2A2A3A")
fig1.update_yaxes(showgrid=True, gridcolor="#2A2A3A")

fig1.write_html("visualization/phase4_regime_timeline.html")
print("Saved: visualization/phase4_regime_timeline.html")

# ── Plot 2: UMAP of latent states ─────────────────────────────────────────────

fig2 = go.Figure()

for regime in sorted(latent_df["regime_label"].unique()):
    if regime == -1:
        continue
    mask   = latent_df["regime_label"] == regime
    subset = latent_df[mask]
    fig2.add_trace(go.Scatter(
        x=subset["umap_x"],
        y=subset["umap_y"],
        mode="markers",
        name=f"Regime {regime}",
        marker=dict(
            color=REGIME_COLORS.get(regime, "#888888"),
            size=5,
            opacity=0.7,
            line=dict(width=0.3, color="white"),
        ),
        text=subset["date"].dt.strftime("%Y-%m-%d"),
        hovertemplate="Regime " + str(regime) +
                      "<br>%{text}<extra></extra>",
    ))

# Add trajectory line (chronological path)
fig2.add_trace(go.Scatter(
    x=latent_df["umap_x"],
    y=latent_df["umap_y"],
    mode="lines",
    line=dict(color="rgba(255,255,255,0.05)", width=0.5),
    showlegend=False,
    hoverinfo="skip",
))

fig2.update_layout(
    title=dict(
        text="Market State Trajectory — UMAP of Latent Space z(t)<br>"
             "<sup>Each dot = one trading day. "
             "Color = discovered regime. "
             "Position = market thermodynamic state.</sup>",
        font=dict(size=16),
    ),
    height=700,
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

fig2.write_html("visualization/phase4_latent_space.html")
print("Saved: visualization/phase4_latent_space.html")

fig1.show()
fig2.show()
print("\nDone.")