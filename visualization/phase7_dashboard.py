"""
Phase 7 — Master Dashboard
==========================
Combines all visualizations into one interactive dashboard.
Shows the full research story from physics features to regimes.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import text
from db.connection import get_engine
from umap import UMAP


engine = get_engine()

REGIME_COLORS = {
    0: "#E05A2B",
    1: "#E8C84C",
    2: "#4C9BE8",
    3: "#E84C9B",
    4: "#4CE8A0",
    5: "#9B59B6",
}


print("Loading all data for dashboard...")


# ── Load physics features ─────────────────────────────────────────────────────

with engine.connect() as conn:
    physics = pd.read_sql(
        text("""
            SELECT date, temperature, entropy, order_parameter,
                   pe_market, d_order_parameter
            FROM market_physics_daily
            WHERE temperature IS NOT NULL
            ORDER BY date
        """),
        conn,
    )

physics["date"] = pd.to_datetime(physics["date"])


# ── Load regimes ──────────────────────────────────────────────────────────────

with engine.connect() as conn:
    regimes = pd.read_sql(
        text("""
            SELECT r.date, r.regime_label,
                   m.temperature, m.order_parameter
            FROM regime_daily r
            JOIN market_physics_daily m ON r.date = m.date
            WHERE m.temperature IS NOT NULL
            ORDER BY r.date
        """),
        conn,
    )

regimes["date"] = pd.to_datetime(regimes["date"])


# ── Load latent states for UMAP ───────────────────────────────────────────────

with engine.connect() as conn:
    latent = pd.read_sql(
        text("""
            SELECT l.date, l.z_vector, r.regime_label
            FROM latent_state_daily l
            JOIN regime_daily r ON l.date = r.date
            ORDER BY l.date
        """),
        conn,
    )

latent["date"] = pd.to_datetime(latent["date"])

latent["z"] = latent["z_vector"].apply(
    lambda x: x if isinstance(x, list) else json.loads(x)
)

Z = np.array(latent["z"].tolist())

print("Running UMAP...")

reducer = UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=30,
    min_dist=0.1,
)

Z_2d = reducer.fit_transform(Z)

latent["ux"] = Z_2d[:, 0]
latent["uy"] = Z_2d[:, 1]


# ── Build master dashboard ────────────────────────────────────────────────────

fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=[
        "Market Temperature T(t) — 2004 to Present",
        "Market State Trajectory — UMAP of Latent Space z(t)",
        "Order Parameter Ψ(t) — Phase Transition Indicator",
        "Regime Timeline — HMM Discovery",
        "Market Entropy S(t)",
        "Potential Energy PE(t) — Volatility Compression",
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)


crises = [
    ("2008-09-01", "2009-03-31"),
    ("2020-02-15", "2020-04-30"),
    ("2022-01-01", "2022-10-31"),
]


# ── Panel 1: Temperature ──────────────────────────────────────────────────────

fig.add_trace(
    go.Scatter(
        x=physics["date"],
        y=physics["temperature"],
        line=dict(color="#E05A2B", width=1),
        name="Temperature",
        showlegend=False,
    ),
    row=1,
    col=1,
)


# ── Panel 2: UMAP latent space ────────────────────────────────────────────────

for regime in sorted(latent["regime_label"].dropna().unique()):
    regime_int = int(regime)
    subset = latent[latent["regime_label"] == regime]

    fig.add_trace(
        go.Scatter(
            x=subset["ux"],
            y=subset["uy"],
            mode="markers",
            name=f"Regime {regime_int}",
            marker=dict(
                color=REGIME_COLORS.get(regime_int, "#888888"),
                size=3,
                opacity=0.6,
            ),
            text=subset["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate=(
                f"Regime {regime_int}<br>"
                "%{text}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )


# ── Panel 3: Order parameter ──────────────────────────────────────────────────

fig.add_trace(
    go.Scatter(
        x=physics["date"],
        y=physics["order_parameter"],
        line=dict(color="#9B59B6", width=1),
        name="Order Parameter",
        showlegend=False,
    ),
    row=2,
    col=1,
)


# ── Panel 4: Regime timeline ──────────────────────────────────────────────────

for regime in range(6):
    subset = regimes[regimes["regime_label"] == regime]

    if subset.empty:
        continue

    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=[regime] * len(subset),
            mode="markers",
            name=f"R{regime}",
            marker=dict(
                color=REGIME_COLORS[regime],
                size=3,
                symbol="square",
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )


# ── Panel 5: Entropy ──────────────────────────────────────────────────────────

fig.add_trace(
    go.Scatter(
        x=physics["date"],
        y=physics["entropy"],
        line=dict(color="#4C9BE8", width=1),
        name="Entropy",
        showlegend=False,
    ),
    row=3,
    col=1,
)


# ── Panel 6: Potential energy ─────────────────────────────────────────────────

fig.add_trace(
    go.Scatter(
        x=physics["date"],
        y=physics["pe_market"],
        line=dict(color="#4CE8A0", width=1),
        name="Potential Energy",
        showlegend=False,
    ),
    row=3,
    col=2,
)


# ── Crisis shading ────────────────────────────────────────────────────────────

for start, end in crises:
    for row, col in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)]:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(255,0,0,0.08)",
            layer="below",
            line_width=0,
            row=row,
            col=col,
        )


# ── Layout ────────────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(
        text=(
            "Statistical Mechanics of the Market Cross-Section — Master Dashboard<br>"
            "<sup>Physics-inspired features + GNN + VAE/HMM + Transformer + RL Agent</sup>"
        ),
        font=dict(size=18),
    ),
    height=1000,
    width=1600,
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
    showlegend=True,
    legend=dict(
        orientation="h",
        y=-0.05,
        font=dict(size=10),
    ),
    hovermode="closest",
)

fig.update_xaxes(showgrid=True, gridcolor="#2A2A3A")
fig.update_yaxes(showgrid=True, gridcolor="#2A2A3A")

# Hide axis labels only on UMAP panel
fig.update_xaxes(showticklabels=False, row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=2)


output = "visualization/phase7_dashboard.html"

fig.write_html(output)
print(f"Saved: {output}")

fig.show()

print("Done.")