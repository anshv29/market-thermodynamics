"""
Phase 2 Visualization
=====================
Plots market temperature, entropy, and order parameter from 2004 to present.
Crisis periods are marked with red shading.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()

# ── Load data ────────────────────────────────────────────────────────────────

print("Loading physics features...")
with engine.connect() as conn:
    df = pd.read_sql(
        text("""
            SELECT date, temperature, entropy, order_parameter,
                   pe_market, d_order_parameter
            FROM market_physics_daily
            WHERE temperature IS NOT NULL
            ORDER BY date
        """),
        conn
    )

df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")

# ── Crisis periods ────────────────────────────────────────────────────────────

crises = [
    ("2008-09-01", "2009-03-31", "GFC 2008"),
    ("2010-04-01", "2010-07-31", "Flash Crash 2010"),
    ("2011-07-01", "2011-10-31", "EU Debt Crisis"),
    ("2020-02-15", "2020-04-30", "COVID Crash"),
    ("2022-01-01", "2022-10-31", "Rate Hike Crisis"),
]

# ── Build figure ──────────────────────────────────────────────────────────────

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Market Temperature T(t) — Cross-sectional volatility of returns",
        "Market Entropy S(t) — Disorder in the return distribution",
        "Order Parameter Ψ(t) — Phase transition indicator (eigenvalue concentration)",
        "Market Potential Energy PE(t) — Volatility compression signal",
    ],
    vertical_spacing=0.06,
    row_heights=[0.25, 0.25, 0.25, 0.25],
)

colors = {
    "temperature":     "#E05A2B",
    "entropy":         "#5B8DD9",
    "order_parameter": "#9B59B6",
    "pe_market":       "#27AE60",
}

# Add traces
fig.add_trace(go.Scatter(
    x=df["date"], y=df["temperature"],
    name="Temperature", line=dict(color=colors["temperature"], width=1),
    hovertemplate="%{x|%Y-%m-%d}<br>T=%{y:.4f}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df["date"], y=df["entropy"],
    name="Entropy", line=dict(color=colors["entropy"], width=1),
    hovertemplate="%{x|%Y-%m-%d}<br>S=%{y:.4f}<extra></extra>"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df["date"], y=df["order_parameter"],
    name="Order Parameter Ψ", line=dict(color=colors["order_parameter"], width=1),
    hovertemplate="%{x|%Y-%m-%d}<br>Ψ=%{y:.4f}<extra></extra>"
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df["date"], y=df["pe_market"],
    name="Potential Energy", line=dict(color=colors["pe_market"], width=1),
    hovertemplate="%{x|%Y-%m-%d}<br>PE=%{y:.4f}<extra></extra>"
), row=4, col=1)

# Add crisis shading to all subplots
for start, end, label in crises:
    for row in range(1, 5):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 0, 0, 0.10)",
            layer="below",
            line_width=0,
            row=row, col=1,
        )
    # Add label on top plot only
    fig.add_annotation(
        x=pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2,
        y=df["temperature"].max() * 0.95,
        text=label,
        showarrow=False,
        font=dict(size=9, color="red"),
        row=1, col=1,
    )

# ── Layout ────────────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(
        text="Statistical Mechanics of the Market Cross-Section<br>"
             "<sup>Physics-inspired features 2004–present | Red shading = crisis periods</sup>",
        font=dict(size=18),
    ),
    height=900,
    width=1400,
    showlegend=True,
    legend=dict(orientation="h", y=-0.05),
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
    hovermode="x unified",
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="#2A2A3A",
    zeroline=False,
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="#2A2A3A",
    zeroline=False,
)

# ── Save and show ─────────────────────────────────────────────────────────────

output_path = "visualization/phase2_physics.html"
fig.write_html(output_path)
print(f"\nSaved to {output_path}")
print("Opening in browser...")
fig.show()