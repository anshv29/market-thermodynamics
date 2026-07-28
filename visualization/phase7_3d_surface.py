import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text
from market_thermo.db.connection import get_engine

engine = get_engine()

print("Loading sector temperatures...")
with engine.connect() as conn:
    df = pd.read_sql(text("""
        SELECT date, sector, temperature
        FROM sector_physics_daily
        WHERE temperature IS NOT NULL
        ORDER BY date, sector
    """), conn)

df["date"] = pd.to_datetime(df["date"])
pivot = df.pivot(index="date", columns="sector", values="temperature")
pivot = pivot.ffill().fillna(0)
pivot_weekly = pivot.resample("W").mean()

sector_order = [
    "Utilities", "Consumer Staples", "Health Care", "Real Estate",
    "Financials", "Industrials", "Materials", "Energy",
    "Communication Services", "Consumer Discretionary", "Information Technology",
]
available_sectors = [s for s in sector_order if s in pivot_weekly.columns]
pivot_weekly = pivot_weekly[available_sectors]

dates   = pivot_weekly.index
sectors = pivot_weekly.columns.tolist()
Z       = pivot_weekly.values.T

print(f"Surface: {len(dates)} weeks x {len(sectors)} sectors")

fig = go.Figure()

fig.add_trace(go.Surface(
    x=[str(d.date()) for d in dates],
    y=sectors,
    z=Z,
    colorscale=[
        [0.0,  "#0a0a2e"],
        [0.2,  "#1a1a6e"],
        [0.4,  "#2255cc"],
        [0.6,  "#44aaff"],
        [0.75, "#ffdd00"],
        [0.9,  "#ff6600"],
        [1.0,  "#ff0000"],
    ],
    opacity=0.9,
    hovertemplate=(
        "Date: %{x}<br>"
        "Sector: %{y}<br>"
        "Temperature: %{z:.4f}<extra></extra>"
    ),
))

fig.update_layout(
    title=dict(
        text="Market Energy Surface — Statistical Mechanics of the Cross-Section<br>"
             "<sup>Z = sector temperature. Red peaks = crisis. Blue = compression.</sup>",
        font=dict(size=16, color="white"),
    ),
    scene=dict(
        xaxis=dict(
            title=dict(text="Time", font=dict(color="white")),
            tickfont=dict(color="white", size=9),
            gridcolor="#2A2A3A",
            backgroundcolor="#0F1117",
            nticks=12,
        ),
        yaxis=dict(
            title=dict(text="Sector", font=dict(color="white")),
            tickfont=dict(color="white", size=8),
            gridcolor="#2A2A3A",
            backgroundcolor="#0F1117",
        ),
        zaxis=dict(
            title=dict(text="Temperature T(t)", font=dict(color="white")),
            tickfont=dict(color="white", size=9),
            gridcolor="#2A2A3A",
            backgroundcolor="#0F1117",
        ),
        camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
        bgcolor="#0F1117",
    ),
    paper_bgcolor="#0F1117",
    font=dict(color="white"),
    height=750,
    width=1400,
    margin=dict(l=0, r=0, t=80, b=0),
)

from pathlib import Path

output = Path(__file__).parent / "phase7_3d_surface.html"
fig.write_html(str(output))

print(f"Saved 3D surface to: {output}")

fig.show()