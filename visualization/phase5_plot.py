"""
Phase 5 Visualization
=====================
Shows the transformer's regime predictions vs actual regimes.
1. Prediction accuracy over time
2. Regime transition probability heatmap
3. Confidence on correct vs incorrect predictions
"""

import json
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from sqlalchemy import text

from db.connection import get_engine
from models.transformer import MarketTransformer
from models.transformer_runner import (
    MarketSequenceDataset, SEQ_LEN, PATCH_SIZE,
    LATENT_DIM, N_REGIMES, D_MODEL, N_HEADS, N_LAYERS
)

engine = get_engine()

REGIME_COLORS = {
    0: "#E05A2B", 1: "#E8C84C", 2: "#4C9BE8",
    3: "#E84C9B", 4: "#4CE8A0", 5: "#9B59B6",
}

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
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

# ── Load model and generate predictions ──────────────────────────────────────

print("Loading transformer...")
model = MarketTransformer(
    latent_dim=LATENT_DIM, n_regimes=N_REGIMES,
    seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
)
model.load_state_dict(torch.load(
    "models/transformer_best.pt", weights_only=True
))
model.eval()

# Run on full dataset
dataset = MarketSequenceDataset(z_array, regime_array, SEQ_LEN)
loader  = DataLoader(dataset, batch_size=64, shuffle=False)

all_probs   = []
all_preds   = []
all_targets = []

print("Generating predictions...")
with torch.no_grad():
    for x, z_target, r_target in loader:
        _, logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_targets.extend(r_target.numpy())

all_probs   = np.array(all_probs)
all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)

# Align dates (offset by SEQ_LEN because of sliding window)
pred_dates = pd.to_datetime(dates[SEQ_LEN + 1:SEQ_LEN + 1 + len(all_preds)])
correct     = (all_preds == all_targets).astype(int)
confidence  = all_probs.max(axis=1)

# ── Plot 1: Predictions vs Actual over time ───────────────────────────────────

fig1 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Actual Regime (ground truth)",
        "Predicted Regime (transformer output)",
        "Prediction Confidence (max probability)",
    ],
    vertical_spacing=0.08,
    row_heights=[0.35, 0.35, 0.30],
)

# Actual regimes
for regime in range(N_REGIMES):
    mask = all_targets == regime
    if mask.sum() == 0:
        continue
    fig1.add_trace(go.Scatter(
        x=pred_dates[mask],
        y=np.full(mask.sum(), regime),
        mode="markers",
        name=f"Regime {regime}",
        marker=dict(color=REGIME_COLORS[regime], size=3, symbol="square"),
        showlegend=True,
    ), row=1, col=1)

# Predicted regimes
for regime in range(N_REGIMES):
    mask = all_preds == regime
    if mask.sum() == 0:
        continue
    fig1.add_trace(go.Scatter(
        x=pred_dates[mask],
        y=np.full(mask.sum(), regime),
        mode="markers",
        name=f"Pred {regime}",
        marker=dict(color=REGIME_COLORS[regime], size=3, symbol="square"),
        showlegend=False,
    ), row=2, col=1)

# Confidence — color by correct/incorrect
fig1.add_trace(go.Scatter(
    x=pred_dates,
    y=confidence,
    mode="lines",
    name="Confidence",
    line=dict(color="#4C9BE8", width=1),
    showlegend=False,
), row=3, col=1)

# Mark wrong predictions in red
wrong_mask = correct == 0
if wrong_mask.sum() > 0:
    fig1.add_trace(go.Scatter(
        x=pred_dates[wrong_mask],
        y=confidence[wrong_mask],
        mode="markers",
        name="Wrong prediction",
        marker=dict(color="#E05A2B", size=3, opacity=0.5),
        showlegend=True,
    ), row=3, col=1)

# Crisis shading
crises = [
    ("2008-09-01", "2009-03-31"),
    ("2020-02-15", "2020-04-30"),
    ("2022-01-01", "2022-10-31"),
]
for start, end in crises:
    for row in range(1, 4):
        fig1.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255,0,0,0.08)",
            layer="below", line_width=0,
            row=row, col=1,
        )

fig1.update_layout(
    title=dict(
        text="Transformer Regime Predictions vs Actual<br>"
             "<sup>Top = ground truth. Middle = predicted. "
             "Bottom = confidence (red dots = wrong predictions)</sup>",
        font=dict(size=16),
    ),
    height=750,
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
    hovermode="x unified",
)
fig1.update_xaxes(showgrid=True, gridcolor="#2A2A3A")
fig1.update_yaxes(showgrid=True, gridcolor="#2A2A3A")

fig1.write_html("visualization/phase5_predictions.html")
print("Saved: visualization/phase5_predictions.html")

# ── Plot 2: Regime transition probability heatmap ─────────────────────────────

# Build transition matrix from predictions
transition_matrix = np.zeros((N_REGIMES, N_REGIMES))
for i in range(len(all_targets) - 1):
    from_regime = all_targets[i]
    to_regime   = all_targets[i + 1]
    transition_matrix[from_regime, to_regime] += 1

# Normalize rows
row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = np.divide(
    transition_matrix, row_sums,
    where=row_sums > 0
)

fig2 = go.Figure(go.Heatmap(
    z=transition_matrix,
    x=[f"→ Regime {i}" for i in range(N_REGIMES)],
    y=[f"From Regime {i}" for i in range(N_REGIMES)],
    colorscale="Viridis",
    text=np.round(transition_matrix, 3),
    texttemplate="%{text}",
    hovertemplate="From %{y}<br>%{x}<br>Probability: %{z:.3f}<extra></extra>",
))

fig2.update_layout(
    title=dict(
        text="Regime Transition Probability Matrix<br>"
             "<sup>How likely is each regime to transition to each other regime?</sup>",
        font=dict(size=16),
    ),
    height=500,
    plot_bgcolor="#0F1117",
    paper_bgcolor="#0F1117",
    font=dict(color="#CCCCCC"),
)

fig2.write_html("visualization/phase5_transitions.html")
print("Saved: visualization/phase5_transitions.html")

fig1.show()
fig2.show()
print("\nDone.")