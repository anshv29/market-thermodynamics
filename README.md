# Market Thermodynamics

A quantitative trading system that models the S&P 500 as a thermodynamic physical system — measuring market temperature, entropy, and phase transitions across 497 stocks simultaneously — to detect hidden market regimes and allocate capital dynamically.

Walk-forward backtest on 2019–2026 (completely unseen during training):

| Metric | Strategy | S&P 500 |
|---|---|---|
| Sharpe Ratio | 1.553 | 0.896 |
| Max Drawdown | 12.83% | 33.72% |
| Annual Volatility | 9.97% | 19.50% |
| 2022 Return | -0.81% | -18.65% |
| 2020 Alpha | +0.64% | — |

Transaction costs modelled at 10bps per unit of turnover. Running live on Alpaca paper trading since July 2026.

---

## Visualizations

All interactive charts viewable here — no download needed:

[View Interactive Visualizations on Google Drive](https://drive.google.com/drive/folders/1OkB3AA21hkWGmjIFz7VA7g06m8d1gSPw?usp=sharing)

Includes 3D market energy surface, GNN sector clustering, regime timeline, latent space UMAP, transformer predictions, and backtest equity curve vs S&P 500.

---

## How It Works

**Physics Feature Engine**

Four thermodynamic observables computed daily across all 497 constituents. Cross-sectional return dispersion as market temperature T(t). Shannon entropy of the return distribution S(t). Volatility compression as potential energy PE(t). The dominant eigenvalue fraction of the rolling correlation matrix as an order parameter Ψ(t) derived from Random Matrix Theory. When Ψ peaked at 0.742 during the March 2020 COVID crash, 74% of all market movement was explained by a single factor.

**Graph Attention Network**

480 stocks modelled as nodes in a daily correlation graph, edges weighted by vol-weighted correlation Φ_ij = ρ_ij · σ_i · σ_j. Two-layer GAT learns how volatility propagates through the network. Validated by recovering GICS sector structure without sector labels. 2,441,280 embeddings generated across full history.

**Variational Autoencoder**

Compresses 50-dimensional daily market snapshots into a 16-dimensional latent fingerprint z(t). 5,117 latent states generated across full history.

**Hidden Markov Model**

Gaussian HMM discovers 6 recurring market regimes from the latent sequence without supervision. Correctly clusters 2008 GFC and COVID crash periods without being told what those events were. Crash regime 94.4% autocorrelated. Bull market regime 92.7% autocorrelated.

**Temporal Transformer**

4-layer transformer with patch-based tokenization predicts tomorrow's regime with 72.5% accuracy on completely held-out 2020–2026 data. Per-regime: bull 84.1%, recovery 81.5%, crash 55.8%.

**Momentum Overlay + Allocation**

Cross-sectional momentum detects recoveries before the regime signal flips. Regime-conditioned allocation manages capital from 0% to 100% equity based on current thermodynamic state and forward regime probabilities.

---

## Results in Detail

The strategy matches the S&P 500's risk-adjusted returns using half the volatility and less than half the maximum drawdown. The edge concentrates during stress periods — lost 0.81% in 2022 while the benchmark lost 18.65%, generating 17.83% alpha during the rate hike crisis. During COVID the momentum overlay detected the recovery early and matched the benchmark through both the crash and the bounce.

Honest weakness: during pure bull markets like 2021 the strategy underperforms because it is cautious by design. The tradeoff is correct — capital preservation during crashes costs some upside during sustained rallies.

---

## Stack

Python · PyTorch · PyTorch Geometric · PostgreSQL · TimescaleDB · hmmlearn · Alpaca API · yfinance · Plotly · UMAP · NumPy · Pandas · SciPy

---

## Structure

market_thermo > data/ # price and macro ingestion, db/ # schema and connection utilities, features/ # physics feature engine, models/ # GNN, VAE, HMM, transformer, backtest/ # regime strategy, paper trading, daily update, visualization/ # all charts and interactive dashboard

---

## Data

All data sourced from free public APIs — Yahoo Finance for OHLCV and macro signals, Wikipedia for S&P 500 constituents. No Bloomberg, no paid data subscriptions. Stored in PostgreSQL with TimescaleDB for time-series optimization.

---

## Live Deployment

The strategy runs fully automated every day at 4:30 pm EST via Windows Task Scheduler. The daily pipeline updates prices, recomputes physics features, updates GNN embeddings, regenerates latent states and regime labels, runs the transformer, determines allocation, and executes trades via the Alpaca API. Every trade is logged to PostgreSQL and CSV with regime label, allocation percentage, and account value.
