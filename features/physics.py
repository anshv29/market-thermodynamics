"""
Physics Feature Engine
======================
Computes all physics-inspired market features daily:

  - Market Temperature  T(t)  : cross-sectional std of returns
  - Market Entropy      S(t)  : Shannon entropy of return distribution
  - Potential Energy    PE(t) : volatility compression signal
  - Acceleration        a(t)  : second derivative of price
  - Order Parameter     Ψ(t)  : eigenvalue concentration (phase transition indicator)
  - Interaction Matrix  Φ(t)  : vol-weighted correlation (stored at sector level)
"""

import numpy as np
import pandas as pd
from scipy import linalg


# MARKET TEMPERATURE 

def compute_temperature(returns: pd.DataFrame) -> pd.Series:
    """
    T(t) = cross-sectional standard deviation of all stock returns on day t.

    Input:  returns — DataFrame (dates x tickers)
    Output: Series  (dates) — one temperature value per day
    """
    return returns.std(axis=1)


def compute_sector_temperature(returns: pd.DataFrame,
                                sectors: dict) -> pd.DataFrame:
    """
    Compute temperature separately for each of the 11 GICS sectors.

    Input:  returns — DataFrame (dates x tickers)
            sectors — dict {ticker: sector_name}
    Output: DataFrame (dates x sectors)
    """
    sector_temps = {}
    for sector in set(sectors.values()):
        tickers_in_sector = [t for t, s in sectors.items()
                             if s == sector and t in returns.columns]
        if len(tickers_in_sector) < 3:
            continue
        sector_temps[sector] = returns[tickers_in_sector].std(axis=1)
    return pd.DataFrame(sector_temps)


# ── MARKET ENTROPY ────────────────────────────────────────────────────────────

def compute_entropy(returns: pd.DataFrame) -> pd.Series:
    """
    S(t) = Shannon entropy of the normalized absolute return distribution.

    Steps:
      1. Take absolute returns |r_i(t)|
      2. Normalize so they sum to 1 → probability weights p_i(t)
      3. S(t) = -sum(p_i * log(p_i))

    High entropy = dispersed, confused market
    Low entropy  = concentrated, directional market
    """
    def daily_entropy(row):
        abs_returns = np.abs(row.dropna().values)
        total = abs_returns.sum()
        if total == 0:
            return np.nan
        p = abs_returns / total
        p = p[p > 0]  # avoid log(0)
        return -np.sum(p * np.log(p))

    return returns.apply(daily_entropy, axis=1)


# ── POTENTIAL ENERGY ──────────────────────────────────────────────────────────

def compute_potential_energy(prices: pd.DataFrame,
                              short_window: int = 10,
                              long_window: int  = 60) -> pd.DataFrame:
    """
    PE_i(t) = (σ_long - σ_short) / σ_long

    Measures volatility compression for each stock.
    High PE = short-term vol collapsed below long-term baseline = coiled spring.

    Input:  prices       — DataFrame (dates x tickers), adjusted close
            short_window — rolling window for short vol (default 10 days)
            long_window  — rolling window for long vol  (default 60 days)
    Output: DataFrame (dates x tickers) — PE for each stock each day
    """
    returns = prices.pct_change()
    vol_short = returns.rolling(short_window).std()
    vol_long  = returns.rolling(long_window).std()

    pe = (vol_long - vol_short) / vol_long.replace(0, np.nan)
    return pe


def compute_market_pe(pe: pd.DataFrame) -> pd.Series:
    """
    Aggregate stock-level PE to market level by taking cross-sectional mean.
    """
    return pe.mean(axis=1)


# ── PRICE ACCELERATION ────────────────────────────────────────────────────────

def compute_acceleration(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Velocity     v_i(t) = r_i(t)           (daily return = first derivative)
    Acceleration a_i(t) = r_i(t) - r_i(t-1) (change in return = second derivative)
    Jerk         j_i(t) = a_i(t) - a_i(t-1) (third derivative)

    Input:  returns — DataFrame (dates x tickers)
    Output: dict of DataFrames {velocity, acceleration, jerk}
    """
    velocity     = returns.copy()
    acceleration = returns.diff()
    jerk         = acceleration.diff()

    return {
        "velocity":     velocity,
        "acceleration": acceleration,
        "jerk":         jerk,
    }


def compute_mean_acceleration(acceleration: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional mean acceleration each day.
    When this turns negative while prices are still rising = exhaustion signal.
    """
    return acceleration.mean(axis=1)


# ── ORDER PARAMETER (PHASE TRANSITION INDICATOR) ──────────────────────────────

def compute_order_parameter(returns: pd.DataFrame,
                             window: int = 60) -> pd.DataFrame:
    """
    Ψ(t) = λ_1(t) / Σ λ_k(t)

    The fraction of total variance explained by the dominant eigenvalue
    of the rolling correlation matrix.

    From Random Matrix Theory:
      - In a random (noise) market, eigenvalues follow Marchenko-Pastur distribution
      - The largest eigenvalue FAR exceeds the theoretical max in real markets
      - When Ψ → 1, all stocks move together → phase transition / crash precursor

    Input:  returns — DataFrame (dates x tickers)
            window  — rolling window for correlation matrix (default 60 days)
    Output: DataFrame with columns:
              order_parameter, lambda_1, lambda_2, lambda_ratio
    """
    dates = returns.index[window:]
    results = []

    for i, date in enumerate(dates):
        window_data = returns.iloc[i:i+window].dropna(axis=1, how="any")

        if window_data.shape[1] < 10:
            results.append({
                "date":            date,
                "order_parameter": np.nan,
                "lambda_1":        np.nan,
                "lambda_2":        np.nan,
                "lambda_ratio":    np.nan,
            })
            continue

        # Compute correlation matrix
        corr = window_data.corr().values
        corr = np.nan_to_num(corr, nan=0.0)

        # Eigendecomposition (symmetric matrix → use eigh, faster than eig)
        try:
            eigenvalues = linalg.eigh(corr, eigvals_only=True)
            eigenvalues = np.sort(eigenvalues)[::-1]  # descending

            total_variance = eigenvalues.sum()
            if total_variance <= 0:
                raise ValueError("Zero total variance")

            psi      = eigenvalues[0] / total_variance
            lambda_1 = float(eigenvalues[0])
            lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else np.nan
            ratio    = lambda_2 / lambda_1 if lambda_1 > 0 else np.nan

        except Exception:
            psi, lambda_1, lambda_2, ratio = np.nan, np.nan, np.nan, np.nan

        results.append({
            "date":            date,
            "order_parameter": float(psi),
            "lambda_1":        lambda_1,
            "lambda_2":        lambda_2,
            "lambda_ratio":    ratio,
        })

    return pd.DataFrame(results).set_index("date")


# ── ROLLING VOLATILITY ────────────────────────────────────────────────────────

def compute_rolling_vols(returns: pd.DataFrame,
                          short_window: int = 10,
                          long_window:  int = 60) -> dict:
    """
    Compute rolling volatility for each stock at two window lengths.
    Used as inputs to potential energy and the GNN node features.
    """
    return {
        "vol_10d": returns.rolling(short_window).std(),
        "vol_60d": returns.rolling(long_window).std(),
    }


# ── MASTER COMPUTE FUNCTION ───────────────────────────────────────────────────

def compute_all_features(prices: pd.DataFrame,
                          sectors: dict) -> dict:
    """
    Run the full physics feature pipeline on a price matrix.

    Input:
        prices  — DataFrame (dates x tickers), adjusted close prices
        sectors — dict {ticker: sector_name}

    Output:
        dict with keys:
            market_level  — DataFrame (dates x market features)
            sector_level  — DataFrame (dates x sectors) for temperature
            stock_level   — dict of DataFrames per stock feature
    """
    print("Computing returns...")
    returns = prices.pct_change()

    print("Computing market temperature...")
    temperature = compute_temperature(returns)

    print("Computing sector temperature...")
    sector_temp = compute_sector_temperature(returns, sectors)

    print("Computing market entropy...")
    entropy = compute_entropy(returns)

    print("Computing potential energy...")
    pe_matrix = compute_potential_energy(prices)
    pe_market  = compute_market_pe(pe_matrix)

    print("Computing price acceleration...")
    accel_dict      = compute_acceleration(returns)
    mean_accel      = compute_mean_acceleration(accel_dict["acceleration"])

    print("Computing rolling volatilities...")
    vols = compute_rolling_vols(returns)

    print("Computing order parameter (this takes a minute)...")
    order_df = compute_order_parameter(returns, window=60)

    # ── Assemble market-level DataFrame ──
    market_df = pd.DataFrame({
        "temperature":       temperature,
        "entropy":           entropy,
        "pe_market":         pe_market,
        "mean_acceleration": mean_accel,
    })

    # Merge order parameter results
    market_df = market_df.join(order_df, how="left")

    # Add rate of change of order parameter
    market_df["d_order_parameter"] = market_df["order_parameter"].diff()

    return {
        "market_level": market_df,
        "sector_level": sector_temp,
        "stock_level": {
            "potential_energy": pe_matrix,
            "acceleration":     accel_dict["acceleration"],
            "jerk":             accel_dict["jerk"],
            "velocity":         accel_dict["velocity"],
            "vol_10d":          vols["vol_10d"],
            "vol_60d":          vols["vol_60d"],
        }
    }