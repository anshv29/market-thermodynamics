"""
Market Trading Environment
===========================
Custom OpenAI Gym environment for the RL agent.

State:  physics features + latent state + regime + portfolio state
Action: continuous portfolio weights across N stocks
Reward: risk-adjusted return - transaction costs - drawdown penalty
"""

import numpy as np
import pandas as pd
import json
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from sqlalchemy import text
from db.connection import get_engine


class MarketEnv(gym.Env):
    """
    A daily portfolio management environment.

    The agent observes the market state each day and decides
    how to allocate capital across N stocks.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 start_date: str = "2006-01-01",
                 end_date:   str = "2018-12-31",
                 n_stocks:   int = 50,
                 initial_capital: float = 1_000_000,
                 transaction_cost: float = 0.001,
                 drawdown_penalty: float = 0.1,
                 window_size: int = 1):

        super().__init__()

        self.start_date       = start_date
        self.end_date         = end_date
        self.n_stocks         = n_stocks
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.drawdown_penalty = drawdown_penalty
        self.window_size      = window_size

        # Load all data upfront
        self._load_data()

        # State dimension:
        # physics (7) + regime_probs (6) + latent_z (16) + portfolio (n_stocks) + extra (3)
        self.state_dim  = 7 + 6 + 16 + n_stocks + 3
        self.action_dim = n_stocks

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # Action space: portfolio weights [0, 1] for each stock
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.reset()

    def _load_data(self):
        """Load all required data from the database."""
        engine = get_engine()
        print("Loading environment data...")

        # Market physics
        with engine.connect() as conn:
            physics = pd.read_sql(text("""
                SELECT date, temperature, entropy, order_parameter,
                       lambda_ratio, d_order_parameter, pe_market, mean_acceleration
                FROM market_physics_daily
                WHERE temperature IS NOT NULL
                  AND date BETWEEN :start AND :end
                ORDER BY date
            """), conn, params={"start": self.start_date, "end": self.end_date})

        physics["date"] = pd.to_datetime(physics["date"])
        physics = physics.set_index("date").fillna(0)

        # Regime data
        with engine.connect() as conn:
            regimes = pd.read_sql(text("""
                SELECT date, regime_label, regime_probs
                FROM regime_daily
                WHERE date BETWEEN :start AND :end
                ORDER BY date
            """), conn, params={"start": self.start_date, "end": self.end_date})

        regimes["date"] = pd.to_datetime(regimes["date"])
        regimes = regimes.set_index("date")
        regimes["regime_probs_parsed"] = regimes["regime_probs"].apply(
            lambda x: x if isinstance(x, list) else json.loads(x)
        )

        # Latent states
        with engine.connect() as conn:
            latent = pd.read_sql(text("""
                SELECT date, z_vector
                FROM latent_state_daily
                WHERE date BETWEEN :start AND :end
                ORDER BY date
            """), conn, params={"start": self.start_date, "end": self.end_date})

        latent["date"] = pd.to_datetime(latent["date"])
        latent = latent.set_index("date")
        latent["z_parsed"] = latent["z_vector"].apply(
            lambda x: x if isinstance(x, list) else json.loads(x)
        )

        # Stock returns — pick top N stocks by data completeness
        with engine.connect() as conn:
            returns_raw = pd.read_sql(text("""
                SELECT date, ticker, return_1d
                FROM ohlcv_daily
                WHERE date BETWEEN :start AND :end
                  AND return_1d IS NOT NULL
                ORDER BY date, ticker
            """), conn, params={"start": self.start_date, "end": self.end_date})

        returns_raw["date"] = pd.to_datetime(returns_raw["date"])
        returns_pivot = returns_raw.pivot(
            index="date", columns="ticker", values="return_1d"
        )

        # Pick stocks with most complete data
        completeness = returns_pivot.notna().sum()
        top_tickers  = completeness.nlargest(self.n_stocks).index.tolist()
        returns_pivot = returns_pivot[top_tickers].fillna(0)

        # Align all data on common dates
        common_dates = (
            physics.index
            .intersection(regimes.index)
            .intersection(latent.index)
            .intersection(returns_pivot.index)
        )
        common_dates = common_dates.sort_values()

        self.physics  = physics.loc[common_dates]
        self.regimes  = regimes.loc[common_dates]
        self.latent   = latent.loc[common_dates]
        self.returns  = returns_pivot.loc[common_dates]
        self.dates    = common_dates
        self.tickers  = top_tickers

        # Normalize physics features
        self.physics_mean = self.physics.mean()
        self.physics_std  = self.physics.std().replace(0, 1)

        print(f"Environment loaded: {len(self.dates)} days, {len(self.tickers)} stocks")
        print(f"Date range: {self.dates[0].date()} → {self.dates[-1].date()}")

    def _get_state(self) -> np.ndarray:
        """Build the state vector for the current timestep."""
        date = self.dates[self.current_step]

        # Physics features (normalized)
        physics_raw = self.physics.loc[date].values
        physics_norm = (physics_raw - self.physics_mean.values) / self.physics_std.values
        physics_norm = np.nan_to_num(physics_norm, 0).astype(np.float32)

        # Regime probabilities
        regime_probs = np.array(
            self.regimes.loc[date, "regime_probs_parsed"],
            dtype=np.float32
        )

        # Latent state
        z = np.array(
            self.latent.loc[date, "z_parsed"],
            dtype=np.float32
        )

        # Portfolio state
        portfolio = self.current_weights.astype(np.float32)

        # Extra: current drawdown, days in current regime, portfolio variance
        drawdown    = np.float32(self.current_drawdown)
        regime_days = np.float32(min(self.days_in_regime / 100, 1.0))
        port_var    = np.float32(np.var(self.current_weights))

        state = np.concatenate([
            physics_norm,
            regime_probs,
            z,
            portfolio,
            [drawdown, regime_days, port_var],
        ])

        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step     = 1
        self.portfolio_value  = self.initial_capital
        self.peak_value       = self.initial_capital
        self.current_drawdown = 0.0
        self.current_weights  = np.ones(self.n_stocks) / self.n_stocks
        self.prev_weights     = self.current_weights.copy()
        self.days_in_regime   = 0
        self.prev_regime      = self.regimes.iloc[0]["regime_label"]
        self.portfolio_history = [self.initial_capital]
        self.returns_history   = []

        return self._get_state(), {}

    def step(self, action: np.ndarray):
        """Execute one trading day."""

        # Normalize action to valid portfolio weights
        action = np.clip(action, 0, 1)
        weight_sum = action.sum()
        if weight_sum > 0:
            weights = action / weight_sum
        else:
            weights = np.ones(self.n_stocks) / self.n_stocks

        # Get today's returns
        date           = self.dates[self.current_step]
        stock_returns  = self.returns.loc[date].values

        # Portfolio return
        port_return = np.dot(weights, stock_returns)

        # Transaction costs — proportional to turnover
        turnover = np.abs(weights - self.prev_weights).sum()
        tc_cost  = turnover * self.transaction_cost

        # Net return
        net_return = port_return - tc_cost

        # Update portfolio value
        self.portfolio_value *= (1 + net_return)
        self.portfolio_history.append(self.portfolio_value)
        self.returns_history.append(net_return)

        # Update drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        self.current_drawdown = (
            (self.peak_value - self.portfolio_value) / self.peak_value
        )

        # Track regime persistence
        current_regime = self.regimes.iloc[self.current_step]["regime_label"]
        if current_regime == self.prev_regime:
            self.days_in_regime += 1
        else:
            self.days_in_regime = 0
        self.prev_regime    = current_regime
        self.prev_weights   = weights.copy()
        self.current_weights = weights

        # ── Reward function ──────────────────────────────────────────────────
        # Rolling Sharpe (last 20 days)
        if len(self.returns_history) >= 20:
            recent = np.array(self.returns_history[-20:])
            sharpe = (recent.mean() / (recent.std() + 1e-8)) * np.sqrt(252)
        else:
            sharpe = net_return * np.sqrt(252)

        # Drawdown penalty
        dd_penalty = self.drawdown_penalty * self.current_drawdown

        # Final reward
        reward = float(sharpe - dd_penalty)

        # Advance timestep
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        if done:
            info = self._compute_episode_stats()
        else:
            info = {}

        return self._get_state(), reward, done, False, info

    def _compute_episode_stats(self) -> dict:
        """Compute performance statistics at end of episode."""
        returns = np.array(self.returns_history)
        total_return     = (self.portfolio_value / self.initial_capital) - 1
        ann_return       = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol          = returns.std() * np.sqrt(252)
        sharpe           = ann_return / (ann_vol + 1e-8)
        max_dd           = self.current_drawdown
        calmar           = ann_return / (max_dd + 1e-8)

        return {
            "total_return":    total_return,
            "annual_return":   ann_return,
            "sharpe_ratio":    sharpe,
            "max_drawdown":    max_dd,
            "calmar_ratio":    calmar,
            "final_value":     self.portfolio_value,
        }