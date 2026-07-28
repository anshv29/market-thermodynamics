"""
Simplified RL Agent — 1-Dimensional Action Space
==================================================
Instead of allocating across 50 individual stocks,
the agent learns ONE number: what fraction of the
portfolio should be in equities vs cash.

State:  physics features + latent state + regime probs + current allocation
Action: single float in [0, 1] — equity allocation percentage
Reward: daily return - loss penalty - transaction cost

This is dramatically simpler than the original SAC agent
and should converge properly with 2000 episodes.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import text
from db.connection import get_engine

# ── Seeds ─────────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

engine = get_engine()

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_START      = "2006-01-01"
TRAIN_END        = "2018-12-31"
TEST_START       = "2019-01-01"
TEST_END         = "2026-05-19"
N_EPISODES       = 1000
WARMUP_STEPS     = 200
BATCH_SIZE       = 128
BUFFER_SIZE      = 100_000
LR               = 3e-4
GAMMA            = 0.99
TAU              = 0.005
TRANSACTION_COST = 0.001
INITIAL_CAPITAL  = 1_000_000
N_STOCKS         = 50
UPDATE_FREQ      = 2


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(start_date, end_date):
    """Load all data needed for the environment."""

    with engine.connect() as conn:
        regimes = pd.read_sql(text("""
            SELECT date, regime_label, regime_probs
            FROM regime_daily
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """), conn, params={"start": start_date, "end": end_date})

    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes["probs"] = regimes["regime_probs"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )
    regimes = regimes.set_index("date")

    with engine.connect() as conn:
        physics = pd.read_sql(text("""
            SELECT date, temperature, entropy, order_parameter,
                   lambda_ratio, d_order_parameter, pe_market, mean_acceleration
            FROM market_physics_daily
            WHERE temperature IS NOT NULL
              AND date BETWEEN :start AND :end
            ORDER BY date
        """), conn, params={"start": start_date, "end": end_date})

    physics["date"] = pd.to_datetime(physics["date"])
    physics = physics.set_index("date").fillna(0)

    with engine.connect() as conn:
        latent = pd.read_sql(text("""
            SELECT date, z_vector
            FROM latent_state_daily
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """), conn, params={"start": start_date, "end": end_date})

    latent["date"] = pd.to_datetime(latent["date"])
    latent["z"] = latent["z_vector"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )
    latent = latent.set_index("date")

    with engine.connect() as conn:
        returns_raw = pd.read_sql(text("""
            SELECT date, ticker, return_1d
            FROM ohlcv_daily
            WHERE date BETWEEN :start AND :end
              AND return_1d IS NOT NULL
            ORDER BY date, ticker
        """), conn, params={"start": start_date, "end": end_date})

    returns_raw["date"] = pd.to_datetime(returns_raw["date"])
    returns_pivot = returns_raw.pivot(
        index="date", columns="ticker", values="return_1d"
    )
    completeness  = returns_pivot.notna().sum()
    top_tickers   = completeness.nlargest(N_STOCKS).index.tolist()
    returns_pivot = returns_pivot[top_tickers].fillna(0)

    with engine.connect() as conn:
        fed = pd.read_sql(text("""
            SELECT date, fed_funds_rate
            FROM macro_daily
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """), conn, params={"start": start_date, "end": end_date})

    fed["date"] = pd.to_datetime(fed["date"])
    fed = fed.set_index("date")
    fed["daily_rfr"] = (fed["fed_funds_rate"] / 100) / 252

    with engine.connect() as conn:
        bench = pd.read_sql(text("""
            SELECT date, sp500_return
            FROM macro_daily
            WHERE date BETWEEN :start AND :end
              AND sp500_return IS NOT NULL
            ORDER BY date
        """), conn, params={"start": start_date, "end": end_date})

    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")

    # Align on common dates
    common = (regimes.index
              .intersection(physics.index)
              .intersection(latent.index)
              .intersection(returns_pivot.index)
              .intersection(bench.index))
    common = common.sort_values()

    print(f"Loaded {len(common)} trading days ({start_date} → {end_date})")

    return {
        "regimes": regimes.loc[common],
        "physics": physics.loc[common],
        "latent":  latent.loc[common],
        "returns": returns_pivot.loc[common],
        "fed":     fed.reindex(common).ffill().fillna(0),
        "bench":   bench.loc[common],
        "dates":   common,
    }


# ── Environment ───────────────────────────────────────────────────────────────

class SimpleAllocationEnv:
    """
    Environment where agent learns a single equity allocation %.
    Much simpler than the original 50-stock environment.
    """

    def __init__(self, data: dict):
        self.data    = data
        self.dates   = data["dates"]
        self.n_days  = len(self.dates)

        # State: physics(7) + regime_probs(6) + latent_z(16) + current_alloc(1) + drawdown(1)
        self.state_dim  = 7 + 6 + 16 + 1 + 1
        self.action_dim = 1

        # Normalize physics
        ph = data["physics"]
        self.ph_mean = ph.mean()
        self.ph_std  = ph.std().replace(0, 1)

        self.reset()

    def reset(self):
        self.step_idx        = getattr(self, 'start_idx', 1)
        self.end_idx_limit   = getattr(self, 'end_idx', self.n_days - 1)
        self.portfolio_value = INITIAL_CAPITAL
        self.peak_value      = INITIAL_CAPITAL
        self.current_alloc   = 0.5
        self.drawdown        = 0.0
        self.returns_history = []
        return self._get_state()

    def _get_state(self):
        date = self.dates[self.step_idx]

        # Physics (normalized)
        ph  = self.data["physics"].loc[date].values
        ph  = (ph - self.ph_mean.values) / self.ph_std.values
        ph  = np.nan_to_num(ph, 0).astype(np.float32)

        # Regime probs
        rp  = np.array(self.data["regimes"].loc[date, "probs"],
                       dtype=np.float32)

        # Latent state
        z   = np.array(self.data["latent"].loc[date, "z"],
                       dtype=np.float32)

        # Portfolio state
        alloc = np.array([self.current_alloc], dtype=np.float32)
        dd    = np.array([self.drawdown],      dtype=np.float32)

        return np.concatenate([ph, rp, z, alloc, dd])

    def step(self, action: float):
        # Clip action to valid range
        allocation = float(np.clip(action, 0.0, 1.0))

        date = self.dates[self.step_idx]

        # Equity return (equal weight)
        equity_ret = float(self.data["returns"].loc[date].mean())

        # Cash return
        rfr = float(self.data["fed"].loc[date, "daily_rfr"])

        # Transaction cost
        tc = abs(allocation - self.current_alloc) * TRANSACTION_COST

        # Portfolio return
        port_ret = allocation * equity_ret + (1 - allocation) * rfr - tc

        # Update portfolio
        self.portfolio_value *= (1 + port_ret)
        self.returns_history.append(port_ret)

        # Update drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        self.drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        self.current_alloc = allocation

        # Reward: return bonus - loss penalty - drawdown penalty
        reward = port_ret
        if port_ret < 0:
            reward += port_ret * 0.5   # extra penalty for losses
        reward -= self.drawdown * 0.01  # small drawdown penalty

        self.step_idx += 1
        done = self.step_idx >= self.end_idx_limit

        return self._get_state(), float(reward), done


# ── SAC Networks ──────────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h       = self.net(x)
        mean    = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), -5, 2)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw  = dist.rsample()
        action   = torch.sigmoid(raw)
        log_prob = dist.log_prob(raw)
        log_prob -= torch.log(action * (1 - action) + 1e-6)
        return action, log_prob, torch.sigmoid(mean)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        def q():
            return nn.Sequential(
                nn.Linear(state_dim + 1, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),         nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = q()
        self.q2 = q()

    def forward(self, s, a):
        if a.dim() == 3:
            a = a.squeeze(-1)
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.FloatTensor(np.array(a)).unsqueeze(1),
                torch.FloatTensor(np.array(r)).unsqueeze(1),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(np.array(d)).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


# ── Training ──────────────────────────────────────────────────────────────────

def train(data: dict):
    print(f"\nTraining simplified RL agent for {N_EPISODES} episodes...")

    env       = SimpleAllocationEnv(data)
    state_dim = env.state_dim

    actor    = Actor(state_dim)
    critic   = Critic(state_dim)
    critic_t = Critic(state_dim)
    critic_t.load_state_dict(critic.state_dict())

    actor_opt  = Adam(actor.parameters(),  lr=LR)
    critic_opt = Adam(critic.parameters(), lr=LR)

    log_alpha   = torch.zeros(1, requires_grad=True)
    alpha_opt   = Adam([log_alpha], lr=LR)
    alpha       = log_alpha.exp().item()
    target_ent  = -1.0

    replay = ReplayBuffer()
    best_sharpe = -np.inf
    best_state  = None
    stats       = []

    for ep in range(N_EPISODES):
        # Sample a random 252-day window from training data
        max_start = len(data["dates"]) - 253
        start_idx = np.random.randint(0, max_start)
        env.start_idx = start_idx
        env.end_idx   = start_idx + 252
        state = env.reset()
        done  = False
        steps = 0

        while not done:
            if len(replay) < WARMUP_STEPS:
                action = np.random.uniform(0, 1)
            else:
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).unsqueeze(0)
                    a_t, _, _ = actor.sample(s_t)
                    action = float(a_t.squeeze())

            next_state, reward, done = env.step(action)
            replay.push(state, [action], reward, next_state, float(done))
            state = next_state
            steps += 1

            if len(replay) >= WARMUP_STEPS and steps % UPDATE_FREQ == 0:
                s, a, r, ns, d = replay.sample(BATCH_SIZE)

                with torch.no_grad():
                    na, nlp, _ = actor.sample(ns)
                    q1n, q2n   = critic_t(ns, na)
                    q_next     = torch.min(q1n, q2n) - alpha * nlp
                    q_tgt      = r + GAMMA * (1 - d) * q_next

                q1, q2      = critic(s, a)
                critic_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()

                na2, lp2, _ = actor.sample(s)
                q1n2, q2n2  = critic(s, na2)
                actor_loss  = (alpha * lp2 - torch.min(q1n2, q2n2)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()

                alpha_loss = -(log_alpha * (lp2 + target_ent).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                alpha = log_alpha.exp().item()

                for p, tp in zip(critic.parameters(), critic_t.parameters()):
                    tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        # Episode stats
        rets = np.array(env.returns_history)
        if len(rets) > 20:
            ann_ret = (1 + rets.mean()) ** 252 - 1
            ann_vol = rets.std() * np.sqrt(252)
            sharpe  = ann_ret / (ann_vol + 1e-8)
            max_dd  = float(env.drawdown)
        else:
            sharpe = 0.0

        stats.append(sharpe)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_state  = {
                "actor":    actor.state_dict(),
                "critic":   critic.state_dict(),
                "critic_t": critic_t.state_dict(),
            }

        if (ep + 1) % 100 == 0:
            recent = np.mean(stats[-100:])
            print(f"  Episode {ep+1}/{N_EPISODES} "
                  f"AvgSharpe={recent:.3f} "
                  f"BestSharpe={best_sharpe:.3f} "
                  f"Alpha={alpha:.4f}")

    torch.save(best_state, "models/simple_rl_agent.pt")
    print(f"\nBest training Sharpe: {best_sharpe:.3f}")
    print("Agent saved to models/simple_rl_agent.pt")
    return actor, best_state


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(data: dict, best_state: dict):
    print("\nRunning backtest on test period...")

    env       = SimpleAllocationEnv(data)
    state_dim = env.state_dim
    actor     = Actor(state_dim)
    actor.load_state_dict(best_state["actor"])
    actor.eval()

    state = env.reset()
    done  = False

    portfolio_values = [INITIAL_CAPITAL]
    daily_returns    = []
    allocations      = []
    dates            = []
    regime_labels    = []

    while not done:
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0)
            _, _, action = actor.sample(s_t)
            allocation   = float(action.squeeze())

        next_state, reward, done = env.step(allocation)
        state = next_state

        portfolio_values.append(env.portfolio_value)
        daily_returns.append(env.returns_history[-1])
        allocations.append(allocation)
        dates.append(env.dates[env.step_idx - 1])
        regime_labels.append(int(data["regimes"].iloc[env.step_idx - 1]["regime_label"]))

    return (np.array(portfolio_values),
            np.array(daily_returns),
            np.array(allocations),
            dates,
            np.array(regime_labels))


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(portfolio_values, daily_returns, label, bench_data):
    returns   = np.array(daily_returns)
    total_ret = (portfolio_values[-1] / portfolio_values[0]) - 1
    n         = len(returns)
    ann_ret   = (1 + total_ret) ** (252 / n) - 1
    ann_vol   = returns.std() * np.sqrt(252)
    sharpe    = ann_ret / (ann_vol + 1e-8)
    peak      = np.maximum.accumulate(portfolio_values)
    max_dd    = ((peak - portfolio_values) / (peak + 1e-8)).max()
    calmar    = ann_ret / (max_dd + 1e-8)

    print(f"\n  {label}:")
    print(f"  Total Return:  {total_ret:.2%}")
    print(f"  Annual Return: {ann_ret:.2%}")
    print(f"  Annual Vol:    {ann_vol:.2%}")
    print(f"  Sharpe:        {sharpe:.3f}")
    print(f"  Max Drawdown:  {max_dd:.2%}")
    print(f"  Calmar:        {calmar:.3f}")

    # Benchmark
    br        = bench_data["sp500_return"].values
    b_total   = np.prod(1 + br) - 1
    b_ann     = (1 + b_total) ** (252 / len(br)) - 1
    b_vol     = br.std() * np.sqrt(252)
    b_sharpe  = b_ann / (b_vol + 1e-8)
    b_peak    = np.maximum.accumulate(np.cumprod(1 + br))
    b_maxdd   = ((b_peak - np.cumprod(1 + br)) / (b_peak + 1e-8)).max()

    print(f"\n  S&P 500 Benchmark:")
    print(f"  Total Return:  {b_total:.2%}")
    print(f"  Annual Return: {b_ann:.2%}")
    print(f"  Sharpe:        {b_sharpe:.3f}")
    print(f"  Max Drawdown:  {b_maxdd:.2%}")
    print(f"\n  Alpha: {ann_ret - b_ann:.2%}")

    # Year by year
    dates_arr = pd.to_datetime(bench_data.index)
    print(f"\n  Year-by-year:")
    print(f"  {'Year':<6} {'Strategy':>10} {'S&P 500':>10} {'Alpha':>10}")
    print(f"  {'-'*40}")

    ret_series = pd.Series(daily_returns, index=pd.to_datetime(dates_arr[:len(daily_returns)]))
    br_series  = pd.Series(br, index=dates_arr[:len(br)])

    for year in sorted(ret_series.index.year.unique()):
        sy = ret_series[ret_series.index.year == year]
        by = br_series[br_series.index.year == year]
        if len(sy) < 10:
            continue
        s_ret = np.prod(1 + sy) - 1
        b_ret = np.prod(1 + by) - 1
        print(f"  {year:<6} {s_ret:>10.2%} {b_ret:>10.2%} {s_ret-b_ret:>10.2%}")

    return {
        "total_return": total_ret, "ann_return": ann_ret,
        "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar,
        "alpha": ann_ret - b_ann,
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_results(portfolio_values, daily_returns, allocations,
                 dates, regime_labels, bench_data, metrics):

    bench_cum = np.cumprod(1 + bench_data["sp500_return"].values) * INITIAL_CAPITAL
    agent_cum = portfolio_values[1:]

    REGIME_COLORS = {
        0: "#E05A2B", 1: "#E8C84C", 2: "#4C9BE8",
        3: "#E84C9B", 4: "#4CE8A0", 5: "#9B59B6",
    }

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Equity Curve — Simplified RL Agent vs S&P 500",
            "Learned Equity Allocation %",
            "Drawdown",
            "Market Regime",
        ],
        vertical_spacing=0.07,
        row_heights=[0.35, 0.20, 0.20, 0.25],
    )

    pd_dates = pd.to_datetime(dates)

    fig.add_trace(go.Scatter(
        x=pd_dates, y=agent_cum,
        name="RL Agent", line=dict(color="#4CE8A0", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pd_dates, y=bench_cum[:len(pd_dates)],
        name="S&P 500", line=dict(color="#4C9BE8", width=2, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pd_dates, y=allocations * 100,
        name="Equity %", fill="tozeroy",
        fillcolor="rgba(76,232,160,0.15)",
        line=dict(color="#4CE8A0", width=1),
        showlegend=False,
    ), row=2, col=1)

    def dd(v):
        p = np.maximum.accumulate(v)
        return (v - p) / (p + 1e-8) * 100

    fig.add_trace(go.Scatter(
        x=pd_dates, y=dd(agent_cum),
        fill="tozeroy", fillcolor="rgba(76,232,160,0.15)",
        line=dict(color="#4CE8A0", width=1),
        showlegend=False,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=pd_dates, y=dd(bench_cum[:len(pd_dates)]),
        fill="tozeroy", fillcolor="rgba(76,155,232,0.15)",
        line=dict(color="#4C9BE8", width=1, dash="dash"),
        showlegend=False,
    ), row=3, col=1)

    regime_s = pd.Series(regime_labels, index=pd_dates)
    for r in range(6):
        mask = regime_s == r
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=regime_s[mask].index,
            y=[r] * mask.sum(),
            mode="markers",
            name=f"Regime {r}",
            marker=dict(color=REGIME_COLORS[r], size=4, symbol="square"),
        ), row=4, col=1)

    for start, end in [("2020-02-15","2020-04-30"),("2022-01-01","2022-10-31")]:
        for row in range(1, 5):
            fig.add_vrect(x0=start, x1=end,
                          fillcolor="rgba(255,0,0,0.08)",
                          layer="below", line_width=0, row=row, col=1)

    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text=(f"<b>RL Agent</b><br>"
              f"Ann Return: {metrics['ann_return']:.1%}<br>"
              f"Sharpe: {metrics['sharpe']:.3f}<br>"
              f"Max DD: {metrics['max_dd']:.1%}<br>"
              f"Alpha: {metrics['alpha']:.1%}"),
        showarrow=False,
        font=dict(size=11, color="#4CE8A0"),
        align="left",
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="#4CE8A0",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="Simplified RL Agent — 1D Action Space, 2000 Episodes<br>"
                 "<sup>State = physics + latent + regime. Action = equity allocation %.</sup>",
            font=dict(size=16),
        ),
        height=900,
        plot_bgcolor="#0F1117",
        paper_bgcolor="#0F1117",
        font=dict(color="#CCCCCC"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2A2A3A")
    fig.update_yaxes(showgrid=True, gridcolor="#2A2A3A")

    output = "visualization/simple_rl_backtest.html"
    fig.write_html(output)
    print(f"\nSaved: {output}")
    fig.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n── Simplified RL Agent ──────────────────────────────\n")

    print("Loading training data...")
    train_data = load_data(TRAIN_START, TRAIN_END)

    if os.path.exists("models/simple_rl_agent.pt"):
        print("Found saved agent. Loading...")
        env       = SimpleAllocationEnv(train_data)
        best_state = torch.load("models/simple_rl_agent.pt", weights_only=True)
        actor      = Actor(env.state_dim)
        actor.load_state_dict(best_state["actor"])
    else:
        actor, best_state = train(train_data)

    print("\nLoading test data...")
    test_data = load_data(TEST_START, TEST_END)

    portfolio_values, daily_returns, allocations, dates, regime_labels = \
        backtest(test_data, best_state)

    print("\n── Results ──────────────────────────────────────────")
    metrics = print_metrics(
        portfolio_values, daily_returns,
        "Simplified RL Agent", test_data["bench"]
    )

    plot_results(
        portfolio_values, daily_returns, allocations,
        dates, regime_labels, test_data["bench"], metrics
    )

    print("\n── Done ─────────────────────────────────────────────\n")


if __name__ == "__main__":
    run()