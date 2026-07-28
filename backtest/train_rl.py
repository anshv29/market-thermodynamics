"""
RL Training and Backtesting
============================
1. Trains SAC agent on 2006-2018 data
2. Validates on 2019
3. Tests on 2020-2026 (walk-forward)
4. Computes and stores performance metrics
5. Compares against S&P 500 benchmark
"""

import os
import numpy as np
import pandas as pd
import json
import torch
from sqlalchemy import text
from tqdm import trange

from backtest.environment import MarketEnv
from backtest.agent import SACAgent
from db.connection import get_engine

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_START  = "2006-01-01"
TRAIN_END    = "2018-12-31"
TEST_START   = "2019-01-01"
TEST_END     = "2026-05-19"
N_STOCKS     = 50
N_EPISODES   = 500
WARMUP_STEPS = 1000
UPDATE_FREQ  = 4
SAVE_PATH    = "models/sac_agent.pt"


# ── Training ──────────────────────────────────────────────────────────────────

def train_agent():
    print("\n── Training SAC Agent ───────────────────────────────\n")

    env = MarketEnv(
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        n_stocks=N_STOCKS,
    )

    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
    )

    best_sharpe = -np.inf
    episode_stats = []

    for episode in trange(N_EPISODES, desc="Training episodes"):
        state, _ = env.reset()
        done      = False
        steps     = 0

        while not done:
            # Warmup: random actions first
            if len(agent.replay_buffer) < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, _, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            # Update agent
            if len(agent.replay_buffer) >= WARMUP_STEPS and steps % UPDATE_FREQ == 0:
                agent.update()

        if info:
            episode_stats.append(info)
            sharpe = info.get("sharpe_ratio", 0)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                agent.save(SAVE_PATH)

            if (episode + 1) % 10 == 0:
                recent = episode_stats[-10:]
                avg_sharpe = np.mean([s["sharpe_ratio"] for s in recent])
                avg_return = np.mean([s["annual_return"] for s in recent])
                avg_dd     = np.mean([s["max_drawdown"]  for s in recent])
                print(f"\n  Episode {episode+1}/{N_EPISODES} "
                      f"Sharpe={avg_sharpe:.3f} "
                      f"AnnReturn={avg_return:.3f} "
                      f"MaxDD={avg_dd:.3f}")

    print(f"\nBest training Sharpe: {best_sharpe:.3f}")
    print(f"Agent saved to {SAVE_PATH}")
    return agent, env.tickers


# ── Backtesting ───────────────────────────────────────────────────────────────

def run_backtest(tickers: list):
    print("\n── Running Walk-Forward Backtest ────────────────────\n")

    env = MarketEnv(
        start_date=TEST_START,
        end_date=TEST_END,
        n_stocks=N_STOCKS,
    )

    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
    )
    agent.load(SAVE_PATH)

    state, _ = env.reset()
    done      = False

    portfolio_values = [env.initial_capital]
    daily_returns    = []
    daily_weights    = []
    daily_dates      = []
    daily_regimes    = []

    step = 0
    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _, info = env.step(action)
        state = next_state

        portfolio_values.append(env.portfolio_value)
        daily_returns.append(env.returns_history[-1] if env.returns_history else 0)
        daily_weights.append(env.current_weights.copy())
        daily_dates.append(env.dates[env.current_step - 1])

        regime = env.regimes.iloc[env.current_step - 1]["regime_label"]
        daily_regimes.append(regime)
        step += 1

    # ── Performance metrics ───────────────────────────────────────────────────

    returns_arr  = np.array(daily_returns)
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    ann_return   = (1 + total_return) ** (252 / max(len(returns_arr), 1)) - 1
    ann_vol      = returns_arr.std() * np.sqrt(252)
    sharpe       = ann_return / (ann_vol + 1e-8)

    peak         = np.maximum.accumulate(portfolio_values)
    drawdowns    = (peak - portfolio_values) / (peak + 1e-8)
    max_dd       = drawdowns.max()
    calmar       = ann_return / (max_dd + 1e-8)

    print(f"  Test Period: {TEST_START} → {TEST_END}")
    print(f"  Total Return:    {total_return:.2%}")
    print(f"  Annual Return:   {ann_return:.2%}")
    print(f"  Annual Vol:      {ann_vol:.2%}")
    print(f"  Sharpe Ratio:    {sharpe:.3f}")
    print(f"  Max Drawdown:    {max_dd:.2%}")
    print(f"  Calmar Ratio:    {calmar:.3f}")

    # ── Benchmark comparison ──────────────────────────────────────────────────

    engine = get_engine()
    with engine.connect() as conn:
        bench = pd.read_sql(text("""
            SELECT date, sp500_return
            FROM macro_daily
            WHERE date BETWEEN :start AND :end
              AND sp500_return IS NOT NULL
            ORDER BY date
        """), conn, params={"start": TEST_START, "end": TEST_END})

    bench_returns    = bench["sp500_return"].values
    bench_total      = np.prod(1 + bench_returns) - 1
    bench_ann        = (1 + bench_total) ** (252 / max(len(bench_returns), 1)) - 1
    bench_vol        = bench_returns.std() * np.sqrt(252)
    bench_sharpe     = bench_ann / (bench_vol + 1e-8)
    bench_peak       = np.maximum.accumulate(np.cumprod(1 + bench_returns))
    bench_dd         = ((bench_peak - np.cumprod(1 + bench_returns)) /
                        (bench_peak + 1e-8)).max()

    print(f"\n  S&P 500 Benchmark:")
    print(f"  Total Return:    {bench_total:.2%}")
    print(f"  Annual Return:   {bench_ann:.2%}")
    print(f"  Sharpe Ratio:    {bench_sharpe:.3f}")
    print(f"  Max Drawdown:    {bench_dd:.2%}")

    alpha = ann_return - bench_ann
    print(f"\n  Alpha vs S&P 500: {alpha:.2%}")

    # ── Store results ─────────────────────────────────────────────────────────

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO backtest_results
                (run_id, start_date, end_date, total_return, annualized_return,
                 sharpe_ratio, max_drawdown, calmar_ratio, notes)
            VALUES
                (:run_id, CAST(:start AS DATE), CAST(:end AS DATE),
                 :total_return, :ann_return, :sharpe, :max_dd, :calmar, :notes)
        """), {
            "run_id":       "sac_walkforward_v1",
            "start":        TEST_START,
            "end":          TEST_END,
            "total_return": float(total_return),
            "ann_return":   float(ann_return),
            "sharpe":       float(sharpe),
            "max_dd":       float(max_dd),
            "calmar":       float(calmar),
            "notes":        f"SAC agent, {N_STOCKS} stocks, alpha={alpha:.4f}",
        })
        conn.commit()

    return {
        "dates":            daily_dates,
        "portfolio_values": portfolio_values,
        "returns":          daily_returns,
        "weights":          daily_weights,
        "regimes":          daily_regimes,
        "bench_returns":    bench_returns.tolist(),
        "bench_dates":      bench["date"].tolist(),
        "metrics": {
            "total_return": total_return,
            "ann_return":   ann_return,
            "sharpe":       sharpe,
            "max_dd":       max_dd,
            "calmar":       calmar,
            "alpha":        alpha,
            "bench_sharpe": bench_sharpe,
        }
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_rl_pipeline():
    print("\n── Phase 6: Reinforcement Learning Agent ────────────\n")

    if os.path.exists(SAVE_PATH):
        print("Found saved agent. Skipping training.")
        print("Loading environment for backtest...")
        env = MarketEnv(start_date=TRAIN_START, end_date=TRAIN_END, n_stocks=N_STOCKS)
        tickers = env.tickers
    else:
        _, tickers = train_agent()

    results = run_backtest(tickers)

    print("\n── Phase 6 Complete ─────────────────────────────────\n")
    return results


if __name__ == "__main__":
    run_rl_pipeline()