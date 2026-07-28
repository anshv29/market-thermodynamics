"""
Simple Regime Strategy — with Momentum Overlay
================================================
Uses regime signal + momentum to determine equity allocation.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()

TEST_START        = "2019-01-01"
TEST_END          = "2026-05-19"
N_STOCKS          = 50
TRANSACTION_COST  = 0.001
INITIAL_CAPITAL   = 1_000_000

REGIME_ALLOCATIONS = {
    0: 0.00,
    1: 0.30,
    2: 0.80,
    3: 0.60,
    4: 1.00,
    5: 1.00,
}

PROB_THRESHOLD = 0.35


def load_data():
    print("Loading regime probabilities...")
    with engine.connect() as conn:
        regimes = pd.read_sql(text("""
            SELECT date, regime_label, regime_probs
            FROM regime_daily
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """), conn, params={"start": TEST_START, "end": TEST_END})

    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes["probs"] = regimes["regime_probs"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )
    regimes = regimes.set_index("date")

    print("Loading stock returns...")
    with engine.connect() as conn:
        returns_raw = pd.read_sql(text("""
            SELECT date, ticker, return_1d
            FROM ohlcv_daily
            WHERE date BETWEEN :start AND :end
              AND return_1d IS NOT NULL
            ORDER BY date, ticker
        """), conn, params={"start": TEST_START, "end": TEST_END})

    returns_raw["date"] = pd.to_datetime(returns_raw["date"])
    returns_pivot = returns_raw.pivot(
        index="date", columns="ticker", values="return_1d"
    )
    completeness  = returns_pivot.notna().sum()
    top_tickers   = completeness.nlargest(N_STOCKS).index.tolist()
    returns_pivot = returns_pivot[top_tickers].fillna(0)

    print("Loading Fed funds rate...")
    with engine.connect() as conn:
        fed = pd.read_sql(text("""
            SELECT date, fed_funds_rate
            FROM macro_daily
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """), conn, params={"start": TEST_START, "end": TEST_END})

    fed["date"] = pd.to_datetime(fed["date"])
    fed = fed.set_index("date")
    fed["daily_rfr"] = (fed["fed_funds_rate"] / 100) / 252

    print("Loading S&P 500 benchmark...")
    with engine.connect() as conn:
        bench = pd.read_sql(text("""
            SELECT date, sp500_return
            FROM macro_daily
            WHERE date BETWEEN :start AND :end
              AND sp500_return IS NOT NULL
            ORDER BY date
        """), conn, params={"start": TEST_START, "end": TEST_END})

    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")

    print("Loading momentum signals...")
    with engine.connect() as conn:
        prices_raw = pd.read_sql(text("""
            SELECT date, ticker, adj_close
            FROM ohlcv_daily
            WHERE date BETWEEN :start AND :end
              AND adj_close IS NOT NULL
            ORDER BY date, ticker
        """), conn, params={"start": TEST_START, "end": TEST_END})

    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    prices_pivot = prices_raw.pivot(
        index="date", columns="ticker", values="adj_close"
    )
    prices_pivot = prices_pivot[top_tickers].ffill()

    mom_20d   = prices_pivot.pct_change(20).mean(axis=1)
    mom_5d    = prices_pivot.pct_change(5).mean(axis=1)
    mom_accel = mom_20d.diff(5)

    momentum_df = pd.DataFrame({
        "mom_20d":   mom_20d,
        "mom_5d":    mom_5d,
        "mom_accel": mom_accel,
    }).ffill().fillna(0)

    # Align all on common dates
    common_dates = (
        regimes.index
        .intersection(returns_pivot.index)
        .intersection(bench.index)
        .intersection(momentum_df.index)
    )
    common_dates = common_dates.sort_values()

    print(f"Common dates: {len(common_dates)} trading days")
    print(f"Date range: {common_dates[0].date()} → {common_dates[-1].date()}")

    return (
        regimes.loc[common_dates],
        returns_pivot.loc[common_dates],
        fed.reindex(common_dates).ffill().fillna(0),
        bench.loc[common_dates],
        momentum_df.loc[common_dates],
        common_dates,
    )


def get_equity_allocation(regime_probs, mom_20d, mom_5d, mom_accel):
    probs           = np.array(regime_probs)
    dominant_regime = int(np.argmax(probs))
    dominant_prob   = float(probs[dominant_regime])

    if dominant_prob >= PROB_THRESHOLD:
        base_allocation = REGIME_ALLOCATIONS[dominant_regime]
    else:
        base_allocation = sum(
            probs[r] * REGIME_ALLOCATIONS[r] for r in range(len(probs))
        )

    in_defensive      = dominant_regime in [0, 1] and dominant_prob >= PROB_THRESHOLD
    momentum_positive = mom_20d  >  0.01
    momentum_recent   = mom_5d   >  0.005
    momentum_speeding = mom_accel > 0

    if in_defensive and momentum_positive and momentum_recent:
        recovery_boost = 0.30 if momentum_speeding else 0.20
        allocation = min(base_allocation + recovery_boost, 0.80)
    elif dominant_regime in [4, 5] and mom_20d > 0.02:
        allocation = 1.0
    elif dominant_regime in [4, 5] and mom_20d < -0.01:
        allocation = 0.70
    else:
        allocation = base_allocation

    return float(allocation)


def run_backtest(regimes, returns, fed, bench, momentum, dates):
    print("\nRunning backtest...")

    portfolio_value  = INITIAL_CAPITAL
    prev_allocation  = 1.0
    portfolio_values = [INITIAL_CAPITAL]
    daily_returns    = []
    allocations      = []
    regime_labels    = []

    for date in dates:
        probs     = regimes.loc[date, "probs"]
        mom_20d   = float(momentum.loc[date, "mom_20d"])
        mom_5d    = float(momentum.loc[date, "mom_5d"])
        mom_accel = float(momentum.loc[date, "mom_accel"])
        regime    = int(regimes.loc[date, "regime_label"])

        allocation = get_equity_allocation(probs, mom_20d, mom_5d, mom_accel)

        turnover   = abs(allocation - prev_allocation)
        tc         = turnover * TRANSACTION_COST

        if allocation > 0:
            equity_return = float(returns.loc[date].mean())
        else:
            equity_return = 0.0

        rfr = float(fed.loc[date, "daily_rfr"]) if date in fed.index else 0.0

        port_return = (allocation * equity_return +
                      (1 - allocation) * rfr - tc)

        portfolio_value *= (1 + port_return)
        portfolio_values.append(portfolio_value)
        daily_returns.append(port_return)
        allocations.append(allocation)
        regime_labels.append(regime)
        prev_allocation = allocation

    return (
        np.array(portfolio_values),
        np.array(daily_returns),
        np.array(allocations),
        np.array(regime_labels),
    )


def compute_metrics(portfolio_values, daily_returns, label="Strategy"):
    returns   = np.array(daily_returns)
    total_ret = (portfolio_values[-1] / portfolio_values[0]) - 1
    n_days    = len(returns)
    ann_ret   = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol   = returns.std() * np.sqrt(252)
    sharpe    = ann_ret / (ann_vol + 1e-8)
    peak      = np.maximum.accumulate(portfolio_values)
    drawdowns = (peak - portfolio_values) / (peak + 1e-8)
    max_dd    = drawdowns.max()
    calmar    = ann_ret / (max_dd + 1e-8)

    print(f"\n  {label}:")
    print(f"  Total Return:   {total_ret:.2%}")
    print(f"  Annual Return:  {ann_ret:.2%}")
    print(f"  Annual Vol:     {ann_vol:.2%}")
    print(f"  Sharpe Ratio:   {sharpe:.3f}")
    print(f"  Max Drawdown:   {max_dd:.2%}")
    print(f"  Calmar Ratio:   {calmar:.3f}")

    return {
        "total_return": total_ret,
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "calmar":       calmar,
    }


def compute_yearly_returns(portfolio_values, dates):
    vals = np.array(portfolio_values)
    if len(vals) != len(dates):
        vals = vals[:len(dates)]

    df = pd.DataFrame({
        "value": vals,
        "date":  dates,
    }).set_index("date")

    yearly = {}
    for year in df.index.year.unique():
        year_data = df[df.index.year == year]
        if len(year_data) < 2:
            continue
        yearly[year] = (year_data["value"].iloc[-1] /
                        year_data["value"].iloc[0]) - 1
    return yearly


def plot_results(portfolio_values, daily_returns, allocations,
                 regime_labels, bench, dates, metrics, bench_metrics):

    bench_cum  = (1 + bench["sp500_return"]).cumprod()
    bench_cum  = bench_cum / bench_cum.iloc[0] * INITIAL_CAPITAL
    agent_cum  = portfolio_values[1:]

    def dd(vals):
        peak = np.maximum.accumulate(vals)
        return (vals - peak) / (peak + 1e-8)

    agent_dd = dd(agent_cum)
    bench_dd = dd(bench_cum.values)

    REGIME_COLORS = {
        0: "#E05A2B", 1: "#E8C84C", 2: "#4C9BE8",
        3: "#E84C9B", 4: "#4CE8A0", 5: "#9B59B6",
    }

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Equity Curve — Regime + Momentum Strategy vs S&P 500",
            "Equity Allocation % (regime + momentum signal)",
            "Drawdown Comparison",
            "Market Regime",
        ],
        vertical_spacing=0.07,
        row_heights=[0.35, 0.20, 0.20, 0.25],
    )

    fig.add_trace(go.Scatter(
        x=dates, y=agent_cum,
        name="Regime+Momentum Strategy",
        line=dict(color="#4CE8A0", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=bench_cum.values,
        name="S&P 500",
        line=dict(color="#4C9BE8", width=2, dash="dash"),
    ), row=1, col=1)

    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=allocations * 100,
        name="Equity %", fill="tozeroy",
        fillcolor="rgba(76,232,160,0.15)",
        line=dict(color="#4CE8A0", width=1),
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=agent_dd * 100,
        fill="tozeroy",
        fillcolor="rgba(76,232,160,0.15)",
        line=dict(color="#4CE8A0", width=1),
        showlegend=False,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=bench_dd * 100,
        fill="tozeroy",
        fillcolor="rgba(76,155,232,0.15)",
        line=dict(color="#4C9BE8", width=1, dash="dash"),
        showlegend=False,
    ), row=3, col=1)

    regime_series = pd.Series(regime_labels, index=dates)
    for regime in range(6):
        mask = regime_series == regime
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=regime_series[mask].index,
            y=[regime] * mask.sum(),
            mode="markers",
            name=f"Regime {regime}",
            marker=dict(color=REGIME_COLORS[regime], size=4, symbol="square"),
        ), row=4, col=1)

    for start, end, label in [
        ("2020-02-15", "2020-04-30", "COVID"),
        ("2022-01-01", "2022-10-31", "Rate Hikes"),
    ]:
        for row in range(1, 5):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(255,0,0,0.08)",
                layer="below", line_width=0,
                row=row, col=1,
            )

    alpha = metrics["ann_return"] - bench_metrics["ann_return"]
    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text=(f"<b>Regime+Momentum</b><br>"
              f"Total: {metrics['total_return']:.1%}<br>"
              f"Ann: {metrics['ann_return']:.1%}<br>"
              f"Sharpe: {metrics['sharpe']:.3f}<br>"
              f"Max DD: {metrics['max_dd']:.1%}<br>"
              f"Alpha: {alpha:.1%}"),
        showarrow=False,
        font=dict(size=11, color="#4CE8A0"),
        align="left",
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="#4CE8A0",
        borderwidth=1,
    )

    fig.add_annotation(
        x=0.01, y=0.75, xref="paper", yref="paper",
        text=(f"<b>S&P 500</b><br>"
              f"Total: {bench_metrics['total_return']:.1%}<br>"
              f"Ann: {bench_metrics['ann_return']:.1%}<br>"
              f"Sharpe: {bench_metrics['sharpe']:.3f}<br>"
              f"Max DD: {bench_metrics['max_dd']:.1%}"),
        showarrow=False,
        font=dict(size=11, color="#4C9BE8"),
        align="left",
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="#4C9BE8",
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="Regime + Momentum Strategy — Walk-Forward Backtest 2019–2026<br>"
                 "<sup>Regime signal + momentum overlay. No RL.</sup>",
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

    output = "visualization/regime_momentum_backtest.html"
    fig.write_html(output)
    print(f"\nSaved: {output}")
    fig.show()


def run():
    print("\n── Regime + Momentum Strategy Backtest ─────────────\n")

    regimes, returns, fed, bench, momentum, dates = load_data()

    portfolio_values, daily_returns, allocations, regime_labels = run_backtest(
        regimes, returns, fed, bench, momentum, dates
    )

    print("\n── Results ──────────────────────────────────────────")

    metrics = compute_metrics(
        portfolio_values, daily_returns, "Regime + Momentum Strategy"
    )

    bench_returns = bench["sp500_return"].values
    bench_values  = np.cumprod(1 + bench_returns) * INITIAL_CAPITAL
    bench_values  = np.insert(bench_values, 0, INITIAL_CAPITAL)
    bench_metrics = compute_metrics(
        bench_values, bench_returns, "S&P 500 Benchmark"
    )

    alpha = metrics["ann_return"] - bench_metrics["ann_return"]
    print(f"\n  Alpha vs S&P 500: {alpha:.2%}")

    print("\n  Year-by-year returns:")
    strategy_yearly = compute_yearly_returns(portfolio_values, dates)
    bench_yearly    = compute_yearly_returns(bench_values[1:], dates)

    print(f"  {'Year':<6} {'Strategy':>10} {'S&P 500':>10} {'Alpha':>10}")
    print(f"  {'-'*40}")
    for year in sorted(strategy_yearly.keys()):
        s = strategy_yearly.get(year, 0)
        b = bench_yearly.get(year, 0)
        print(f"  {year:<6} {s:>10.2%} {b:>10.2%} {s-b:>10.2%}")

    print(f"\n  Allocation stats:")
    print(f"  Avg equity allocation: {allocations.mean():.1%}")
    print(f"  Days at 100% equity:   {(allocations == 1.0).sum()} ({(allocations == 1.0).mean():.1%})")
    print(f"  Days at 0% equity:     {(allocations == 0.0).sum()} ({(allocations == 0.0).mean():.1%})")

    plot_results(
        portfolio_values, daily_returns, allocations,
        regime_labels, bench, dates, metrics, bench_metrics,
    )

    print("\n── Done ─────────────────────────────────────────────\n")


if __name__ == "__main__":
    run()