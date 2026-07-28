"""
Alpaca Paper Trading — Regime + Momentum Strategy
===================================================
Runs daily after market close (4:30pm EST).
1. Downloads today's closing prices via yfinance
2. Computes regime signal from database
3. Computes momentum signal
4. Determines equity allocation
5. Submits orders to Alpaca paper account
6. Logs everything to CSV and database

Run manually or schedule with Windows Task Scheduler.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from sqlalchemy import text
import alpaca_trade_api as tradeapi

from db.connection import get_engine
from data.universe import get_universe_tickers

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("backtest/paper_trade_log.txt"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

N_STOCKS         = 50
TRANSACTION_COST = 0.001

REGIME_ALLOCATIONS = {
    0: 0.00,
    1: 0.30,
    2: 0.80,
    3: 0.60,
    4: 1.00,
    5: 1.00,
}
PROB_THRESHOLD = 0.35

# ── Alpaca Connection ─────────────────────────────────────────────────────────

def get_alpaca():
    api = tradeapi.REST(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        os.getenv("ALPACA_BASE_URL"),
    )
    return api


# ── Get Today's Regime Signal ─────────────────────────────────────────────────

def get_latest_regime():
    """Get the most recent regime label and probabilities from the database."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT date, regime_label, regime_probs
            FROM regime_daily
            ORDER BY date DESC
            LIMIT 1
        """)).fetchone()

    if result is None:
        log.warning("No regime data found. Using default allocation.")
        return None, None, None

    regime_date  = result[0]
    regime_label = int(result[1])
    regime_probs = result[2] if isinstance(result[2], list) else json.loads(result[2])

    log.info(f"Latest regime: {regime_label} (date: {regime_date})")
    log.info(f"Regime probs: {[f'{p:.3f}' for p in regime_probs]}")

    return regime_date, regime_label, regime_probs


# ── Get Momentum Signal ───────────────────────────────────────────────────────

def get_momentum_signal(tickers: list) -> dict:
    """Compute 20-day and 5-day momentum for the stock universe."""
    today     = date.today()
    start     = today - timedelta(days=60)

    log.info("Downloading price data for momentum calculation...")
    raw = yf.download(
        " ".join(tickers[:N_STOCKS]),
        start=str(start),
        end=str(today),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        log.warning("No price data downloaded.")
        return {"mom_20d": 0, "mom_5d": 0, "mom_accel": 0}

    prices = raw["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices

    mom_20d   = prices.pct_change(20).iloc[-1].mean()
    mom_5d    = prices.pct_change(5).iloc[-1].mean()
    mom_accel = (prices.pct_change(20) - prices.pct_change(20).shift(5)).iloc[-1].mean()

    log.info(f"Momentum 20d: {mom_20d:.4f} | 5d: {mom_5d:.4f} | accel: {mom_accel:.4f}")

    return {
        "mom_20d":   float(mom_20d)   if not np.isnan(mom_20d)   else 0,
        "mom_5d":    float(mom_5d)    if not np.isnan(mom_5d)    else 0,
        "mom_accel": float(mom_accel) if not np.isnan(mom_accel) else 0,
    }


# ── Determine Allocation ──────────────────────────────────────────────────────

def get_equity_allocation(regime_probs, mom_20d, mom_5d, mom_accel) -> float:
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
    momentum_positive = mom_20d  > 0.01
    momentum_recent   = mom_5d   > 0.005
    momentum_speeding = mom_accel > 0

    if in_defensive and momentum_positive and momentum_recent:
        recovery_boost = 0.30 if momentum_speeding else 0.20
        allocation = min(base_allocation + recovery_boost, 0.80)
        reason = f"recovery_override (base={base_allocation:.2f}+{recovery_boost:.2f})"
    elif dominant_regime in [4, 5] and mom_20d > 0.02:
        allocation = 1.0
        reason = "bull_confirmed"
    elif dominant_regime in [4, 5] and mom_20d < -0.01:
        allocation = 0.70
        reason = "bull_caution"
    else:
        allocation = base_allocation
        reason = f"regime_{dominant_regime}"

    log.info(f"Allocation: {allocation:.1%} | Reason: {reason}")
    return float(allocation)


# ── Get Current Portfolio ─────────────────────────────────────────────────────

def get_current_portfolio(api) -> dict:
    """Get current positions from Alpaca."""
    positions = api.list_positions()
    portfolio = {}
    for p in positions:
        portfolio[p.symbol] = {
            "qty":        float(p.qty),
            "market_val": float(p.market_value),
        }
    return portfolio


# ── Execute Trades ────────────────────────────────────────────────────────────

def execute_allocation(api, tickers: list, target_allocation: float):
    """
    Rebalance portfolio to target equity allocation.
    Equal-weight across top N stocks.
    """
    account = api.get_account()
    total_equity = float(account.equity)
    log.info(f"Account equity: ${total_equity:,.2f}")

    target_equity_value = total_equity * target_allocation
    per_stock_value     = target_equity_value / N_STOCKS if target_allocation > 0 else 0

    log.info(f"Target allocation: {target_allocation:.1%}")
    log.info(f"Total equity value: ${target_equity_value:,.2f}")
    log.info(f"Per stock target: ${per_stock_value:,.2f}")

    # Get current prices
    current_positions = get_current_portfolio(api)

    orders_placed = []

    for ticker in tickers[:N_STOCKS]:
        try:
            # Get current price
            bar = api.get_latest_bar(ticker)
            current_price = float(bar.c)

            target_shares = int(per_stock_value / current_price)
            current_shares = int(current_positions.get(ticker, {}).get("qty", 0))
            diff = target_shares - current_shares

            if diff == 0:
                continue

            side = "buy" if diff > 0 else "sell"
            qty  = abs(diff)

            if qty == 0:
                continue

            order = api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            orders_placed.append({
                "ticker": ticker,
                "side":   side,
                "qty":    qty,
                "price":  current_price,
            })
            log.info(f"  {side.upper()} {qty} {ticker} @ ~${current_price:.2f}")

        except Exception as e:
            log.warning(f"  Failed to order {ticker}: {e}")
            continue

    # If allocation is 0, close all positions
    if target_allocation == 0:
        log.info("Closing all positions (crash regime)...")
        try:
            api.close_all_positions()
            log.info("All positions closed.")
        except Exception as e:
            log.warning(f"Failed to close positions: {e}")

    return orders_placed


# ── Log to CSV ────────────────────────────────────────────────────────────────

def log_to_csv(today, regime_label, regime_probs,
               momentum, allocation, orders, account_value):
    """Append today's trading summary to a CSV file."""
    log_path = "backtest/paper_trade_history.csv"

    row = {
        "date":           str(today),
        "regime_label":   regime_label,
        "regime_probs":   json.dumps([round(p, 3) for p in regime_probs]),
        "mom_20d":        round(momentum["mom_20d"], 4),
        "mom_5d":         round(momentum["mom_5d"], 4),
        "allocation":     round(allocation, 3),
        "n_orders":       len(orders),
        "account_value":  round(account_value, 2),
    }

    df = pd.DataFrame([row])

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

    log.info(f"Logged to {log_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info(f"Paper Trading Run — {date.today()}")
    log.info("=" * 60)

    # Connect to Alpaca
    api = get_alpaca()
    account = api.get_account()
    log.info(f"Account status: {account.status}")
    log.info(f"Account equity: ${float(account.equity):,.2f}")
    log.info(f"Buying power:   ${float(account.buying_power):,.2f}")

    # Get tickers
    tickers = get_universe_tickers()
    log.info(f"Universe: {len(tickers)} tickers")

    # Get regime signal
    regime_date, regime_label, regime_probs = get_latest_regime()

    if regime_probs is None:
        log.error("No regime signal available. Exiting.")
        return

    # Get momentum signal
    momentum = get_momentum_signal(tickers)

    # Determine allocation
    allocation = get_equity_allocation(
        regime_probs,
        momentum["mom_20d"],
        momentum["mom_5d"],
        momentum["mom_accel"],
    )

    # Execute trades
    orders = execute_allocation(api, tickers, allocation)
    log.info(f"Orders placed: {len(orders)}")

    # Wait a moment for orders to register
    import time
    time.sleep(3)

    # Get updated account value including open positions at current prices
    account    = api.get_account()
    portfolio  = api.list_positions()
    
    # Calculate total portfolio value
    cash       = float(account.cash)
    equity_val = sum(float(p.market_value) for p in portfolio)
    acct_value = cash + equity_val

    log.info(f"Cash: ${cash:,.2f} | Equity: ${equity_val:,.2f} | Total: ${acct_value:,.2f}")

    # Log to CSV
    log_to_csv(
        date.today(), regime_label, regime_probs,
        momentum, allocation, orders, acct_value,
    )

    log.info("=" * 60)
    log.info(f"Done. Account value: ${acct_value:,.2f}")
    log.info("=" * 60)


if __name__ == "__main__":
    run()