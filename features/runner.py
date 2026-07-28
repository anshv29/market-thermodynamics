"""
Physics Feature Runner
======================
Loads price data from the database, runs the physics feature engine,
and stores all results back into the database.

Run this once to compute features for the full history.
Subsequent runs are incremental.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from db.connection import get_engine, get_session
from db.schema import (
    Universe, MarketPhysicsDaily,
    SectorPhysicsDaily, StockPhysicsDaily
)
from features.physics import compute_all_features


def load_prices_from_db(start_date: str = "2004-01-01") -> tuple:
    """
    Load adjusted close prices and sector map from the database.

    Returns:
        prices  — DataFrame (dates x tickers)
        sectors — dict {ticker: sector_name}
    """
    engine = get_engine()

    print("Loading prices from database...")
    query = text("""
        SELECT o.date, o.ticker, o.adj_close
        FROM ohlcv_daily o
        JOIN universe u ON o.ticker = u.ticker
        WHERE o.date >= :start AND o.adj_close IS NOT NULL
        ORDER BY o.date, o.ticker
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"start": start_date})

    prices = df.pivot(index="date", columns="ticker", values="adj_close")
    prices.index = pd.to_datetime(prices.index)
    print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} tickers.")

    # Load sector map
    session = get_session()
    universe = session.query(Universe).filter_by(is_active=1).all()
    sectors = {u.ticker: u.sector for u in universe if u.sector}
    session.close()

    return prices, sectors


def insert_market_features(market_df: pd.DataFrame):
    """Store market-level physics features in the database."""
    engine = get_engine()
    records = []

    for date, row in market_df.iterrows():
        def sf(val):
            try:
                return float(val) if pd.notna(val) else None
            except:
                return None

        records.append({
            "date":              date.date() if hasattr(date, "date") else date,
            "temperature":       sf(row.get("temperature")),
            "entropy":           sf(row.get("entropy")),
            "mean_acceleration": sf(row.get("mean_acceleration")),
            "order_parameter":   sf(row.get("order_parameter")),
            "lambda_1":          sf(row.get("lambda_1")),
            "lambda_2":          sf(row.get("lambda_2")),
            "lambda_ratio":      sf(row.get("lambda_ratio")),
            "d_order_parameter": sf(row.get("d_order_parameter")),
            "pe_market":         sf(row.get("pe_market")),
        })

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO market_physics_daily
                    (date, temperature, entropy, mean_acceleration,
                     order_parameter, lambda_1, lambda_2, lambda_ratio,
                     d_order_parameter, pe_market)
                VALUES
                    (:date, :temperature, :entropy, :mean_acceleration,
                     :order_parameter, :lambda_1, :lambda_2, :lambda_ratio,
                     :d_order_parameter, :pe_market)
                ON CONFLICT (date) DO UPDATE SET
                    temperature       = EXCLUDED.temperature,
                    entropy           = EXCLUDED.entropy,
                    order_parameter   = EXCLUDED.order_parameter,
                    pe_market         = EXCLUDED.pe_market,
                    d_order_parameter = EXCLUDED.d_order_parameter
            """),
            records
        )
        conn.commit()
    print(f"Inserted {len(records)} market physics rows.")


def insert_sector_features(sector_df: pd.DataFrame):
    """Store sector-level temperature features."""
    engine = get_engine()
    records = []

    for date, row in sector_df.iterrows():
        for sector, val in row.items():
            if pd.isna(val):
                continue
            records.append({
                "date":        date.date() if hasattr(date, "date") else date,
                "sector":      sector,
                "temperature": float(val),
                "entropy":     None,
                "pe_sector":   None,
                "mean_accel":  None,
                "stock_count": None,
            })

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO sector_physics_daily
                    (date, sector, temperature, entropy,
                     pe_sector, mean_accel, stock_count)
                VALUES
                    (:date, :sector, :temperature, :entropy,
                     :pe_sector, :mean_accel, :stock_count)
                ON CONFLICT (date, sector) DO UPDATE SET
                    temperature = EXCLUDED.temperature
            """),
            records
        )
        conn.commit()
    print(f"Inserted {len(records)} sector physics rows.")


def insert_stock_features(stock_level: dict, prices: pd.DataFrame):
    """Store stock-level physics features."""
    engine = get_engine()

    pe     = stock_level["potential_energy"]
    accel  = stock_level["acceleration"]
    jerk   = stock_level["jerk"]
    vel    = stock_level["velocity"]
    vol10  = stock_level["vol_10d"]
    vol60  = stock_level["vol_60d"]

    all_dates  = pe.index
    all_tickers = pe.columns.tolist()

    print("Inserting stock-level features...")
    total_inserted = 0

    # Process in chunks of 30 days to avoid memory issues
    chunk_size = 30
    date_chunks = [all_dates[i:i+chunk_size]
                   for i in range(0, len(all_dates), chunk_size)]

    for chunk in tqdm(date_chunks, desc="Stock features"):
        records = []
        for date in chunk:
            for ticker in all_tickers:
                def sg(df_):
                    try:
                        v = df_.loc[date, ticker]
                        return float(v) if pd.notna(v) else None
                    except:
                        return None

                records.append({
                    "ticker":           ticker,
                    "date":             date.date() if hasattr(date, "date") else date,
                    "potential_energy": sg(pe),
                    "velocity":         sg(vel),
                    "acceleration":     sg(accel),
                    "jerk":             sg(jerk),
                    "vol_10d":          sg(vol10),
                    "vol_60d":          sg(vol60),
                })

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO stock_physics_daily
                        (ticker, date, potential_energy, velocity,
                         acceleration, jerk, vol_10d, vol_60d)
                    VALUES
                        (:ticker, :date, :potential_energy, :velocity,
                         :acceleration, :jerk, :vol_10d, :vol_60d)
                    ON CONFLICT (ticker, date) DO NOTHING
                """),
                records
            )
            conn.commit()
        total_inserted += len(records)

    print(f"Inserted {total_inserted} stock physics rows.")


def run_physics_pipeline(start_date: str = "2004-01-01"):
    """
    Master function — runs the full physics feature pipeline.
    """
    print("-- Phase 2: Physics Feature Engine --\n")

    # 1. Load data
    prices, sectors = load_prices_from_db(start_date)

    # 2. Compute all features
    features = compute_all_features(prices, sectors)

    # 3. Store results
    print("\nStoring market-level features...")
    insert_market_features(features["market_level"])

    print("Storing sector-level features...")
    insert_sector_features(features["sector_level"])

    print("Storing stock-level features...")
    insert_stock_features(features["stock_level"], prices)

    print("\n-- Phase 2 Complete --\n")


if __name__ == "__main__":
    run_physics_pipeline()