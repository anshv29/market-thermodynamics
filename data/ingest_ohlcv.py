import time
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from sqlalchemy import text
from tqdm import tqdm
from db.connection import get_engine
from data.universe import get_universe_tickers

START_DATE = "2004-01-01"
BATCH_SIZE = 25
SLEEP_BETWEEN_BATCHES = 2


def get_last_date_for_ticker(ticker: str):
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(date) FROM ohlcv_daily WHERE ticker = :t"),
            {"t": ticker}
        ).fetchone()
    if result[0] is None:
        return START_DATE
    return str(result[0] + timedelta(days=1))


def insert_ohlcv(records: list):
    if not records:
        return
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO ohlcv_daily
                    (ticker, date, open, high, low, close, adj_close, volume, return_1d)
                VALUES
                    (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume, :return_1d)
                ON CONFLICT (ticker, date) DO NOTHING
            """),
            records
        )
        conn.commit()


def run_ingestion():
    tickers = get_universe_tickers()
    today = str(date.today())
    failed = []
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    print(f"Downloading data for {len(tickers)} tickers in {len(batches)} batches...")

    for batch in tqdm(batches, desc="Batches"):
        batch_str = " ".join(batch)
        try:
            raw = yf.download(
                batch_str,
                start=START_DATE,
                end=today,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"Batch failed: {e}")
            failed.extend(batch)
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df = pd.DataFrame({
                        "open":   raw["Open"][ticker],
                        "high":   raw["High"][ticker],
                        "low":    raw["Low"][ticker],
                        "close":  raw["Close"][ticker],
                        "volume": raw["Volume"][ticker],
                    })

                df = df.dropna(subset=["close"])
                df["adj_close"] = df["close"]
                df["return_1d"] = df["close"].pct_change()
                df.index.name = "date"
                df = df.reset_index()

                records = []
                for _, row in df.iterrows():
                    records.append({
                        "ticker":    ticker,
                        "date":      row["date"].date() if hasattr(row["date"], "date") else row["date"],
                        "open":      float(row["open"])      if pd.notna(row["open"])      else None,
                        "high":      float(row["high"])      if pd.notna(row["high"])      else None,
                        "low":       float(row["low"])       if pd.notna(row["low"])       else None,
                        "close":     float(row["close"])     if pd.notna(row["close"])     else None,
                        "adj_close": float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
                        "volume":    int(row["volume"])      if pd.notna(row["volume"])    else None,
                        "return_1d": float(row["return_1d"]) if pd.notna(row["return_1d"]) else None,
                    })
                insert_ohlcv(records)
            except Exception as e:
                failed.append(ticker)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    if failed:
        print(f"\nFailed tickers: {failed}")
    else:
        print("\nAll tickers ingested successfully.")


if __name__ == "__main__":
    run_ingestion()