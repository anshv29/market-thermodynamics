"""
Macro data ingestion — updated with VIX term structure.
Downloads VIX9D, VIX (30d), VIX3M, VIX6M plus original macro signals.
"""

import pandas as pd
import yfinance as yf
from datetime import date
from sqlalchemy import text
from db.connection import get_engine

START_DATE = "2004-01-01"


def safe_float(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except:
        return None


def fetch_macro() -> pd.DataFrame:
    today = str(date.today())

    symbols = {
        "vix":            "^VIX",
        "treasury_10y":   "^TNX",
        "treasury_2y":    "^FVX",
        "fed_funds_rate": "^IRX",
        "vix9d":          "^VIX9D",
        "vix3m":          "^VIX3M",
        "vix6m":          "^VIX6M",
    }

    frames = {}
    for name, symbol in symbols.items():
        try:
            raw = yf.download(symbol, start=START_DATE, end=today,
                              auto_adjust=True, progress=False)
            if not raw.empty:
                close = raw["Close"]
                # Handle both Series and DataFrame returns
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                s = close.copy()
                s.index = pd.to_datetime(s.index).date
                s.name = name
                frames[name] = s
                print(f"  {symbol}: {len(s)} rows")
        except Exception as e:
            print(f"  Failed {symbol}: {e}")

    if not frames:
        print("No data downloaded.")
        return pd.DataFrame()

    series_list = list(frames.values())
    df = series_list[0].to_frame()
    for s in series_list[1:]:
        df = df.join(s.to_frame(), how="outer")

    df.index.name = "date"
    df = df.reset_index()
    df = df.dropna(subset=["date"])

    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        df["yield_spread"] = df["treasury_10y"] - df["treasury_2y"]
    else:
        df["yield_spread"] = None

    # VIX term structure slopes
    if "vix9d" in df.columns and "vix" in df.columns:
        df["vix_slope_short"] = df["vix9d"] - df["vix"]      # negative = inverted = fear
    if "vix" in df.columns and "vix3m" in df.columns:
        df["vix_slope_mid"]   = df["vix"] - df["vix3m"]
    if "vix3m" in df.columns and "vix6m" in df.columns:
        df["vix_slope_long"]  = df["vix3m"] - df["vix6m"]

    try:
        spy = yf.download("SPY", start=START_DATE, end=today,
                          auto_adjust=True, progress=False)
        spy_ret = spy["Close"].pct_change().copy()
        spy_ret.index = pd.to_datetime(spy_ret.index).date
        spy_ret.name = "sp500_return"
        spy_df = spy_ret.reset_index()
        spy_df.columns = ["date", "sp500_return"]
        df = df.merge(spy_df, on="date", how="left")
    except Exception as e:
        print(f"SPY failed: {e}")
        df["sp500_return"] = None

    df["usd_index"] = None
    return df


def insert_macro(df: pd.DataFrame):
    if df.empty:
        print("Nothing to insert.")
        return

    engine = get_engine()

    # First add new columns to the table if they don't exist
    with engine.connect() as conn:
        for col in ["vix9d", "vix3m", "vix6m",
                    "vix_slope_short", "vix_slope_mid", "vix_slope_long"]:
            try:
                conn.execute(text(
                    f"ALTER TABLE macro_daily ADD COLUMN IF NOT EXISTS {col} FLOAT"
                ))
            except Exception:
                pass
        conn.commit()

    records = []
    for _, row in df.iterrows():
        d = row["date"]
        if d is None or (isinstance(d, float) and pd.isna(d)):
            continue
        records.append({
            "date":             d,
            "vix":              safe_float(row.get("vix")),
            "fed_funds_rate":   safe_float(row.get("fed_funds_rate")),
            "treasury_10y":     safe_float(row.get("treasury_10y")),
            "treasury_2y":      safe_float(row.get("treasury_2y")),
            "yield_spread":     safe_float(row.get("yield_spread")),
            "sp500_return":     safe_float(row.get("sp500_return")),
            "usd_index":        None,
            "vix9d":            safe_float(row.get("vix9d")),
            "vix3m":            safe_float(row.get("vix3m")),
            "vix6m":            safe_float(row.get("vix6m")),
            "vix_slope_short":  safe_float(row.get("vix_slope_short")),
            "vix_slope_mid":    safe_float(row.get("vix_slope_mid")),
            "vix_slope_long":   safe_float(row.get("vix_slope_long")),
        })

    print(f"Inserting {len(records)} rows...")
    chunk_size = 100
    inserted = 0
    with engine.connect() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            try:
                conn.execute(text("""
                    INSERT INTO macro_daily
                        (date, vix, fed_funds_rate, treasury_10y, treasury_2y,
                         yield_spread, sp500_return, usd_index,
                         vix9d, vix3m, vix6m,
                         vix_slope_short, vix_slope_mid, vix_slope_long)
                    VALUES
                        (:date, :vix, :fed_funds_rate, :treasury_10y, :treasury_2y,
                         :yield_spread, :sp500_return, :usd_index,
                         :vix9d, :vix3m, :vix6m,
                         :vix_slope_short, :vix_slope_mid, :vix_slope_long)
                    ON CONFLICT (date) DO UPDATE SET
                        vix              = EXCLUDED.vix,
                        treasury_10y     = EXCLUDED.treasury_10y,
                        yield_spread     = EXCLUDED.yield_spread,
                        sp500_return     = EXCLUDED.sp500_return,
                        vix9d            = EXCLUDED.vix9d,
                        vix3m            = EXCLUDED.vix3m,
                        vix6m            = EXCLUDED.vix6m,
                        vix_slope_short  = EXCLUDED.vix_slope_short,
                        vix_slope_mid    = EXCLUDED.vix_slope_mid,
                        vix_slope_long   = EXCLUDED.vix_slope_long
                """), chunk)
                conn.commit()
                inserted += len(chunk)
            except Exception as e:
                print(f"Chunk {i} failed: {e}")
                conn.rollback()

    print(f"Done. Inserted/updated {inserted} macro rows.")


if __name__ == "__main__":
    print("Fetching macro data with VIX term structure...")
    df = fetch_macro()
    print(f"Downloaded {len(df)} rows.")
    insert_macro(df)