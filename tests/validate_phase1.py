import pandas as pd
from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()


def check(label: str, passed: bool, detail: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {label}")
    if detail:
        print(f"           {detail}")


def run_validation():
    print("\n── Phase 1 Validation ──────────────────────────────\n")

    with engine.connect() as conn:

        # 1. Universe count
        n = conn.execute(text("SELECT COUNT(*) FROM universe")).scalar()
        check("Universe loaded", n >= 490, f"{n} tickers")

        # 2. OHLCV row count
        rows = conn.execute(text("SELECT COUNT(*) FROM ohlcv_daily")).scalar()
        check("OHLCV rows exist", rows > 1_000_000, f"{rows:,} rows")

        # 3. Date range
        result = conn.execute(text("SELECT MIN(date), MAX(date) FROM ohlcv_daily")).fetchone()
        check("OHLCV date range", str(result[0]) <= "2004-12-31",
              f"{result[0]} → {result[1]}")

        # 4. No nulls in close
        nulls = conn.execute(text("SELECT COUNT(*) FROM ohlcv_daily WHERE close IS NULL")).scalar()
        check("No null close prices", nulls == 0, f"{nulls} nulls")

        # 5. Returns populated
        ret_nulls = conn.execute(text("SELECT COUNT(*) FROM ohlcv_daily WHERE return_1d IS NULL")).scalar()
        total = conn.execute(text("SELECT COUNT(*) FROM ohlcv_daily")).scalar()
        check("Returns mostly populated", ret_nulls < total * 0.01,
              f"{ret_nulls:,} nulls out of {total:,}")

        # 6. Crisis dates present
        lehman = conn.execute(text("SELECT COUNT(*) FROM ohlcv_daily WHERE date = '2008-09-15'")).scalar()
        covid = conn.execute(text("SELECT COUNT(*) FROM ohlcv_daily WHERE date = '2020-03-16'")).scalar()
        check("Lehman date present", lehman > 400, f"{lehman} tickers")
        check("COVID crash date present", covid > 400, f"{covid} tickers")

        # 7. Macro data
        macro_rows = conn.execute(text("SELECT COUNT(*) FROM macro_daily")).scalar()
        check("Macro data loaded", macro_rows > 3000, f"{macro_rows} rows")

        vix_check = conn.execute(text(
            "SELECT vix FROM macro_daily WHERE date = '2020-03-16'"
        )).scalar()
        check("VIX on COVID crash > 70", vix_check and vix_check > 70,
              f"VIX = {vix_check}")

        # 8. SPY return on COVID crash
        spy_ret = conn.execute(text(
            "SELECT return_1d FROM ohlcv_daily WHERE ticker = 'SPY' AND date = '2020-03-16'"
        )).scalar()
        check("SPY return on COVID crash < -5%",
              spy_ret and spy_ret < -0.05,
              f"SPY return = {spy_ret:.2%}" if spy_ret else "no data")

        # 9. Sector breakdown
        sectors = conn.execute(text(
            "SELECT sector, COUNT(*) FROM universe GROUP BY sector ORDER BY sector"
        )).fetchall()
        print(f"\n  Sector breakdown:")
        for s, c in sectors:
            print(f"    {s:<45} {c}")

    print("\n── Validation complete ─────────────────────────────\n")


if __name__ == "__main__":
    run_validation()