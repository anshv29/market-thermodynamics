"""
Phase 2 Validation
Checks that physics features are computed correctly.
Key test: do temperature, entropy, and order parameter
spike during known crisis periods?
"""

from sqlalchemy import text
from db.connection import get_engine

engine = get_engine()


def check(label: str, passed: bool, detail: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {label}")
    if detail:
        print(f"           {detail}")


def run_validation():
    print("\n── Phase 2 Validation ──────────────────────────────\n")

    with engine.connect() as conn:

        # 1. Row counts
        mkt = conn.execute(text("SELECT COUNT(*) FROM market_physics_daily")).scalar()
        sec = conn.execute(text("SELECT COUNT(*) FROM sector_physics_daily")).scalar()
        stk = conn.execute(text("SELECT COUNT(*) FROM stock_physics_daily")).scalar()
        check("Market physics rows", mkt > 5000, f"{mkt:,} rows")
        check("Sector physics rows", sec > 50000, f"{sec:,} rows")
        check("Stock physics rows",  stk > 1000000, f"{stk:,} rows")

        # 2. Temperature spikes during crises
        lehman_temp = conn.execute(text("""
            SELECT AVG(temperature) FROM market_physics_daily
            WHERE date BETWEEN '2008-09-01' AND '2008-11-30'
        """)).scalar()
        normal_temp = conn.execute(text("""
            SELECT AVG(temperature) FROM market_physics_daily
            WHERE date BETWEEN '2006-01-01' AND '2006-12-31'
        """)).scalar()
        check("Temperature higher in 2008 crisis vs 2006",
              lehman_temp and normal_temp and lehman_temp > normal_temp,
              f"Crisis: {lehman_temp:.4f} vs Normal: {normal_temp:.4f}")

        covid_temp = conn.execute(text("""
            SELECT AVG(temperature) FROM market_physics_daily
            WHERE date BETWEEN '2020-03-01' AND '2020-04-30'
        """)).scalar()
        check("Temperature spikes in COVID crash",
              covid_temp and covid_temp > normal_temp * 2,
              f"COVID temp: {covid_temp:.4f}")

        # 3. Order parameter spikes during crises
        crisis_psi = conn.execute(text("""
            SELECT AVG(order_parameter) FROM market_physics_daily
            WHERE date BETWEEN '2008-09-01' AND '2008-11-30'
        """)).scalar()
        normal_psi = conn.execute(text("""
            SELECT AVG(order_parameter) FROM market_physics_daily
            WHERE date BETWEEN '2006-01-01' AND '2006-12-31'
        """)).scalar()
        check("Order parameter higher in 2008 crisis",
              crisis_psi and normal_psi and crisis_psi > normal_psi,
              f"Crisis Ψ: {crisis_psi:.4f} vs Normal Ψ: {normal_psi:.4f}")

        # 4. Entropy populated
        null_entropy = conn.execute(text("""
            SELECT COUNT(*) FROM market_physics_daily WHERE entropy IS NULL
        """)).scalar()
        check("Entropy populated", null_entropy < mkt * 0.05,
              f"{null_entropy} nulls")

        # 5. Top 5 highest temperature days — should be crisis days
        print("\n  Top 10 highest temperature days:")
        top_temp = conn.execute(text("""
            SELECT date, temperature, order_parameter, entropy
            FROM market_physics_daily
            WHERE temperature IS NOT NULL
            ORDER BY temperature DESC
            LIMIT 10
        """)).fetchall()
        for row in top_temp:
            print(f"    {row[0]}  T={row[1]:.4f}  Ψ={row[2]:.4f}  S={row[3]:.4f}")

        # 6. Top 5 highest order parameter days
        print("\n  Top 10 highest order parameter (Ψ) days:")
        top_psi = conn.execute(text("""
            SELECT date, order_parameter, temperature
            FROM market_physics_daily
            WHERE order_parameter IS NOT NULL
            ORDER BY order_parameter DESC
            LIMIT 10
        """)).fetchall()
        for row in top_psi:
            print(f"    {row[0]}  Ψ={row[1]:.4f}  T={row[2]:.4f}")

    print("\n── Validation complete ─────────────────────────────\n")


if __name__ == "__main__":
    run_validation()