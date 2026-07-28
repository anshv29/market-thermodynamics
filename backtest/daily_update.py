"""
Daily Update Pipeline
======================
Runs every day at 4:30pm EST automatically.
Updates all data and executes paper trades.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Fix Windows encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("backtest/daily_update_log.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def run_step(module: str, description: str) -> bool:
    log.info(f"Starting: {description}...")
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-m", module],
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            log.info(f"Completed: {description}")
            return True
        else:
            log.error(f"Failed: {description}")
            log.error(result.stderr[-500:] if result.stderr else "No error output")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"Timeout: {description}")
        return False
    except Exception as e:
        log.error(f"Error in {description}: {e}")
        return False


def run():
    log.info("=" * 60)
    log.info(f"Daily Update Pipeline -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    steps = [
        ("data.ingest_ohlcv",        "Ingesting price data"),
        ("data.ingest_macro",         "Ingesting macro data"),
        ("features.runner",           "Computing physics features"),
        ("models.gnn_runner",         "Updating GNN embeddings"),
        ("models.vae_runner",         "Updating VAE latent states"),
        ("models.transformer_runner", "Updating transformer"),
        ("backtest.paper_trade",      "Executing paper trade"),
    ]

    results = []
    for module, description in steps:
        success = run_step(module, description)
        results.append((description, success))

        if not success and module != "models.transformer_runner":
            log.error(f"Critical step failed: {description}. Stopping pipeline.")
            break

    log.info("\nDaily Update Summary:")
    for description, success in results:
        status = "PASS" if success else "FAIL"
        log.info(f"  [{status}] {description}")

    log.info("=" * 60)
    log.info("Daily update complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    run()