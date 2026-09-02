import os
import sys
import yaml
import logging
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.connection import init_db, SessionLocal
from src.database.crud import bulk_insert_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def run_db_initialization(config_path: str = "configs/default.yaml"):
    logger.info("Initializing relational database schema...")
    init_db()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prices_file = os.path.join(config["paths"]["processed_data_dir"], "prices_clean.csv")
    if not os.path.exists(prices_file):
        logger.warning(f"Clean prices file not found at {prices_file}. Please run scripts/download_data.py first.")
        return

    logger.info(f"Loading cleaned market data from {prices_file}...")
    prices_df = pd.read_csv(prices_file, index_col=0, parse_dates=True)

    db = SessionLocal()
    try:
        bulk_insert_prices(
            db=db,
            clean_prices_df=prices_df,
            asset_metadata=config["data"]["assets"]
        )
        logger.info("Database successfully populated with historical prices and returns.")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize database schema and seed data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    args = parser.parse_args()
    run_db_initialization(args.config)
