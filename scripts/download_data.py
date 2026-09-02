import os
import sys
import yaml
import logging
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.downloader import MarketDataDownloader
from src.data.validator import DataValidator
from src.data.cleaner import DataCleaner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def run_data_ingestion(config_path: str = "configs/default.yaml"):
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    symbols = [a["symbol"] for a in config["data"]["assets"]]
    benchmark = config["data"]["benchmark"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    price_col = config["data"].get("price_column", "Adj Close")

    downloader = MarketDataDownloader(
        raw_data_dir=config["paths"]["raw_data_dir"],
        metadata_dir=config["paths"]["metadata_dir"]
    )
    raw_df = downloader.download_universe(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        price_col=price_col
    )

    validator = DataValidator()
    validation_report = validator.validate_prices(raw_df)
    logger.info(f"Validation Report: {validation_report['total_rows']} rows, {validation_report['total_columns']} columns.")

    cleaner = DataCleaner(processed_data_dir=config["paths"]["processed_data_dir"])
    clean_df, clean_log = cleaner.clean_prices(raw_df)
    logger.info(f"Cleaned dataset: {clean_df.shape[0]} rows x {clean_df.shape[1]} assets.")

    return clean_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and clean historical financial market data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_data_ingestion(args.config)
