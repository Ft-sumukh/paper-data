import os
import pytest
import numpy as np
import pandas as pd
from src.data.downloader import MarketDataDownloader
from src.data.validator import DataValidator
from src.data.cleaner import DataCleaner

def test_market_data_downloader(tmp_path):
    downloader = MarketDataDownloader(raw_data_dir=str(tmp_path / "raw"), metadata_dir=str(tmp_path / "metadata"))
    symbols = ["AAPL", "MSFT", "GOOGL"]
    df = downloader.download_universe(symbols=symbols, start_date="2023-01-01", end_date="2023-12-31", benchmark="SPY")
    
    assert not df.empty
    assert "SPY" in df.columns
    assert "AAPL" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing

def test_data_validator_clean():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    df = pd.DataFrame({"A": np.linspace(100, 110, 10), "B": np.linspace(50, 55, 10)}, index=dates)
    
    report = DataValidator.validate_prices(df)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

def test_data_validator_detects_non_positive_price():
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    df = pd.DataFrame({"A": [100.0, -5.0, 102.0, 103.0, 104.0]}, index=dates)
    
    report = DataValidator.validate_prices(df)
    assert report["is_valid"] is False
    assert "A" in report["negative_or_zero_prices"]

def test_data_cleaner(tmp_path):
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    dates_with_dup = dates.insert(2, dates[1])
    raw_df = pd.DataFrame({
        "A": [100, 101, 101.5, np.nan, 103, 104, 105, 106, 107, 108, 109],
        "B": [50, 51, 51.5, 52, 53, 54, 55, 56, 57, 58, 59]
    }, index=dates_with_dup)
    
    cleaner = DataCleaner(processed_data_dir=str(tmp_path / "processed"))
    clean_df, log = cleaner.clean_prices(raw_df)
    
    assert len(clean_df) == 10
    assert not clean_df.isna().any().any()
    assert log["duplicates_removed"] == 1
