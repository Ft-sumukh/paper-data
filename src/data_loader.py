import os
import urllib.request
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

TSLA_URL = "https://raw.githubusercontent.com/thertrader/Using-random-forest-to-model-limit-order-book-dynamic/master/TSLA_2015-01-07_34200000_57600000_orderbook_10_SAMPLE.csv"

def download_tsla_data(target_path: str) -> bool:
    """
    Downloads the TSLA sample LOB dataset from raw GitHub URL.
    """
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger.info(f"Downloading TSLA LOB dataset from {TSLA_URL}...")
        urllib.request.urlretrieve(TSLA_URL, target_path)
        logger.info("Download completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to download TSLA LOB dataset: {e}")
        return False

def generate_synthetic_lob_data(n_samples: int = 10000, levels: int = 10, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates high-fidelity synthetic Limit Order Book (LOB) data mimicking 
    market microstructure features (bid-ask spread, depth decay, price autocorrelation).
    """
    np.random.seed(random_seed)
    logger.info(f"Generating {n_samples} samples of synthetic LOB level-{levels} data...")

    # Mid price random walk with mean reversion
    mid_price = 200.0
    prices = []
    drift = 0.0
    for _ in range(n_samples):
        # Incorporate autocorrelation in price changes
        change = np.random.normal(0, 0.05) + drift
        mid_price += change
        # Slow mean-reversion drift
        drift = 0.05 * (200.0 - mid_price) + np.random.normal(0, 0.01)
        prices.append(mid_price)

    data = {}
    for level in range(1, levels + 1):
        # Spreads and price steps
        # Level 1 spread is around 0.10 - 0.30
        spread_level_1 = np.random.gamma(shape=2, scale=0.1) + 0.05
        # Deeper levels have wider spreads
        offset = (level - 1) * (np.random.uniform(0.05, 0.15, n_samples))
        
        # Calculate bids and asks
        ask_prices = [p + (spread_level_1 / 2.0) + o for p, o in zip(prices, offset)]
        bid_prices = [p - (spread_level_1 / 2.0) - o for p, o in zip(prices, offset)]
        
        # Scale prices to integer-like NASDAQ values (multiplied by 10000)
        data[f"ask_price_{level}"] = (np.array(ask_prices) * 10000).astype(int)
        data[f"bid_price_{level}"] = (np.array(bid_prices) * 10000).astype(int)

        # Volume decay with level (LOB volume decreases as we go deeper, with random noise)
        base_ask_vol = 100.0 / level
        base_bid_vol = 100.0 / level
        
        # Add autocorrelation to volume sequences
        ask_volumes = []
        bid_volumes = []
        av, bv = base_ask_vol, base_bid_vol
        for _ in range(n_samples):
            av = 0.8 * av + 0.2 * np.random.exponential(base_ask_vol)
            bv = 0.8 * bv + 0.2 * np.random.exponential(base_bid_vol)
            ask_volumes.append(max(1, int(av)))
            bid_volumes.append(max(1, int(bv)))
            
        data[f"ask_vol_{level}"] = ask_volumes
        data[f"bid_vol_{level}"] = bid_volumes

    df = pd.DataFrame(data)
    
    # Interleave columns in LOBSTER format: ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...
    cols = []
    for level in range(1, levels + 1):
        cols.extend([f"ask_price_{level}", f"ask_vol_{level}", f"bid_price_{level}", f"bid_vol_{level}"])
        
    return df[cols]

def load_lob_data(config: Dict) -> pd.DataFrame:
    """
    Loads LOB data based on configuration. Tries to load local file, downloads if missing,
    and falls back to synthetic generation if all else fails.
    """
    data_dir = config["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    target_file = os.path.join(data_dir, "TSLA_LOB.csv")

    loaded = False
    df = None

    if os.path.exists(target_file):
        try:
            logger.info(f"Loading LOB data from local path: {target_file}")
            df = pd.read_csv(target_file, header=None)
            loaded = True
        except Exception as e:
            logger.error(f"Error reading local file {target_file}: {e}")

    if not loaded:
        # Attempt download
        success = download_tsla_data(target_file)
        if success:
            try:
                df = pd.read_csv(target_file, header=None)
                loaded = True
            except Exception as e:
                logger.error(f"Error reading downloaded file: {e}")

    if loaded and len(df) < 1000:
        logger.warning(f"Loaded dataset is too small ({len(df)} rows). Discarding and falling back to synthetic generation.")
        loaded = False

    if not loaded:
        logger.warning("Could not load real dataset. Falling back to synthetic generation.")
        df = generate_synthetic_lob_data(
            n_samples=config["data"]["simulated_samples"],
            levels=config["data"]["levels"],
            random_seed=config["data"]["random_seed"]
        )
        return df

    # Standardize column headers for LOBSTER format
    # columns are: ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...
    cols = []
    num_levels = df.shape[1] // 4
    for level in range(1, num_levels + 1):
        cols.extend([f"ask_price_{level}", f"ask_vol_{level}", f"bid_price_{level}", f"bid_vol_{level}"])
    
    # Slice the columns to match what's in df (in case of truncation/expansion)
    df.columns = cols[:df.shape[1]]
    return df

def split_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs chronological train/validation/test splits to avoid look-ahead bias.
    """
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    logger.info(f"Split data chronologically: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df
