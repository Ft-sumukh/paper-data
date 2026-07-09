import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger(__name__)

def compute_ofi(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    Computes Order Flow Imbalance (OFI) for a given book level as defined by Cont et al. (2014).
    """
    bid_p = df[f"bid_price_{level}"]
    bid_v = df[f"bid_vol_{level}"]
    ask_p = df[f"ask_price_{level}"]
    ask_v = df[f"ask_vol_{level}"]
    
    # Shifted values to get t-1
    bid_p_prev = bid_p.shift(1)
    bid_v_prev = bid_v.shift(1)
    ask_p_prev = ask_p.shift(1)
    ask_v_prev = ask_v.shift(1)
    
    # Compute delta bid volume
    delta_v_b = pd.Series(0.0, index=df.index)
    delta_v_b[bid_p > bid_p_prev] = bid_v[bid_p > bid_p_prev]
    delta_v_b[bid_p == bid_p_prev] = bid_v[bid_p == bid_p_prev] - bid_v_prev[bid_p == bid_p_prev]
    delta_v_b[bid_p < bid_p_prev] = 0.0
    
    # Compute delta ask volume
    delta_v_a = pd.Series(0.0, index=df.index)
    delta_v_a[ask_p < ask_p_prev] = ask_v[ask_p < ask_p_prev]
    delta_v_a[ask_p == ask_p_prev] = ask_v[ask_p == ask_p_prev] - ask_v_prev[ask_p == ask_p_prev]
    delta_v_a[ask_p > ask_p_prev] = 0.0
    
    # OFI = Delta Bid Vol - Delta Ask Vol
    ofi = delta_v_b - delta_v_a
    # Fill NaN for first row
    ofi.iloc[0] = 0.0
    return ofi

def engineer_features(df: pd.DataFrame, max_level: int = 10) -> pd.DataFrame:
    """
    Computes LOB features including spread, microprice, return, OFI, depth imbalance,
    and multi-level book features.
    """
    features = pd.DataFrame(index=df.index)
    
    # Mid-price and Return
    mid_price = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    features["mid_price"] = mid_price
    
    # Spread and Relative Spread
    spread = df["ask_price_1"] - df["bid_price_1"]
    features["spread"] = spread
    features["relative_spread"] = spread / mid_price
    
    # Microprice
    vol_sum_1 = df["ask_vol_1"] + df["bid_vol_1"]
    features["microprice"] = (df["ask_price_1"] * df["bid_vol_1"] + df["bid_price_1"] * df["ask_vol_1"]) / vol_sum_1
    
    # Log Return of Mid-price
    features["mid_return"] = np.log(mid_price / mid_price.shift(1))
    features["mid_return"] = features["mid_return"].fillna(0.0)
    
    # Order Flow Imbalance (OFI) for levels 1 to 5
    for lvl in range(1, min(6, max_level + 1)):
        features[f"ofi_lvl_{lvl}"] = compute_ofi(df, level=lvl)
        
    # Multi-level LOB features (Volume Imbalances, Depth Imbalances)
    for lvl in range(1, max_level + 1):
        bid_vol = df[f"bid_vol_{lvl}"]
        ask_vol = df[f"ask_vol_{lvl}"]
        sum_vol = bid_vol + ask_vol
        sum_vol = sum_vol.replace(0, 1.0) # Avoid divide by zero
        
        # Volume Imbalance
        features[f"vol_imbalance_{lvl}"] = bid_vol - ask_vol
        
        # Depth Imbalance
        features[f"depth_imbalance_{lvl}"] = (bid_vol - ask_vol) / sum_vol
        
        # Spread at level lvl
        features[f"spread_lvl_{lvl}"] = df[f"ask_price_{lvl}"] - df[f"bid_price_{lvl}"]
        
    return features

def create_labels(df: pd.DataFrame, horizon: int = 50, threshold: float = 0.0001, binary: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Creates target prediction labels based on future mid-prices.
    Deletes the last 'horizon' rows to prevent look-ahead bias and handle boundary conditions.
    
    Target Logic:
    - UP (Class 2): future_mid > current_mid * (1 + threshold)
    - DOWN (Class 0): future_mid < current_mid * (1 - threshold)
    - STATIONARY (Class 1): otherwise
    
    If binary=True:
    - UP (1): future_mid > current_mid
    - DOWN (0): otherwise
    """
    if "mid_price" in df.columns:
        mid_price = df["mid_price"]
    else:
        mid_price = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    future_mid = mid_price.shift(-horizon)
    
    if binary:
        # Binary target: 1 if future mid is higher than current mid, else 0
        labels = (future_mid > mid_price).astype(int).values
    else:
        # 3-class target: 0 = DOWN, 1 = STATIONARY, 2 = UP
        labels = np.ones(len(df), dtype=int)  # Default is STATIONARY (1)
        
        up_cond = future_mid > mid_price * (1.0 + threshold)
        down_cond = future_mid < mid_price * (1.0 - threshold)
        
        labels[up_cond] = 2  # UP
        labels[down_cond] = 0  # DOWN
        
    # Trim last 'horizon' rows to remove NaNs from shift(-horizon)
    valid_len = len(df) - horizon
    trimmed_df = df.iloc[:valid_len].copy()
    trimmed_labels = labels[:valid_len]
    
    return trimmed_df, trimmed_labels
