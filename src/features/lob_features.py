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

def engineer_features(df: pd.DataFrame, levels_to_use: int = 10) -> pd.DataFrame:
    """
    Computes microstructure features:
    - Spreads (absolute and relative)
    - Mid-price and Microprice
    - Volume and depth imbalances
    - Order Flow Imbalance (OFI) across levels
    - Log returns of mid-price
    """
    logger.info(f"Engineering microstructure features up to level {levels_to_use}...")
    df_feat = df.copy()
    
    # 1. Spreads & Mid-prices for Level 1
    df_feat["spread_1"] = df_feat["ask_price_1"] - df_feat["bid_price_1"]
    df_feat["mid_price"] = (df_feat["ask_price_1"] + df_feat["bid_price_1"]) / 2.0
    df_feat["relative_spread"] = df_feat["spread_1"] / df_feat["mid_price"]
    
    # Microprice (weighted by opposite side depth)
    total_vol_l1 = df_feat["bid_vol_1"] + df_feat["ask_vol_1"]
    df_feat["microprice"] = (df_feat["bid_price_1"] * df_feat["ask_vol_1"] + df_feat["ask_price_1"] * df_feat["bid_vol_1"]) / np.maximum(total_vol_l1, 1e-6)
    
    # 2. Imbalance and Spreads for each requested level
    for i in range(1, levels_to_use + 1):
        vol_sum = df_feat[f"bid_vol_{i}"] + df_feat[f"ask_vol_{i}"]
        df_feat[f"imbalance_{i}"] = (df_feat[f"bid_vol_{i}"] - df_feat[f"ask_vol_{i}"]) / np.maximum(vol_sum, 1e-6)
        
        if i > 1:
            df_feat[f"spread_{i}"] = df_feat[f"ask_price_{i}"] - df_feat[f"bid_price_{i}"]
            
        # Compute OFI for this level
        df_feat[f"ofi_level_{i}"] = compute_ofi(df_feat, level=i)
        
    # Multi-level cumulative volume imbalance (e.g. depth 5 and depth 10)
    for depth in [5, min(10, levels_to_use)]:
        bid_vol_cols = [f"bid_vol_{k}" for k in range(1, depth + 1)]
        ask_vol_cols = [f"ask_vol_{k}" for k in range(1, depth + 1)]
        total_bid = df_feat[bid_vol_cols].sum(axis=1)
        total_ask = df_feat[ask_vol_cols].sum(axis=1)
        df_feat[f"depth_imbalance_{depth}"] = (total_bid - total_ask) / np.maximum(total_bid + total_ask, 1e-6)
        
    # Log returns of mid-price
    df_feat["mid_return_1"] = np.log(df_feat["mid_price"] / df_feat["mid_price"].shift(1)).fillna(0.0)
    
    # Clean any inf or nan
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logger.info(f"Feature engineering completed. Features shape: {df_feat.shape}")
    return df_feat

def create_labels(
    df: pd.DataFrame, 
    horizon: int = 50, 
    stationary_threshold: float = 0.0001,
    binary: bool = False
) -> pd.DataFrame:
    """
    Creates target labels based on future price change over horizon k events:
    r_k = (mid_{t+k} - mid_t) / mid_t
    Classes:
    - 0: DOWN (r_k < -threshold)
    - 1: STATIONARY (|r_k| <= threshold)
    - 2: UP (r_k > threshold)
    """
    df_labeled = df.copy()
    if "mid_price" not in df_labeled.columns:
        df_labeled["mid_price"] = (df_labeled["ask_price_1"] + df_labeled["bid_price_1"]) / 2.0
        
    future_mid = df_labeled["mid_price"].shift(-horizon)
    target_return = (future_mid - df_labeled["mid_price"]) / df_labeled["mid_price"]
    
    df_labeled["target_return"] = target_return
    
    if binary:
        # Binary: 1 for UP, 0 for DOWN (ignore stationary or split by sign)
        labels = pd.Series(0, index=df.index)
        labels[target_return > 0] = 1
        df_labeled["label"] = labels
    else:
        # 3-class: 0: Down, 1: Stationary, 2: Up
        labels = pd.Series(1, index=df.index) # default stationary
        labels[target_return < -stationary_threshold] = 0
        labels[target_return > stationary_threshold] = 2
        df_labeled["label"] = labels
        
    # Drop rows at the end that have NaN due to shifting
    df_labeled = df_labeled.dropna(subset=["target_return"]).reset_index(drop=True)
    logger.info(f"Labels created for horizon {horizon}. Label counts:\n{df_labeled['label'].value_counts().to_dict()}")
    return df_labeled
