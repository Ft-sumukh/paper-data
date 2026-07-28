import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_lob_data
from src.features import engineer_features, create_labels

def test_generate_synthetic_lob_data():
    df = generate_synthetic_lob_data(n_samples=100, levels=10, random_seed=42)
    assert len(df) == 100
    assert df.shape[1] == 40  # 10 levels * 4 columns (ask_p, ask_v, bid_p, bid_v)
    assert "ask_price_1" in df.columns
    assert "bid_vol_10" in df.columns

def test_engineer_features():
    df = generate_synthetic_lob_data(n_samples=50, levels=10, random_seed=42)
    feat_df = engineer_features(df, max_level=10)
    assert len(feat_df) == 50
    assert "mid_price" in feat_df.columns
    assert "spread" in feat_df.columns
    assert "ofi_lvl_1" in feat_df.columns
    assert "depth_imbalance_10" in feat_df.columns
    assert not feat_df.isnull().any().any()  # Verify no NaN values exist

def test_create_labels():
    df = generate_synthetic_lob_data(n_samples=100, levels=10, random_seed=42)
    feat_df = engineer_features(df, max_level=10)
    trimmed_df, labels = create_labels(feat_df, horizon=10, threshold=0.0001, binary=False)
    
    assert len(trimmed_df) == 90
    assert len(labels) == 90
    assert set(np.unique(labels)).issubset({0, 1, 2})
