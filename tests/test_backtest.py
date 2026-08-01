import pytest
import numpy as np
import pandas as pd
from src.backtest import run_backtest
from src.data_loader import generate_synthetic_lob_data

def test_run_backtest():
    # Generate small dummy dataset
    test_df = generate_synthetic_lob_data(n_samples=50, levels=2, random_seed=12)
    
    # 50 mock predictions: mostly hold (1), buy (2), sell (0)
    # class mapping: 0: DOWN, 1: STATIONARY, 2: UP
    predictions = np.ones(50, dtype=int)
    predictions[10:15] = 2  # Buy
    predictions[25:30] = 0  # Sell
    
    # Mock probabilities: confidence of 0.8 for the predictions
    probabilities = np.zeros((50, 3))
    for t in range(50):
        probabilities[t, predictions[t]] = 0.8
        
    config = {
        "backtest": {
            "initial_capital": 10000.0,
            "trading_fee": 0.001,      # 0.1% fee
            "slippage": 0.0002,        # 0.02% slippage
            "min_confidence": 0.5
        }
    }
    
    metrics, bt_df = run_backtest(test_df, predictions, probabilities, config)
    
    assert "gross_return" in metrics
    assert "net_return" in metrics
    assert "sharpe_ratio" in metrics
    assert len(bt_df) == 50
    assert bt_df["portfolio_value"].iloc[0] == 10000.0
    assert not np.isnan(bt_df["portfolio_value"]).any()
