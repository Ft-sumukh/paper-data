import pytest
import numpy as np
import pandas as pd
from src.regimes import (
    RollingVolatilityRegimeDetector,
    GaussianMixtureRegimeDetector,
    RegimePerformanceAnalyzer
)

@pytest.fixture
def synthetic_benchmark_returns():
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    # Low vol period (first 200 days), High vol period (next 150 days), Normal vol (last 150 days)
    r1 = np.random.normal(0.0005, 0.005, 200) # 8% vol
    r2 = np.random.normal(-0.001, 0.025, 150) # 40% vol
    r3 = np.random.normal(0.0004, 0.012, 150) # 19% vol
    rets = np.concatenate([r1, r2, r3])
    return pd.Series(rets, index=dates)

def test_rolling_vol_regime_detector(synthetic_benchmark_returns):
    detector = RollingVolatilityRegimeDetector(window=21)
    df = detector.fit_predict(synthetic_benchmark_returns)
    
    assert not df.empty
    assert "regime" in df.columns
    assert "realized_volatility" in df.columns
    assert set(df["regime"].unique()) == {"LOW_VOL", "NORMAL_VOL", "HIGH_VOL"}

def test_gmm_regime_detector(synthetic_benchmark_returns):
    detector = GaussianMixtureRegimeDetector(n_regimes=3, window=21)
    df = detector.fit_predict(synthetic_benchmark_returns)
    
    assert not df.empty
    assert "regime" in df.columns
    assert "regime_code" in df.columns
    assert len(df["regime"].unique()) == 3

def test_regime_performance_analyzer(synthetic_benchmark_returns):
    detector = RollingVolatilityRegimeDetector(window=21)
    regime_df = detector.fit_predict(synthetic_benchmark_returns)
    
    # Strategy returns
    strat_rets = synthetic_benchmark_returns * 0.8
    analyzer = RegimePerformanceAnalyzer()
    res = analyzer.analyze_regime_performance(strat_rets, regime_df["regime"])
    
    assert "LOW_VOL" in res.index
    assert "HIGH_VOL" in res.index
    assert "NORMAL_VOL" in res.index
    assert "ALL" in res.index
    assert "Sharpe Ratio" in res.columns
