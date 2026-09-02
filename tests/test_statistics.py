import pytest
import numpy as np
import pandas as pd
from src.statistics import (
    StationaryBlockBootstrap,
    jobson_korkie_memmel_test,
    paired_wilcoxon_test,
    regime_anova_kruskal_test,
    ResearchHypothesisTester
)
from src.risk.metrics import calculate_sharpe_ratio

@pytest.fixture
def return_series():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(42)
    rets_a = pd.Series(np.random.normal(0.0006, 0.01, 300), index=dates)
    rets_b = pd.Series(np.random.normal(0.0004, 0.012, 300), index=dates)
    return rets_a, rets_b

def test_stationary_block_bootstrap(return_series):
    rets_a, _ = return_series
    boot = StationaryBlockBootstrap(block_size=5, n_bootstraps=100, seed=42)
    ci = boot.compute_metric_ci(rets_a, lambda r: calculate_sharpe_ratio(r, risk_free_rate=0.02))
    
    assert "point_estimate" in ci
    assert "ci_lower" in ci
    assert "ci_upper" in ci
    assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]

def test_jobson_korkie_memmel(return_series):
    rets_a, rets_b = return_series
    res = jobson_korkie_memmel_test(rets_a, rets_b)
    
    assert "z_statistic" in res
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0

def test_paired_wilcoxon(return_series):
    rets_a, rets_b = return_series
    res = paired_wilcoxon_test(rets_a, rets_b)
    assert "statistic" in res
    assert 0.0 <= res["p_value"] <= 1.0

def test_regime_anova(return_series):
    rets_a, rets_b = return_series
    rets_c = pd.Series(np.random.normal(-0.001, 0.02, 300), index=rets_a.index)
    
    groups = {"LOW_VOL": rets_a, "NORMAL_VOL": rets_b, "HIGH_VOL": rets_c}
    res = regime_anova_kruskal_test(groups)
    
    assert "anova_p_value" in res
    assert "kruskal_p_value" in res
    assert 0.0 <= res["anova_p_value"] <= 1.0

def test_hypothesis_h1_diversification():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    # Generate 10 uncorrelated assets
    prices_data = np.exp(np.cumsum(np.random.normal(0.0004, 0.01, (100, 10)), axis=0))
    prices = pd.DataFrame(prices_data, index=dates, columns=[f"A{i}" for i in range(10)])
    
    h1_res = ResearchHypothesisTester.test_h1_diversification(prices, subsets_per_k=10)
    assert "slope_beta_1" in h1_res
    assert "supported" in h1_res
    assert h1_res["slope_beta_1"] > 0
