import pytest
import numpy as np
import pandas as pd
from src.features.returns import (
    calculate_simple_returns, calculate_log_returns, 
    calculate_cumulative_returns, calculate_covariance_matrix
)
from src.risk.metrics import (
    calculate_cagr, calculate_annualized_volatility, calculate_sharpe_ratio,
    calculate_sortino_ratio, calculate_max_drawdown, calculate_historical_var,
    calculate_parametric_var, calculate_cornish_fisher_var,
    calculate_expected_shortfall, calculate_beta, calculate_all_risk_metrics
)

def test_return_calculations():
    prices = pd.DataFrame({
        "Asset1": [100.0, 110.0, 121.0],
        "Asset2": [50.0, 50.0, 50.0]
    })
    simple_rets = calculate_simple_returns(prices)
    assert len(simple_rets) == 2
    assert pytest.approx(simple_rets["Asset1"].iloc[0], 0.001) == 0.10
    assert pytest.approx(simple_rets["Asset1"].iloc[1], 0.001) == 0.10
    assert pytest.approx(simple_rets["Asset2"].iloc[0], 0.001) == 0.0

    log_rets = calculate_log_returns(prices)
    assert pytest.approx(log_rets["Asset1"].iloc[0], 0.001) == np.log(1.10)

    cum_rets = calculate_cumulative_returns(simple_rets["Asset1"])
    assert pytest.approx(cum_rets.iloc[-1], 0.001) == 0.21

def test_volatility_and_sharpe():
    np.random.seed(42)
    # Generate daily returns with known annual characteristics
    daily_rets = pd.Series(np.random.normal(0.0004, 0.01, 252)) # ~10% return, ~16% vol
    
    vol = calculate_annualized_volatility(daily_rets, periods_per_year=252)
    assert 0.13 <= vol <= 0.19
    
    sharpe = calculate_sharpe_ratio(daily_rets, risk_free_rate=0.02, periods_per_year=252)
    assert isinstance(sharpe, float)

def test_drawdowns():
    # Sequence: 100 -> 120 -> 90 -> 110 (Max drop is 120 to 90 = 25%)
    returns = pd.Series([0.20, -0.25, 0.2222])
    mdd = calculate_max_drawdown(returns)
    assert pytest.approx(mdd, 0.01) == 0.25

def test_tail_risk_metrics():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 1000))
    
    hist_var = calculate_historical_var(returns, alpha=0.05)
    param_var = calculate_parametric_var(returns, alpha=0.05)
    cf_var = calculate_cornish_fisher_var(returns, alpha=0.05)
    cvar = calculate_expected_shortfall(returns, alpha=0.05)
    
    assert hist_var > 0.0
    assert param_var > 0.0
    assert cf_var > 0.0
    # Expected shortfall must be greater than or equal to VaR for any non-trivial distribution
    assert cvar >= hist_var - 1e-5

def test_beta():
    benchmark = pd.Series([0.01, -0.02, 0.015, -0.01, 0.03])
    # Asset with beta = 1.5
    asset = benchmark * 1.5
    b = calculate_beta(asset, benchmark)
    assert pytest.approx(b, 0.001) == 1.5

def test_calculate_all_risk_metrics():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.0005, 0.01, 500))
    bmk = pd.Series(np.random.normal(0.0004, 0.01, 500))
    
    metrics = calculate_all_risk_metrics(rets, bmk, risk_free_rate=0.02)
    assert "cagr" in metrics
    assert "annualized_volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "var_95_historical" in metrics
    assert "cvar_95" in metrics
    assert "beta" in metrics
