import pytest
import numpy as np
from src.optimization import (
    EqualWeightOptimizer, MinimumVarianceOptimizer, 
    MeanVarianceOptimizer, RiskParityOptimizer, get_optimizer
)

@pytest.fixture
def market_data():
    symbols = ["AAPL", "MSFT", "JPM", "XOM", "TLT"]
    sectors = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JPM": "Financials",
        "XOM": "Energy",
        "TLT": "Fixed Income"
    }
    # Annual expected returns
    mu = np.array([0.15, 0.14, 0.10, 0.08, 0.04])
    # Annual covariance matrix with realistic volatilities and correlations
    vols = np.array([0.25, 0.22, 0.20, 0.28, 0.12])
    corr = np.array([
        [1.0, 0.7, 0.4, 0.3, -0.2],
        [0.7, 1.0, 0.4, 0.2, -0.2],
        [0.4, 0.4, 1.0, 0.5, -0.1],
        [0.3, 0.2, 0.5, 1.0, -0.1],
        [-0.2, -0.2, -0.1, -0.1, 1.0]
    ])
    cov = np.outer(vols, vols) * corr
    return mu, cov, symbols, sectors

def test_equal_weight_optimizer(market_data):
    mu, cov, symbols, sectors = market_data
    opt = EqualWeightOptimizer()
    res = opt.optimize(mu, cov, symbols, sectors)
    
    assert res.converged
    assert pytest.approx(np.sum(res.weights), 1e-6) == 1.0
    assert np.all(pytest.approx(res.weights, 1e-6) == 0.20)
    assert res.volatility > 0.0

def test_min_variance_optimizer(market_data):
    mu, cov, symbols, sectors = market_data
    opt = MinimumVarianceOptimizer(long_only=True)
    res = opt.optimize(mu, cov, symbols, sectors)
    
    ew_res = EqualWeightOptimizer().optimize(mu, cov, symbols, sectors)
    
    assert res.converged
    assert pytest.approx(np.sum(res.weights), 1e-6) == 1.0
    assert np.all(res.weights >= -1e-6)
    # GMV portfolio must achieve lower or equal volatility compared to 1/N
    assert res.volatility <= ew_res.volatility + 1e-5

def test_mean_variance_optimizer_with_sector_bounds(market_data):
    mu, cov, symbols, sectors = market_data
    opt = MeanVarianceOptimizer(max_sector_weight=0.40)
    res = opt.optimize(mu, cov, symbols, sectors)
    
    assert res.converged
    assert pytest.approx(np.sum(res.weights), 1e-6) == 1.0
    assert np.all(res.weights >= -1e-6)
    
    # Verify Technology sector weight (AAPL + MSFT) <= 40%
    tech_weight = res.symbol_weights["AAPL"] + res.symbol_weights["MSFT"]
    assert tech_weight <= 0.40 + 1e-3

def test_risk_parity_optimizer(market_data):
    mu, cov, symbols, sectors = market_data
    opt = RiskParityOptimizer()
    res = opt.optimize(mu, cov, symbols, sectors)
    
    assert res.converged
    assert pytest.approx(np.sum(res.weights), 1e-6) == 1.0
    assert np.all(res.weights > 0.0)
    
    # In Risk Parity, percentage risk contributions should be approximately equal (20% each)
    expected_prc = 1.0 / len(symbols)
    assert np.all(pytest.approx(res.percentage_risk_contributions, abs=0.03) == expected_prc)

def test_get_optimizer_factory():
    opt1 = get_optimizer("equal_weight")
    assert isinstance(opt1, EqualWeightOptimizer)
    
    opt2 = get_optimizer("min_variance")
    assert isinstance(opt2, MinimumVarianceOptimizer)
    
    opt3 = get_optimizer("mean_variance")
    assert isinstance(opt3, MeanVarianceOptimizer)
    
    opt4 = get_optimizer("risk_parity")
    assert isinstance(opt4, RiskParityOptimizer)
