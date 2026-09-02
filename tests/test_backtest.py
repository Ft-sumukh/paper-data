import pytest
import numpy as np
import pandas as pd
from src.backtesting import WalkForwardBacktester, TransactionCostModel, PerformanceReporter

def test_transaction_cost_model():
    cost_model = TransactionCostModel(transaction_cost_bps=10.0, slippage_bps=5.0)
    
    # 2 assets: shift from [0.5, 0.5] to [0.8, 0.2]
    # Traded volume: |0.8 - 0.5| + |0.2 - 0.5| = 0.3 + 0.3 = 0.6
    w_curr = np.array([0.5, 0.5])
    w_targ = np.array([0.8, 0.2])
    turnover = cost_model.calculate_turnover(w_targ, w_curr)
    assert pytest.approx(turnover, 1e-6) == 0.6
    
    costs = cost_model.calculate_cost(w_targ, w_curr, portfolio_value=1000000.0)
    # Total rate = 15 bps = 0.0015. Cost = 1,000,000 * 0.6 * 0.0015 = $900
    assert pytest.approx(costs["cost_dollars"], 0.01) == 900.0

def test_weight_drift():
    cost_model = TransactionCostModel()
    weights = np.array([0.5, 0.5])
    # Asset 1 surges 10%, Asset 2 is flat
    rets = np.array([0.10, 0.0])
    drifted = cost_model.compute_drifted_weights(weights, rets)
    
    # Growth: 0.5 * 1.1 = 0.55, 0.5 * 1.0 = 0.50. Total = 1.05
    # Drifted w1 = 0.55 / 1.05 = 0.5238
    assert pytest.approx(drifted[0], 0.001) == 0.55 / 1.05
    assert pytest.approx(np.sum(drifted), 1e-6) == 1.0

def test_walk_forward_backtest_execution():
    dates = pd.date_range("2020-01-01", periods=350, freq="B")
    np.random.seed(42)
    # Generate prices for 3 assets + SPY
    log_rets = np.random.normal(0.0004, 0.01, (350, 4))
    prices_data = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    prices = pd.DataFrame(prices_data, index=dates, columns=["AAPL", "MSFT", "JPM", "SPY"])

    backtester = WalkForwardBacktester(
        strategy_name="equal_weight",
        prices=prices,
        benchmark_symbol="SPY",
        estimation_window=252,
        rebalance_frequency="monthly",
        transaction_cost_bps=10.0,
        slippage_bps=5.0
    )
    result = backtester.run()

    perf_df = result["performance_df"]
    assert not perf_df.empty
    # Returns length is 350 - 1 = 349. Out of sample length is 349 - 252 = 97
    assert len(perf_df) == len(prices) - 1 - 252
    
    # Net return should be <= gross return due to transaction costs
    assert np.all(perf_df["net_return"] <= perf_df["gross_return"] + 1e-8)
    
    # Check weights DataFrame
    weights_df = result["weights_df"]
    assert not weights_df.empty
    row_sums = weights_df.sum(axis=1)
    assert np.all(pytest.approx(row_sums.values, 1e-5) == 1.0)
    
    # Check summary metrics
    summary = result["summary"]
    assert "sharpe_net" in summary
    assert "cagr_net" in summary
    assert "max_drawdown_net" in summary
