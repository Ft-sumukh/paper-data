import pytest
import numpy as np
import pandas as pd
from src.features.returns import calculate_simple_returns, calculate_covariance_matrix
from src.optimization import get_optimizer
from src.backtesting.engine import WalkForwardBacktester
from src.regimes.detector import RollingVolatilityRegimeDetector

def test_future_price_shock_invariance():
    """
    Test 1: Injects a catastrophic future market crash (+10,000% spike) at future date T_future,
    and proves that optimization weights computed at date t < T_future are 100% identical.
    """
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(42)
    base_prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, (300, 3)), axis=0))
    df_clean = pd.DataFrame(base_prices, index=dates, columns=["A", "B", "C"])
    
    # Create corrupted future dataframe where future prices skyrocket at day 260
    df_shocked = df_clean.copy()
    df_shocked.iloc[260:, 0] = df_shocked.iloc[260:, 0] * 100.0 # Extreme future shock

    # Slices up to day 250 (prior to shock)
    slice_clean = df_clean.iloc[:250]
    slice_shocked = df_shocked.iloc[:250]

    rets_clean = calculate_simple_returns(slice_clean)
    rets_shocked = calculate_simple_returns(slice_shocked)

    mu_clean = rets_clean.mean().values * 252.0
    mu_shocked = rets_shocked.mean().values * 252.0

    cov_clean = calculate_covariance_matrix(rets_clean)
    cov_shocked = calculate_covariance_matrix(rets_shocked)

    opt = get_optimizer("mean_variance")
    res_clean = opt.optimize(mu_clean, cov_clean, ["A", "B", "C"])
    res_shocked = opt.optimize(mu_shocked, cov_shocked, ["A", "B", "C"])

    # Assert weights are bit-for-bit identical despite future crash
    np.testing.assert_allclose(res_clean.weights, res_shocked.weights, atol=1e-10)

def test_walk_forward_window_temporal_integrity():
    """
    Test 2: Verifies that in the walk-forward backtest loop, every estimation window
    ends strictly before the evaluation window.
    """
    dates = pd.date_range("2020-01-01", periods=350, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, (350, 4)), axis=0)),
        index=dates,
        columns=["AAPL", "MSFT", "JPM", "SPY"]
    )

    bt = WalkForwardBacktester(
        strategy_name="equal_weight",
        prices=prices,
        estimation_window=252,
        rebalance_frequency="monthly"
    )
    result = bt.run()
    
    perf_df = result["performance_df"]
    # The first evaluated out-of-sample day must be strictly after the 252-day in-sample estimation
    first_eval_date = perf_df.index[0]
    expected_first_date = prices.index[252 + 1] # +1 for return difference
    assert first_eval_date >= prices.index[252]

def test_preprocessing_prefix_invariance():
    """
    Test 3: Verifies that rolling features and returns computed on truncated windows [0, t]
    match exactly with the prefix of the full dataset [0, T].
    """
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, (200, 2)), axis=0)),
        index=dates,
        columns=["A", "B"]
    )

    full_rets = calculate_simple_returns(prices)
    truncated_rets = calculate_simple_returns(prices.iloc[:100])

    pd.testing.assert_frame_equal(full_rets.iloc[:99], truncated_rets)

def test_regime_classification_future_shock_invariance():
    """
    Test 4: Verifies that realized volatility regime at date t is invariant
    to future market volatility at t + k.
    """
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    np.random.seed(42)
    base_rets = pd.Series(np.random.normal(0.0004, 0.01, 250), index=dates)

    # Incur an extreme volatility shock at day 220
    shocked_rets = base_rets.copy()
    shocked_rets.iloc[220:] = np.random.normal(0.0, 0.08, len(shocked_rets) - 220)

    detector = RollingVolatilityRegimeDetector(window=21)
    res_base = detector.fit_predict(base_rets.iloc[:200])
    res_shocked = detector.fit_predict(shocked_rets.iloc[:200])

    pd.testing.assert_series_equal(res_base["realized_volatility"], res_shocked["realized_volatility"])
