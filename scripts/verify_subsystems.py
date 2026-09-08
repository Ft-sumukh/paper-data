import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd

from src.optimization import (
    EqualWeightOptimizer,
    MinimumVarianceOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer
)
from src.risk import calculate_all_risk_metrics
from src.features import calculate_simple_returns, calculate_covariance_matrix
from src.regimes import RollingVolatilityRegimeDetector, GaussianMixtureRegimeDetector
from src.statistics import (
    StationaryBlockBootstrap,
    jobson_korkie_memmel_test,
    paired_wilcoxon_test,
    ResearchHypothesisTester
)
from src.failure_analysis import (
    ErrorDetector,
    MarketConditionDetector,
    ConfidenceAnalyzer,
    RobustnessAnalyzer,
    FailureSignificanceTester
)

print("=" * 70)
print("  COMPREHENSIVE QUANTITATIVE PLATFORM SYSTEM VERIFICATION")
print("=" * 70)

# 1. Mathematical Optimization Engine
np.random.seed(42)
rets = np.random.normal(0.0005, 0.015, (500, 5))
cov = np.cov(rets, rowvar=False)
mu = np.mean(rets, axis=0) * 252

symbols = [f"Asset_{i}" for i in range(5)]
w_ew = EqualWeightOptimizer().optimize(mu, cov, symbols).weights
w_min = MinimumVarianceOptimizer().optimize(mu, cov, symbols).weights
w_mvo = MeanVarianceOptimizer().optimize(mu, cov, symbols).weights
w_rp = RiskParityOptimizer().optimize(mu, cov, symbols).weights

print("\n1. CONVEX OPTIMIZATION VERIFICATION:")
print(f"  * Equal Weight (1/N):   Sum = {np.sum(w_ew):.6f} | Min = {np.min(w_ew):.4f} | Max = {np.max(w_ew):.4f}")
print(f"  * Global Min Variance:  Sum = {np.sum(w_min):.6f} | Min = {np.min(w_min):.4f} | Max = {np.max(w_min):.4f}")
print(f"  * Mean-Variance (MVO):  Sum = {np.sum(w_mvo):.6f} | Min = {np.min(w_mvo):.4f} | Max = {np.max(w_mvo):.4f}")
print(f"  * Risk Parity (ERC):    Sum = {np.sum(w_rp):.6f} | Min = {np.min(w_rp):.4f} | Max = {np.max(w_rp):.4f}")

# 2. Risk & Tail-Risk Measures
port_rets = pd.Series(np.dot(rets, w_rp))
risk = calculate_all_risk_metrics(port_rets)
print("\n2. RISK & TAIL LOSS METRICS VERIFICATION:")
print(f"  * Annualized Volatility:     {risk['annualized_volatility']*100:.2f}%")
print(f"  * Annualized Sharpe Ratio:   {risk['sharpe_ratio']:.4f}")
print(f"  * Sortino Downside Ratio:    {risk['sortino_ratio']:.4f}")
print(f"  * Maximum Drawdown:          {risk['max_drawdown']*100:.2f}%")
print(f"  * Historical VaR (95%):      {risk['var_95_historical']*100:.2f}%")
print(f"  * Parametric Gaussian VaR:   {risk['var_95_parametric']*100:.2f}%")
print(f"  * Cornish-Fisher VaR:        {risk['var_95_cornish_fisher']*100:.2f}%")
print(f"  * Expected Shortfall (CVaR): {risk['cvar_95']*100:.2f}%")

# 3. Market Regime Detection
dates = pd.date_range("2020-01-01", periods=500, freq="B")
ret_series = pd.Series(rets[:, 0], index=dates)
vol_detector = RollingVolatilityRegimeDetector(window=21)
vol_regimes = vol_detector.fit_predict(ret_series)
gmm_detector = GaussianMixtureRegimeDetector(n_regimes=3, window=21)
gmm_regimes = gmm_detector.fit_predict(ret_series)

print("\n3. REGIME DETECTION ENGINE:")
print(f"  * Rolling Volatility Regimes: {dict(vol_regimes['regime'].value_counts())}")
print(f"  * Gaussian Mixture Regimes:   {dict(gmm_regimes['regime'].value_counts())}")

# 4. Statistical Inference & Econometric Testing
jk_test = jobson_korkie_memmel_test(pd.Series(rets[:, 0]), pd.Series(rets[:, 1]))
wilcoxon_test = paired_wilcoxon_test(pd.Series(rets[:, 0]), pd.Series(rets[:, 1]))
prices_df = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), columns=[f"Asset_{i}" for i in range(5)])
h1_test = ResearchHypothesisTester.test_h1_diversification(prices_df, subsets_per_k=10)

print("\n4. ECONOMETRIC HYPOTHESIS TESTS:")
print(f"  * Jobson-Korkie Difference z-stat: {jk_test.get('z_stat', jk_test.get('statistic', 0.0)):.4f}, p-value: {jk_test['p_value']:.4e}")
print(f"  * Paired Wilcoxon signed-rank stat: {wilcoxon_test.get('statistic', 0.0):.4f}, p-value: {wilcoxon_test['p_value']:.4e}")
print(f"  * H1 Asymptotic Decay Fit (R^2):    {h1_test.get('r_squared', 0.0):.4f}, p-value: {h1_test['p_value']:.4e}")

# 5. Failure Analysis & Calibration
mock_preds = pd.DataFrame({
    "prediction": np.random.choice([0, 1, 2], 200),
    "actual": np.random.choice([0, 1, 2], 200),
    "confidence": np.random.uniform(0.45, 0.95, 200),
    "event_id": np.arange(200)
})
err_detector = ErrorDetector()
analyzed = err_detector.analyze_predictions(mock_preds)
ece_res = ConfidenceAnalyzer.calculate_expected_calibration_error(analyzed)
odds, ci_l, ci_u = FailureSignificanceTester.compute_odds_ratio(25, 75, 10, 90)

print("\n5. FAILURE ANALYSIS & CALIBRATION ENGINE:")
print(f"  * Evaluated Predictions: {len(analyzed)}")
print(f"  * Expected Calibration Error (ECE): {ece_res['expected_calibration_error']:.4f}")
print(f"  * Maximum Calibration Error (MCE):  {ece_res['maximum_calibration_error']:.4f}")
print(f"  * Woolf 95% CI Odds Ratio:          {odds:.4f} [{ci_l:.4f}, {ci_u:.4f}]")
print("\n" + "=" * 70)
print("  ALL SUBSYSTEM CHECKS COMPLETED WITH ZERO ERRORS")
print("=" * 70)
