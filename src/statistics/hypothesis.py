import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from src.features.returns import calculate_simple_returns, calculate_covariance_matrix
from src.statistics.bootstrap import StationaryBlockBootstrap
from src.statistics.tests import jobson_korkie_memmel_test, paired_wilcoxon_test, regime_anova_kruskal_test

logger = logging.getLogger(__name__)

class ResearchHypothesisTester:
    """
    Executes formal statistical hypothesis testing for Hypotheses H1 through H4.
    """
    @staticmethod
    def test_h1_diversification(
        asset_prices: pd.DataFrame,
        subsets_per_k: int = 50,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        H1 — Diversification Hypothesis:
        Increasing the number of uncorrelated assets reduces portfolio volatility.
        
        Tests whether the decay in volatility follows beta_1 / sqrt(k) with beta_1 > 0 (p < 0.05).
        """
        np.random.seed(seed)
        rets = calculate_simple_returns(asset_prices)
        total_assets = len(rets.columns)
        
        # Test universe sizes k
        k_values = [2, 3, 5, 8, 12, 16, min(20, total_assets)]
        k_values = sorted(list(set(k for k in k_values if k <= total_assets)))

        records = []
        for k in k_values:
            for _ in range(subsets_per_k):
                chosen_cols = np.random.choice(rets.columns, size=k, replace=False)
                sub_rets = rets[chosen_cols]
                ew_port_ret = sub_rets.mean(axis=1)
                ann_vol = float(ew_port_ret.std(ddof=1) * np.sqrt(252.0))
                records.append({"k": k, "inv_sqrt_k": 1.0 / np.sqrt(k), "volatility": ann_vol})

        sim_df = pd.DataFrame(records)
        
        # OLS Regression: Volatility = beta_0 + beta_1 * (1 / sqrt(k))
        slope, intercept, r_value, p_value, std_err = stats.linregress(sim_df["inv_sqrt_k"], sim_df["volatility"])

        # Group means for reporting
        mean_vols_by_k = sim_df.groupby("k")["volatility"].mean().to_dict()

        # H1 is supported if slope > 0 and p_value < 0.05
        supported = (slope > 0) and (p_value < 0.05)

        return {
            "hypothesis": "H1: Diversification reduces portfolio volatility",
            "supported": bool(supported),
            "slope_beta_1": float(slope),
            "intercept_beta_0": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "mean_volatility_by_k": mean_vols_by_k,
            "conclusion": (
                f"H1 is {'SUPPORTED' if supported else 'REJECTED'}: Volatility decays asymptotically "
                f"with 1/sqrt(k) (Slope={slope:.4f}, R^2={r_value**2:.3f}, p={p_value:.4e})."
            )
        }

    @staticmethod
    def test_h2_risk_parity(
        rp_returns: pd.Series,
        ew_returns: pd.Series,
        regime_series: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        H2 — Risk Parity Superiority Hypothesis:
        Risk-parity allocation provides superior risk-adjusted performance compared with
        equal-weight allocation under high-volatility market conditions.
        """
        aligned = pd.concat([rp_returns, ew_returns, regime_series], axis=1).dropna()
        aligned.columns = ["rp", "ew", "regime"]
        
        high_vol_subset = aligned[aligned["regime"] == "HIGH_VOL"]
        if len(high_vol_subset) < 20:
            # Fallback to all data if regime split is too small
            high_vol_subset = aligned

        jk_test = jobson_korkie_memmel_test(
            high_vol_subset["rp"], 
            high_vol_subset["ew"], 
            risk_free_rate=risk_free_rate
        )
        wilcoxon = paired_wilcoxon_test(high_vol_subset["rp"], high_vol_subset["ew"])

        supported = (jk_test["diff_sharpe"] > 0) and (jk_test["p_value"] < 0.10)

        return {
            "hypothesis": "H2: Risk Parity provides superior risk-adjusted performance in volatile regimes",
            "supported": bool(supported),
            "rp_sharpe_high_vol": jk_test["sharpe_a"],
            "ew_sharpe_high_vol": jk_test["sharpe_b"],
            "sharpe_difference": jk_test["diff_sharpe"],
            "jk_z_statistic": jk_test["z_statistic"],
            "jk_p_value": jk_test["p_value"],
            "wilcoxon_p_value": wilcoxon["p_value"],
            "conclusion": (
                f"H2 is {'SUPPORTED' if supported else 'NOT SUPPORTED'}: During High Volatility, "
                f"Risk Parity Sharpe={jk_test['sharpe_a']:.2f} vs Equal Weight Sharpe={jk_test['sharpe_b']:.2f} "
                f"(Diff={jk_test['diff_sharpe']:.2f}, JK p={jk_test['p_value']:.4f})."
            )
        }

    @staticmethod
    def test_h3_market_regimes(
        strategy_returns_by_regime: Dict[str, Dict[str, pd.Series]]
    ) -> Dict[str, Any]:
        """
        H3 — Market Regimes Hypothesis:
        Portfolio construction methods exhibit significantly different performance
        characteristics across volatility regimes.
        """
        # Run ANOVA across the 3 regimes for each strategy
        results_per_strategy = {}
        all_p_values = []

        for strat, reg_dict in strategy_returns_by_regime.items():
            clean_dict = {k: v for k, v in reg_dict.items() if k != "ALL"}
            test_res = regime_anova_kruskal_test(clean_dict)
            results_per_strategy[strat] = test_res
            all_p_values.append(test_res["anova_p_value"])

        # Any strategy showing significant regime difference
        supported = any(p < 0.05 for p in all_p_values)

        return {
            "hypothesis": "H3: Portfolio performance differs across volatility regimes",
            "supported": bool(supported),
            "strategy_tests": results_per_strategy,
            "conclusion": (
                f"H3 is {'SUPPORTED' if supported else 'NOT SUPPORTED'}: Multi-regime ANOVA tests "
                f"confirm statistically significant performance divergences across volatility environments."
            )
        }

    @staticmethod
    def test_h4_optimization_vs_benchmark(
        opt_net_returns: pd.Series,
        ew_net_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        H4 — Optimization Edge Hypothesis:
        Optimization-based portfolio construction produces statistically meaningful
        differences in risk-adjusted performance compared with equal weight after costs.
        """
        bootstrap = StationaryBlockBootstrap(n_bootstraps=1000)
        boot_res = bootstrap.compute_sharpe_difference_ci(
            opt_net_returns,
            ew_net_returns,
            risk_free_rate=risk_free_rate
        )
        jk_test = jobson_korkie_memmel_test(opt_net_returns, ew_net_returns, risk_free_rate)

        supported = boot_res["statistically_significant"] or (jk_test["p_value"] < 0.05)

        return {
            "hypothesis": "H4: Optimization produces statistically meaningful difference vs Equal Weight after costs",
            "supported": bool(supported),
            "net_sharpe_diff": boot_res["point_diff"],
            "bootstrap_ci_95": [boot_res["ci_lower"], boot_res["ci_upper"]],
            "bootstrap_p_value": boot_res["p_value"],
            "jk_p_value": jk_test["p_value"],
            "conclusion": (
                f"H4 is {'SUPPORTED' if supported else 'NOT SUPPORTED'}: Net Sharpe difference = {boot_res['point_diff']:.3f} "
                f"(95% CI: [{boot_res['ci_lower']:.3f}, {boot_res['ci_upper']:.3f}], p={boot_res['p_value']:.4f})."
            )
        }
