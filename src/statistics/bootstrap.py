import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Callable, Optional
from src.risk.metrics import calculate_cagr, calculate_annualized_volatility, calculate_sharpe_ratio

class StationaryBlockBootstrap:
    """
    Implements Politis & Romano (1994) Stationary Block Bootstrap
    to account for serial correlation and heteroskedasticity in financial returns.
    """
    def __init__(self, block_size: int = 10, n_bootstraps: int = 1000, seed: int = 42):
        self.block_size = block_size
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    def generate_resamples(self, data: np.ndarray) -> np.ndarray:
        """
        Generates bootstrap indices using circular block resampling.
        """
        np.random.seed(self.seed)
        n = len(data)
        n_blocks = int(np.ceil(n / self.block_size))
        
        bootstrapped_samples = np.zeros((self.n_bootstraps, n))
        
        for b in range(self.n_bootstraps):
            start_indices = np.random.randint(0, n, size=n_blocks)
            resampled_series = []
            for idx in start_indices:
                block = [data[(idx + i) % n] for i in range(self.block_size)]
                resampled_series.extend(block)
            bootstrapped_samples[b] = np.array(resampled_series[:n])
            
        return bootstrapped_samples

    def compute_metric_ci(
        self,
        returns: pd.Series,
        metric_func: Callable[[pd.Series], float],
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Computes (1 - alpha) bootstrap confidence interval for a metric function.
        """
        rets_arr = returns.values
        resamples = self.generate_resamples(rets_arr)
        
        boot_values = np.zeros(self.n_bootstraps)
        for i in range(self.n_bootstraps):
            boot_values[i] = metric_func(pd.Series(resamples[i]))
            
        point_estimate = metric_func(returns)
        ci_lower = float(np.percentile(boot_values, (alpha / 2.0) * 100.0))
        ci_upper = float(np.percentile(boot_values, (1.0 - alpha / 2.0) * 100.0))
        std_err = float(np.std(boot_values, ddof=1))
        
        return {
            "point_estimate": point_estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std_error": std_err
        }

    def compute_sharpe_difference_ci(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        risk_free_rate: float = 0.02,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Computes bootstrap confidence interval for the Sharpe Ratio difference: SR_A - SR_B.
        """
        aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
        n = len(aligned)
        
        np.random.seed(self.seed)
        n_blocks = int(np.ceil(n / self.block_size))
        diffs = np.zeros(self.n_bootstraps)
        
        for b in range(self.n_bootstraps):
            start_indices = np.random.randint(0, n, size=n_blocks)
            sampled_idx = []
            for idx in start_indices:
                sampled_idx.extend([(idx + i) % n for i in range(self.block_size)])
            sample_a = aligned.iloc[sampled_idx[:n], 0]
            sample_b = aligned.iloc[sampled_idx[:n], 1]
            
            sr_a = calculate_sharpe_ratio(sample_a, risk_free_rate)
            sr_b = calculate_sharpe_ratio(sample_b, risk_free_rate)
            diffs[b] = sr_a - sr_b
            
        sr_a_pt = calculate_sharpe_ratio(returns_a, risk_free_rate)
        sr_b_pt = calculate_sharpe_ratio(returns_b, risk_free_rate)
        pt_diff = sr_a_pt - sr_b_pt
        
        ci_lower = float(np.percentile(diffs, (alpha / 2.0) * 100.0))
        ci_upper = float(np.percentile(diffs, (1.0 - alpha / 2.0) * 100.0))
        p_val_approx = float(np.mean(diffs <= 0.0) if pt_diff > 0 else np.mean(diffs >= 0.0)) * 2.0
        p_val_approx = min(p_val_approx, 1.0)
        
        return {
            "point_diff": pt_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_val_approx,
            "statistically_significant": ci_lower > 0 or ci_upper < 0
        }
