import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple

def jobson_korkie_memmel_test(
    returns_a: pd.Series, 
    returns_b: pd.Series, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Computes the Jobson-Korkie test with Memmel (2003) correction for testing the
    equality of two Sharpe ratios: H0: SR_A = SR_B vs H1: SR_A != SR_B.
    """
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    r_a = aligned.iloc[:, 0].values
    r_b = aligned.iloc[:, 1].values
    n = len(aligned)

    if n < 30:
        return {"statistic": 0.0, "p_value": 1.0, "message": "Sample size too small."}

    rf_daily = risk_free_rate / periods_per_year
    
    mu_a = np.mean(r_a) - rf_daily
    mu_b = np.mean(r_b) - rf_daily
    
    var_a = np.var(r_a, ddof=1)
    var_b = np.var(r_b, ddof=1)
    
    std_a = np.sqrt(var_a)
    std_b = np.sqrt(var_b)
    
    cov_ab = np.cov(r_a, r_b)[0, 1]
    gamma = cov_ab / (std_a * std_b) # correlation

    sr_a_annual = (mu_a / std_a) * np.sqrt(periods_per_year)
    sr_b_annual = (mu_b / std_b) * np.sqrt(periods_per_year)
    
    # Memmel asymptotic variance formula
    theta = (
        2.0 * (1.0 - gamma) + 
        0.5 * (sr_a_annual**2 + sr_b_annual**2 - 2.0 * sr_a_annual * sr_b_annual * (gamma**2))
    )
    
    # Test statistic z
    diff_sr = sr_a_annual - sr_b_annual
    z_stat = (diff_sr * np.sqrt(n)) / np.sqrt(max(theta, 1e-6))
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    return {
        "sharpe_a": sr_a_annual,
        "sharpe_b": sr_b_annual,
        "diff_sharpe": diff_sr,
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "reject_null": p_value < 0.05
    }

def paired_wilcoxon_test(returns_a: pd.Series, returns_b: pd.Series) -> Dict[str, Any]:
    """
    Non-parametric Wilcoxon signed-rank test for paired daily return differences.
    """
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    diff_nonzero = diff[diff != 0.0]

    if len(diff_nonzero) < 10:
        return {"statistic": 0.0, "p_value": 1.0, "reject_null": False}

    stat, p_val = stats.wilcoxon(diff_nonzero)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "mean_difference": float(diff.mean()),
        "reject_null": p_val < 0.05
    }

def regime_anova_kruskal_test(returns_by_regime: Dict[str, pd.Series]) -> Dict[str, Any]:
    """
    Computes One-Way ANOVA (parametric) and Kruskal-Wallis (non-parametric) tests
    to evaluate whether returns or Sharpe ratios differ significantly across market regimes.
    """
    clean_groups = [s.dropna().values for s in returns_by_regime.values() if len(s.dropna()) > 5]
    
    if len(clean_groups) < 2:
        return {"anova_p": 1.0, "kruskal_p": 1.0, "reject_null": False}

    anova_stat, anova_p = stats.f_oneway(*clean_groups)
    kruskal_stat, kruskal_p = stats.kruskal(*clean_groups)

    return {
        "anova_f_stat": float(anova_stat),
        "anova_p_value": float(anova_p),
        "kruskal_h_stat": float(kruskal_stat),
        "kruskal_p_value": float(kruskal_p),
        "reject_null": min(anova_p, kruskal_p) < 0.05
    }

def adjust_p_values_holm(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni sequential step-down correction for multiple testing comparisons.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(n)
    
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        multiplier = n - rank
        raw_p = p_values[idx]
        adj_p = min(raw_p * multiplier, 1.0)
        adj_p = max(adj_p, running_max)
        running_max = adj_p
        adjusted[idx] = adj_p
        
    return list(adjusted)
