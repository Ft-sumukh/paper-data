import logging
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes bootstrap confidence intervals for a given metric.
    Default metric is macro F1 score.
    """
    if metric_func is None:
        metric_func = lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)
        
    np.random.seed(random_seed)
    n = len(y_true)
    boot_metrics = []
    
    # Calculate point estimate
    point_estimate = metric_func(y_true, y_pred)
    
    for _ in range(n_resamples):
        indices = np.random.choice(n, size=n, replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        
        try:
            val = metric_func(boot_y_true, boot_y_pred)
            boot_metrics.append(val)
        except Exception:
            continue
            
    if not boot_metrics:
        return point_estimate, point_estimate, point_estimate
        
    alpha = 1.0 - confidence_level
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    
    lower_bound = np.percentile(boot_metrics, lower_pct)
    upper_bound = np.percentile(boot_metrics, upper_pct)
    
    return float(point_estimate), float(lower_bound), float(upper_bound)

def diebold_mariano_test(
    y_true: np.ndarray,
    proba_1: np.ndarray,
    proba_2: np.ndarray,
    num_classes: int = 3
) -> Tuple[float, float]:
    """
    Performs a Diebold-Mariano test comparing the prediction accuracy of two models.
    We define the loss series at step t as the Brier score (squared error of the forecast probability
    for the true class):
        e_t = (1 - proba[t, true_class])^2
    
    Returns:
        dm_stat: The Diebold-Mariano statistic.
        p_value: The two-tailed p-value under H0: both models have equal accuracy.
    """
    n = len(y_true)
    
    # Calculate Brier score for each model
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    
    for t in range(n):
        true_cls = int(y_true[t])
        
        # Guard against index out of bounds
        p1_cls = proba_1[t, true_cls] if true_cls < proba_1.shape[1] else 0.0
        p2_cls = proba_2[t, true_cls] if true_cls < proba_2.shape[1] else 0.0
        
        e1[t] = (1.0 - p1_cls) ** 2
        e2[t] = (1.0 - p2_cls) ** 2
        
    # Loss differential
    d = e1 - e2
    mean_d = np.mean(d)
    
    # Variance of the loss differential with standard autocovariance check (lag=1)
    if n > 1:
        gamma_0 = np.var(d)
        # Covariance at lag 1
        d_dem = d - mean_d
        gamma_1 = np.mean(d_dem[1:] * d_dem[:-1]) if len(d_dem) > 1 else 0.0
        # Variance estimate for DM (HAC type variance)
        var_d = (gamma_0 + 2.0 * gamma_1) / n
    else:
        var_d = 1.0
        
    if var_d <= 0.0:
        var_d = 1e-8
        
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    
    return float(dm_stat), float(p_value)

def mcnemars_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray
) -> Tuple[float, float]:
    """
    Performs McNemar's test comparing classification correctness of two models.
    
    Contingency table:
                     Model 2 Correct    Model 2 Incorrect
    Model 1 Correct       a                   b
    Model 1 Incorrect     c                   d
    
    Returns:
        chi2_stat: Chi-squared statistic.
        p_value: Two-tailed p-value.
    """
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)
    
    b = np.sum(correct_1 & ~correct_2)  # Model 1 correct, Model 2 incorrect
    c = np.sum(~correct_1 & correct_2)  # Model 1 incorrect, Model 2 correct
    
    if b + c == 0:
        return 0.0, 1.0
        
    # McNemar's test statistic with continuity correction
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_value = stats.chi2.sf(chi2_stat, df=1)
    
    return float(chi2_stat), float(p_value)
