import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, Optional

def calculate_cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Computes Compound Annual Growth Rate (CAGR).
    """
    if len(returns) == 0:
        return 0.0
    cum_ret = (1.0 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0 or cum_ret <= 0:
        return float(returns.mean() * periods_per_year)
    return float(cum_ret ** (1.0 / n_years) - 1.0)

def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Computes annualized standard deviation of returns.
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))

def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02, 
    periods_per_year: int = 252
) -> float:
    """
    Computes annualized Sharpe ratio: (CAGR - Rf) / Volatility.
    """
    cagr = calculate_cagr(returns, periods_per_year)
    vol = calculate_annualized_volatility(returns, periods_per_year)
    if vol <= 1e-8:
        return 0.0
    return float((cagr - risk_free_rate) / vol)

def calculate_downside_deviation(
    returns: pd.Series, 
    target_return: float = 0.0, 
    periods_per_year: int = 252
) -> float:
    """
    Computes annualized downside semi-deviation below target_return.
    """
    downside_diff = returns - (target_return / periods_per_year)
    downside_diff = np.minimum(downside_diff, 0.0)
    semi_variance = np.mean(downside_diff ** 2)
    return float(np.sqrt(semi_variance) * np.sqrt(periods_per_year))

def calculate_sortino_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02, 
    periods_per_year: int = 252
) -> float:
    """
    Computes annualized Sortino ratio: (CAGR - Rf) / Downside Deviation.
    """
    cagr = calculate_cagr(returns, periods_per_year)
    downside_vol = calculate_downside_deviation(returns, risk_free_rate, periods_per_year)
    if downside_vol <= 1e-8:
        return 0.0
    return float((cagr - risk_free_rate) / downside_vol)

def calculate_drawdown_series(returns: pd.Series) -> pd.DataFrame:
    """
    Computes historical wealth index, previous peaks, and drawdown percentages.
    """
    wealth_index = (1.0 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "wealth_index": wealth_index,
        "previous_peaks": previous_peaks,
        "drawdown": drawdowns
    }, index=returns.index)

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Computes maximum historical peak-to-trough drawdown percentage.
    """
    if len(returns) == 0:
        return 0.0
    dd_df = calculate_drawdown_series(returns)
    return float(abs(dd_df["drawdown"].min()))

def calculate_historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Computes non-parametric Historical Value at Risk at (1 - alpha) confidence.
    Expressed as a positive loss percentage: -Percentile(returns, alpha * 100).
    """
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, alpha * 100.0))

def calculate_parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Computes Parametric Gaussian VaR: -(mu + z_alpha * sigma).
    """
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = stats.norm.ppf(alpha)
    return float(-(mu + z * sigma))

def calculate_cornish_fisher_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Computes modified Value at Risk adjusting for skewness and kurtosis via Cornish-Fisher expansion.
    """
    if len(returns) < 4:
        return calculate_parametric_var(returns, alpha)
        
    z = stats.norm.ppf(alpha)
    s = stats.skew(returns)
    k = stats.kurtosis(returns) # excess kurtosis
    
    # Cornish-Fisher expansion quantile approximation
    z_tilde = (
        z +
        (z**2 - 1) * s / 6.0 +
        (z**3 - 3*z) * k / 24.0 -
        (2*z**3 - 5*z) * (s**2) / 36.0
    )
    
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    return float(-(mu + z_tilde * sigma))

def calculate_expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Computes Conditional Value at Risk (Expected Shortfall): -E[R | R <= -VaR].
    """
    if len(returns) == 0:
        return 0.0
    var_threshold = -calculate_historical_var(returns, alpha)
    tail_losses = returns[returns <= var_threshold]
    if len(tail_losses) == 0:
        return calculate_historical_var(returns, alpha)
    return float(-tail_losses.mean())

def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Computes CAPM Beta relative to a benchmark: Cov(R_i, R_m) / Var(R_m).
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 5:
        return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var_m = np.var(aligned.iloc[:, 1], ddof=1)
    if var_m <= 1e-8:
        return 1.0
    return float(cov / var_m)

def calculate_tracking_error(
    returns: pd.Series, 
    benchmark_returns: pd.Series, 
    periods_per_year: int = 252
) -> float:
    """
    Computes annualized tracking error (standard deviation of excess return).
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(diff.std(ddof=1) * np.sqrt(periods_per_year))

def calculate_information_ratio(
    returns: pd.Series, 
    benchmark_returns: pd.Series, 
    periods_per_year: int = 252
) -> float:
    """
    Computes Information Ratio: (Annualized Excess Return) / Tracking Error.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    excess_mean = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).mean() * periods_per_year
    te = calculate_tracking_error(returns, benchmark_returns, periods_per_year)
    if te <= 1e-8:
        return 0.0
    return float(excess_mean / te)

def calculate_all_risk_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    alpha: float = 0.05,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Generates a comprehensive summary dictionary of all portfolio risk and performance metrics.
    """
    metrics = {
        "cagr": calculate_cagr(returns, periods_per_year),
        "annualized_volatility": calculate_annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": calculate_max_drawdown(returns),
        "var_95_historical": calculate_historical_var(returns, alpha),
        "var_95_parametric": calculate_parametric_var(returns, alpha),
        "var_95_cornish_fisher": calculate_cornish_fisher_var(returns, alpha),
        "cvar_95": calculate_expected_shortfall(returns, alpha),
        "skewness": float(stats.skew(returns)) if len(returns) >= 3 else 0.0,
        "excess_kurtosis": float(stats.kurtosis(returns)) if len(returns) >= 4 else 0.0
    }

    if benchmark_returns is not None:
        metrics["beta"] = calculate_beta(returns, benchmark_returns)
        metrics["tracking_error"] = calculate_tracking_error(returns, benchmark_returns, periods_per_year)
        metrics["information_ratio"] = calculate_information_ratio(returns, benchmark_returns, periods_per_year)

    return metrics
