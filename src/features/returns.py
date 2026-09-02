import numpy as np
import pandas as pd
from typing import Tuple, Optional

def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes percentage daily arithmetic returns: R_t = (P_t - P_{t-1}) / P_{t-1}.
    """
    return prices.pct_change().dropna()

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily logarithmic returns: r_t = ln(P_t / P_{t-1}).
    """
    return np.log(prices / prices.shift(1)).dropna()

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Computes the compound cumulative return series starting from 0.0: (1 + R).cumprod() - 1.
    """
    return (1.0 + returns).cumprod() - 1.0

def calculate_wealth_index(returns: pd.Series, initial_capital: float = 1000000.0) -> pd.Series:
    """
    Computes portfolio dollar equity curve over time: initial_capital * (1 + R).cumprod().
    """
    return initial_capital * (1.0 + returns).cumprod()

def calculate_rolling_volatility(
    returns: pd.DataFrame, 
    window: int = 63, 
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Computes rolling annualized sample standard deviation.
    """
    return returns.rolling(window=window).std() * np.sqrt(periods_per_year)

def calculate_covariance_matrix(
    returns: pd.DataFrame, 
    annualize: bool = True, 
    periods_per_year: int = 252
) -> np.ndarray:
    """
    Computes empirical sample covariance matrix (annualized by default).
    """
    cov = returns.cov().values
    if annualize:
        cov = cov * periods_per_year
    return cov

def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson cross-asset correlation matrix.
    """
    return returns.corr()
