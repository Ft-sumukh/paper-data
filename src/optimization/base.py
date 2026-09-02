from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

@dataclass
class OptimizationResult:
    weights: np.ndarray
    symbol_weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    marginal_risk_contributions: np.ndarray
    total_risk_contributions: np.ndarray
    percentage_risk_contributions: np.ndarray
    converged: bool
    message: str

class BaseOptimizer(ABC):
    """
    Abstract base class for all portfolio construction and optimization strategies.
    """
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        long_only: bool = True,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        max_sector_weight: Optional[float] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.long_only = long_only
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_sector_weight = max_sector_weight

    @abstractmethod
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: List[str],
        sector_mapping: Optional[Dict[str, str]] = None,
        current_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Executes optimization and returns OptimizationResult.
        """
        pass

    def compute_risk_contributions(self, weights: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates Marginal Risk Contribution (MRC), Total Risk Contribution (TRC),
        and Percentage Risk Contribution (PRC).
        
        sigma_p = sqrt(w^T Sigma w)
        MRC_i = (Sigma w)_i / sigma_p
        TRC_i = w_i * MRC_i
        PRC_i = TRC_i / sigma_p = (w_i * (Sigma w)_i) / (sigma_p^2)
        """
        port_var = float(weights.T @ cov_matrix @ weights)
        port_vol = np.sqrt(max(port_var, 1e-8))
        
        marginal_risk = (cov_matrix @ weights) / port_vol
        total_risk = weights * marginal_risk
        pct_risk = total_risk / port_vol

        return {
            "port_vol": port_vol,
            "mrc": marginal_risk,
            "trc": total_risk,
            "prc": pct_risk
        }

    def get_bounds(self, n_assets: int) -> List[tuple]:
        """
        Builds asset-level weight bounds.
        """
        lower = self.min_weight if self.long_only else -1.0
        upper = self.max_weight
        return [(lower, upper) for _ in range(n_assets)]

    def get_sector_constraints(
        self, 
        symbols: List[str], 
        sector_mapping: Optional[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Constructs linear inequality constraints ensuring sector weights <= max_sector_weight.
        """
        constraints = []
        if self.max_sector_weight is None or not sector_mapping:
            return constraints

        sectors = set(sector_mapping.get(s, "Other") for s in symbols)
        for sec in sectors:
            indices = [i for i, s in enumerate(symbols) if sector_mapping.get(s) == sec]
            if indices:
                # Constraint: max_sector_weight - sum(w_i for i in sector) >= 0
                def make_sector_constraint(sec_indices):
                    return lambda w: self.max_sector_weight - sum(w[i] for i in sec_indices)
                constraints.append({
                    "type": "ineq",
                    "fun": make_sector_constraint(indices)
                })
        return constraints
