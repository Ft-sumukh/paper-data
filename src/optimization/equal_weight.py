import numpy as np
from typing import List, Dict, Optional
from src.optimization.base import BaseOptimizer, OptimizationResult

class EqualWeightOptimizer(BaseOptimizer):
    """
    Equal-Weight Portfolio Strategy (1/N benchmark).
    Sets w_i = 1 / N for all N assets.
    """
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        symbols: List[str],
        sector_mapping: Optional[Dict[str, str]] = None,
        current_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        n = len(symbols)
        if n == 0:
            raise ValueError("Asset list cannot be empty.")
            
        weights = np.full(n, 1.0 / n)
        port_return = float(weights @ expected_returns)
        
        risk_decomp = self.compute_risk_contributions(weights, cov_matrix)
        port_vol = risk_decomp["port_vol"]
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 1e-8 else 0.0

        symbol_weights = dict(zip(symbols, [float(w) for w in weights]))

        return OptimizationResult(
            weights=weights,
            symbol_weights=symbol_weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            marginal_risk_contributions=risk_decomp["mrc"],
            total_risk_contributions=risk_decomp["trc"],
            percentage_risk_contributions=risk_decomp["prc"],
            converged=True,
            message="Equal-weight allocation computed analytically."
        )
