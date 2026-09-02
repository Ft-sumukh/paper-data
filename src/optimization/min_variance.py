import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional
from src.optimization.base import BaseOptimizer, OptimizationResult

class MinimumVarianceOptimizer(BaseOptimizer):
    """
    Global Minimum Variance (GMV) Portfolio Strategy.
    Minimizes total portfolio variance w^T Sigma w subject to budget and sector constraints.
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

        # Objective function: portfolio variance w^T Sigma w
        def objective(w):
            return float(w.T @ cov_matrix @ w)

        def objective_gradient(w):
            return 2.0 * (cov_matrix @ w)

        # Constraints: sum(w) = 1.0
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        
        # Sector constraints
        constraints.extend(self.get_sector_constraints(symbols, sector_mapping))

        # Asset bounds
        bounds = self.get_bounds(n)

        # Initial guess: equal weights
        init_weights = np.full(n, 1.0 / n)

        res = minimize(
            objective,
            init_weights,
            method="SLSQP",
            jac=objective_gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9}
        )

        if not res.success:
            # Fallback to equal weight if solver fails
            weights = np.full(n, 1.0 / n)
            converged = False
            msg = f"Optimization failed ({res.message}). Fallback to 1/N."
        else:
            weights = res.x
            # Clean tiny numerical noise and renormalize
            weights = np.clip(weights, 0.0 if self.long_only else -1.0, 1.0)
            weights = weights / np.sum(weights)
            converged = True
            msg = "Minimum variance optimization converged successfully."

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
            converged=converged,
            message=msg
        )
