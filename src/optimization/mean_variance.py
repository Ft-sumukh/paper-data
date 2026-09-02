import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional
from src.optimization.base import BaseOptimizer, OptimizationResult

class MeanVarianceOptimizer(BaseOptimizer):
    """
    Markowitz Mean-Variance Optimization Strategy.
    Supports both Quadratic Utility (with configurable risk aversion lambda)
    and Maximum Sharpe Ratio optimization.
    """
    def __init__(
        self,
        risk_aversion: float = 2.5,
        target_metric: str = "sharpe", # "sharpe" or "quadratic_utility"
        risk_free_rate: float = 0.02,
        long_only: bool = True,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        max_sector_weight: Optional[float] = None
    ):
        super().__init__(
            risk_free_rate=risk_free_rate,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            max_sector_weight=max_sector_weight
        )
        self.risk_aversion = risk_aversion
        self.target_metric = target_metric

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

        if self.target_metric == "sharpe":
            # Maximize Sharpe ratio <=> Minimize negative Sharpe
            def objective(w):
                p_ret = float(w @ expected_returns)
                p_vol = np.sqrt(max(float(w.T @ cov_matrix @ w), 1e-8))
                return -float((p_ret - self.risk_free_rate) / p_vol)
        else:
            # Quadratic utility: Minimize 0.5 * lambda * w^T Sigma w - w^T mu
            def objective(w):
                var = float(w.T @ cov_matrix @ w)
                ret = float(w @ expected_returns)
                return 0.5 * self.risk_aversion * var - ret

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        constraints.extend(self.get_sector_constraints(symbols, sector_mapping))
        bounds = self.get_bounds(n)
        init_weights = np.full(n, 1.0 / n)

        res = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9}
        )

        if not res.success:
            weights = np.full(n, 1.0 / n)
            converged = False
            msg = f"Mean-variance optimization failed ({res.message}). Fallback to 1/N."
        else:
            weights = res.x
            weights = np.clip(weights, 0.0 if self.long_only else -1.0, 1.0)
            weights = weights / np.sum(weights)
            converged = True
            msg = f"Mean-variance ({self.target_metric}) optimization converged."

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
