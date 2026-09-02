import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional
from src.optimization.base import BaseOptimizer, OptimizationResult

class RiskParityOptimizer(BaseOptimizer):
    """
    Equal Risk Contribution (ERC) / Risk Parity Portfolio Strategy.
    Optimizes portfolio weights such that every asset contributes equally to total portfolio risk:
    TRC_i = (w_i * (Sigma w)_i) / sigma_p = sigma_p / N.
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

        # Target percentage risk contribution for equal risk parity: 1 / N
        target_prc = 1.0 / n

        # Objective: minimize sum of squared deviations from target risk contribution
        def objective(w):
            port_var = float(w.T @ cov_matrix @ w)
            if port_var <= 1e-12:
                return 1e6
            port_vol = np.sqrt(port_var)
            
            # Marginal & Total risk contributions
            mrc = (cov_matrix @ w) / port_vol
            trc = w * mrc
            prc = trc / port_vol
            
            # Sum of squared risk contribution errors
            return float(np.sum((prc - target_prc) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        constraints.extend(self.get_sector_constraints(symbols, sector_mapping))

        # Asset bounds: for risk parity, each asset must have strictly positive weight (min_weight >= 1e-4)
        lower_bound = max(self.min_weight, 1e-4) if self.long_only else -1.0
        bounds = [(lower_bound, self.max_weight) for _ in range(n)]

        # Initial guess: inverse volatility weighting (good starting point for risk parity)
        asset_vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / np.maximum(asset_vols, 1e-4)
        init_weights = inv_vols / np.sum(inv_vols)

        res = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12}
        )

        if not res.success:
            # Fallback to inverse volatility weighting if convergence fails
            weights = init_weights
            converged = False
            msg = f"Risk parity solver did not fully converge ({res.message}). Fallback to inverse-volatility weights."
        else:
            weights = res.x
            weights = np.clip(weights, lower_bound, 1.0)
            weights = weights / np.sum(weights)
            converged = True
            msg = "Risk parity (Equal Risk Contribution) optimization converged successfully."

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
