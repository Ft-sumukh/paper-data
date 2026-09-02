import numpy as np
import pandas as pd
from typing import Tuple, Dict

class TransactionCostModel:
    """
    Models turnover, exchange trading fees, and market impact/slippage.
    """
    def __init__(self, transaction_cost_bps: float = 10.0, slippage_bps: float = 5.0):
        # Basis points to decimal (1 bp = 0.0001 = 0.01%)
        self.fee_rate = transaction_cost_bps * 1e-4
        self.slippage_rate = slippage_bps * 1e-4
        self.total_rate = self.fee_rate + self.slippage_rate

    def calculate_turnover(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray
    ) -> float:
        """
        Computes 1-way turnover percentage: 0.5 * sum(|w_target - w_current|).
        For total volume traded: sum(|w_target - w_current|).
        """
        if current_weights is None or len(current_weights) != len(target_weights):
            return float(np.sum(np.abs(target_weights))) # Initial purchase
        return float(np.sum(np.abs(target_weights - current_weights)))

    def calculate_cost(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculates trading costs in dollars and as a fraction of portfolio equity.
        """
        turnover = self.calculate_turnover(target_weights, current_weights)
        cost_fraction = turnover * self.total_rate
        cost_dollars = portfolio_value * cost_fraction
        
        return {
            "turnover": turnover,
            "cost_fraction": cost_fraction,
            "cost_dollars": cost_dollars,
            "fee_dollars": portfolio_value * (turnover * self.fee_rate),
            "slippage_dollars": portfolio_value * (turnover * self.slippage_rate)
        }

    def compute_drifted_weights(
        self,
        weights: np.ndarray,
        asset_returns: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the drifted weights after asset price moves:
        w_i^+ = w_i * (1 + R_i) / sum(w_j * (1 + R_j)).
        """
        gross_growth = weights * (1.0 + asset_returns)
        total_growth = np.sum(gross_growth)
        if total_growth <= 1e-8:
            return weights
        return gross_growth / total_growth
