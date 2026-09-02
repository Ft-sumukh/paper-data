from src.optimization.base import BaseOptimizer, OptimizationResult
from src.optimization.equal_weight import EqualWeightOptimizer
from src.optimization.min_variance import MinimumVarianceOptimizer
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer

def get_optimizer(
    strategy_name: str,
    risk_free_rate: float = 0.02,
    long_only: bool = True,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    max_sector_weight: float = None,
    **kwargs
) -> BaseOptimizer:
    """
    Factory function to instantiate portfolio optimizers by strategy name.
    """
    name = strategy_name.lower().strip()
    if name in ["equal_weight", "1/n", "ew"]:
        return EqualWeightOptimizer(
            risk_free_rate=risk_free_rate,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            max_sector_weight=max_sector_weight
        )
    elif name in ["min_variance", "minimum_variance", "gmv"]:
        return MinimumVarianceOptimizer(
            risk_free_rate=risk_free_rate,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            max_sector_weight=max_sector_weight
        )
    elif name in ["mean_variance", "mvo", "max_sharpe"]:
        return MeanVarianceOptimizer(
            risk_aversion=kwargs.get("risk_aversion", 2.5),
            target_metric=kwargs.get("target_metric", "sharpe"),
            risk_free_rate=risk_free_rate,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            max_sector_weight=max_sector_weight
        )
    elif name in ["risk_parity", "erc", "equal_risk_contribution"]:
        return RiskParityOptimizer(
            risk_free_rate=risk_free_rate,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            max_sector_weight=max_sector_weight
        )
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy_name}")

__all__ = [
    "BaseOptimizer", "OptimizationResult", "EqualWeightOptimizer",
    "MinimumVarianceOptimizer", "MeanVarianceOptimizer", "RiskParityOptimizer",
    "get_optimizer"
]
