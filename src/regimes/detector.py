import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class RollingVolatilityRegimeDetector:
    """
    Classifies market conditions into Low, Normal, and High volatility regimes
    based on rolling realized benchmark volatility quantiles.
    """
    def __init__(
        self,
        window: int = 63,
        quantiles: Tuple[float, float] = (0.333, 0.667),
        periods_per_year: int = 252
    ):
        self.window = window
        self.quantiles = quantiles
        self.periods_per_year = periods_per_year

    def fit_predict(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Computes rolling realized volatility and assigns regime labels.
        """
        rolling_vol = benchmark_returns.rolling(window=self.window).std() * np.sqrt(self.periods_per_year)
        rolling_vol = rolling_vol.dropna()

        # Compute quantile cutoffs
        q_low, q_high = rolling_vol.quantile(self.quantiles[0]), rolling_vol.quantile(self.quantiles[1])

        # Assign regimes
        regime_labels = []
        regime_codes = []

        for vol in rolling_vol:
            if vol < q_low:
                regime_labels.append("LOW_VOL")
                regime_codes.append(0)
            elif vol < q_high:
                regime_labels.append("NORMAL_VOL")
                regime_codes.append(1)
            else:
                regime_labels.append("HIGH_VOL")
                regime_codes.append(2)

        df = pd.DataFrame({
            "realized_volatility": rolling_vol,
            "regime": regime_labels,
            "regime_code": regime_codes
        }, index=rolling_vol.index)

        return df
