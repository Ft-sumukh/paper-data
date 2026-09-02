import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.risk.metrics import (
    calculate_cagr, calculate_annualized_volatility, 
    calculate_sharpe_ratio, calculate_max_drawdown
)

class RegimePerformanceAnalyzer:
    """
    Computes performance and risk metrics conditioned on market regimes (Low, Normal, High Volatility).
    """
    @staticmethod
    def analyze_regime_performance(
        strategy_returns: pd.Series,
        regime_series: pd.Series,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """
        Segments returns by regime and computes annualized CAGR, Volatility, Sharpe, and Max DD.
        """
        aligned = pd.concat([strategy_returns, regime_series], axis=1).dropna()
        aligned.columns = ["return", "regime"]

        regimes = ["ALL", "LOW_VOL", "NORMAL_VOL", "HIGH_VOL"]
        results = []

        for reg in regimes:
            if reg == "ALL":
                sub_rets = aligned["return"]
            else:
                sub_rets = aligned[aligned["regime"] == reg]["return"]

            if len(sub_rets) > 10:
                cagr = calculate_cagr(sub_rets)
                vol = calculate_annualized_volatility(sub_rets)
                sharpe = calculate_sharpe_ratio(sub_rets, risk_free_rate=risk_free_rate)
                mdd = calculate_max_drawdown(sub_rets)
                obs = len(sub_rets)
            else:
                cagr, vol, sharpe, mdd, obs = 0.0, 0.0, 0.0, 0.0, len(sub_rets)

            results.append({
                "Regime": reg,
                "Observations": obs,
                "CAGR (%)": round(cagr * 100, 2),
                "Volatility (%)": round(vol * 100, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "Max Drawdown (%)": round(mdd * 100, 2)
            })

        return pd.DataFrame(results).set_index("Regime")
