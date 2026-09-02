import pandas as pd
import numpy as np
from typing import Dict, Any, List

class PerformanceReporter:
    """
    Generates comparative tabular and analytical performance summaries across portfolio strategies.
    """
    @staticmethod
    def generate_strategy_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Creates a clean multi-strategy comparative DataFrame of key risk-adjusted metrics.
        results_dict: {strategy_name: backtest_result}
        """
        rows = []
        for name, res in results_dict.items():
            s = res["summary"]
            rows.append({
                "Strategy": name.replace("_", " ").title(),
                "Gross CAGR (%)": round(s["cagr_gross"] * 100, 2),
                "Net CAGR (%)": round(s["cagr_net"] * 100, 2),
                "Net Volatility (%)": round(s["volatility_net"] * 100, 2),
                "Gross Sharpe": round(s["sharpe_gross"], 2),
                "Net Sharpe": round(s["sharpe_net"], 2),
                "Net Sortino": round(s["sortino_net"], 2),
                "Max Drawdown (%)": round(s["max_drawdown_net"] * 100, 2),
                "Historical VaR 95 (%)": round(s["var_95_net"] * 100, 2),
                "CVaR 95 (%)": round(s["cvar_95_net"] * 100, 2),
                "Annual Turnover (%)": round(s["annualized_turnover"] * 100, 2),
                "Total Costs ($)": round(s["total_transaction_costs"], 2)
            })
        df = pd.DataFrame(rows).set_index("Strategy")
        return df

    @staticmethod
    def generate_cost_drag_analysis(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyzes the return drag caused by turnover and transaction fees.
        """
        rows = []
        for name, res in results_dict.items():
            s = res["summary"]
            cagr_drag_bps = (s["cagr_gross"] - s["cagr_net"]) * 10000.0
            sharpe_drag = s["sharpe_gross"] - s["sharpe_net"]
            rows.append({
                "Strategy": name.replace("_", " ").title(),
                "Annual Turnover (%)": round(s["annualized_turnover"] * 100, 2),
                "Total Costs Paid ($)": round(s["total_transaction_costs"], 2),
                "Return Drag (bps)": round(cagr_drag_bps, 1),
                "Sharpe Reduction": round(sharpe_drag, 3)
            })
        return pd.DataFrame(rows).set_index("Strategy")
