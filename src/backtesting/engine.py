import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.features.returns import calculate_simple_returns, calculate_covariance_matrix
from src.optimization import get_optimizer, BaseOptimizer
from src.backtesting.transaction_cost import TransactionCostModel
from src.risk.metrics import calculate_all_risk_metrics

logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    """
    Walk-forward backtesting framework supporting rolling/expanding estimation windows,
    configurable rebalance schedules, turnover tracking, transaction costs, and slippage.
    """
    def __init__(
        self,
        strategy_name: str,
        prices: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        benchmark_symbol: str = "SPY",
        estimation_window: int = 252,
        rebalance_frequency: str = "monthly", # "monthly" (21 days), "quarterly" (63 days), "weekly" (5 days)
        window_type: str = "rolling",         # "rolling" or "expanding"
        initial_capital: float = 1000000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        risk_free_rate: float = 0.02,
        sector_mapping: Optional[Dict[str, str]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.strategy_name = strategy_name
        self.prices = prices.copy().sort_index()
        self.benchmark_symbol = benchmark_symbol
        
        # Universe symbols (excluding benchmark if present in prices)
        if symbols:
            self.symbols = [s for s in symbols if s in self.prices.columns and s != benchmark_symbol]
        else:
            self.symbols = [c for c in self.prices.columns if c != benchmark_symbol]
            
        self.estimation_window = estimation_window
        self.rebalance_frequency = rebalance_frequency
        self.window_type = window_type
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.sector_mapping = sector_mapping or {}
        
        self.cost_model = TransactionCostModel(
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps
        )
        
        opt_kw = optimizer_kwargs or {}
        self.optimizer: BaseOptimizer = get_optimizer(
            strategy_name=strategy_name,
            risk_free_rate=risk_free_rate,
            **opt_kw
        )

    def _get_rebalance_step(self) -> int:
        if self.rebalance_frequency == "weekly":
            return 5
        elif self.rebalance_frequency == "quarterly":
            return 63
        else: # monthly default
            return 21

    def run(self) -> Dict[str, Any]:
        """
        Executes the out-of-sample walk-forward backtest.
        Returns daily returns, portfolio values, turnover, weights history, and risk metrics.
        """
        asset_prices = self.prices[self.symbols]
        asset_returns = calculate_simple_returns(asset_prices)
        
        n_days = len(asset_returns)
        if n_days <= self.estimation_window:
            raise ValueError(f"Total trading days ({n_days}) must exceed estimation window ({self.estimation_window}).")

        step = self._get_rebalance_step()
        rebalance_indices = list(range(self.estimation_window, n_days, step))
        if rebalance_indices[-1] != n_days:
            rebalance_indices.append(n_days)

        # Result collectors
        dates = asset_returns.index[self.estimation_window:]
        daily_gross_returns = []
        daily_net_returns = []
        daily_turnover = []
        daily_costs = []
        daily_portfolio_value = []
        weights_records = []

        current_capital = self.initial_capital
        current_weights = None
        target_weights = None

        logger.info(f"Running {self.strategy_name} walk-forward backtest across {len(dates)} out-of-sample days...")

        for idx in range(len(rebalance_indices) - 1):
            start_idx = rebalance_indices[idx]
            end_idx = rebalance_indices[idx + 1]
            rebalance_date = asset_returns.index[start_idx]

            # In-sample estimation slice (Strictly prior to rebalance date -> Zero lookahead)
            if self.window_type == "expanding":
                in_sample_rets = asset_returns.iloc[:start_idx]
            else:
                in_sample_rets = asset_returns.iloc[start_idx - self.estimation_window : start_idx]

            # Estimate parameters
            expected_rets = in_sample_rets.mean().values * 252.0
            cov_matrix = calculate_covariance_matrix(in_sample_rets, annualize=True)

            # Optimize portfolio
            opt_result = self.optimizer.optimize(
                expected_returns=expected_rets,
                cov_matrix=cov_matrix,
                symbols=self.symbols,
                sector_mapping=self.sector_mapping,
                current_weights=current_weights
            )
            target_weights = opt_result.weights

            # Forward testing loop until next rebalance
            for day_idx in range(start_idx, end_idx):
                curr_date = asset_returns.index[day_idx]
                day_asset_rets = asset_returns.iloc[day_idx].values

                if day_idx == start_idx:
                    # Rebalance executed at market open
                    cost_info = self.cost_model.calculate_cost(
                        target_weights=target_weights,
                        current_weights=current_weights,
                        portfolio_value=current_capital
                    )
                    turnover = cost_info["turnover"]
                    cost_dollars = cost_info["cost_dollars"]
                    active_weights = target_weights
                else:
                    turnover = 0.0
                    cost_dollars = 0.0
                    active_weights = current_weights

                # Daily return
                gross_ret = float(np.sum(active_weights * day_asset_rets))
                cost_fraction = cost_dollars / current_capital if current_capital > 0 else 0.0
                net_ret = gross_ret - cost_fraction

                # Update portfolio equity
                current_capital = current_capital * (1.0 + net_ret)

                # Record metrics
                daily_gross_returns.append(gross_ret)
                daily_net_returns.append(net_ret)
                daily_turnover.append(turnover)
                daily_costs.append(cost_dollars)
                daily_portfolio_value.append(current_capital)
                
                # Record weights
                weight_dict = dict(zip(self.symbols, active_weights))
                weight_dict["date"] = curr_date
                weights_records.append(weight_dict)

                # Weight drift at day end
                current_weights = self.cost_model.compute_drifted_weights(active_weights, day_asset_rets)

        # Build DataFrames
        perf_df = pd.DataFrame({
            "gross_return": daily_gross_returns,
            "net_return": daily_net_returns,
            "turnover": daily_turnover,
            "transaction_cost": daily_costs,
            "portfolio_value": daily_portfolio_value
        }, index=dates)

        weights_df = pd.DataFrame(weights_records).set_index("date")

        # Benchmark returns alignment
        bmk_returns = None
        if self.benchmark_symbol in self.prices.columns:
            bmk_prices = self.prices[self.benchmark_symbol]
            bmk_all_rets = calculate_simple_returns(bmk_prices)
            bmk_returns = bmk_all_rets.reindex(dates).dropna()

        # Risk metrics (Gross & Net)
        risk_metrics_net = calculate_all_risk_metrics(
            returns=perf_df["net_return"],
            benchmark_returns=bmk_returns,
            risk_free_rate=self.risk_free_rate
        )
        risk_metrics_gross = calculate_all_risk_metrics(
            returns=perf_df["gross_return"],
            benchmark_returns=bmk_returns,
            risk_free_rate=self.risk_free_rate
        )

        summary_metrics = {
            "strategy": self.strategy_name,
            "start_date": str(dates[0].date()),
            "end_date": str(dates[-1].date()),
            "total_trading_days": len(dates),
            "initial_capital": self.initial_capital,
            "final_value": float(perf_df["portfolio_value"].iloc[-1]),
            "total_turnover": float(perf_df["turnover"].sum()),
            "annualized_turnover": float(perf_df["turnover"].mean() * 252.0),
            "total_transaction_costs": float(perf_df["transaction_cost"].sum()),
            "cagr_gross": risk_metrics_gross["cagr"],
            "cagr_net": risk_metrics_net["cagr"],
            "volatility_net": risk_metrics_net["annualized_volatility"],
            "sharpe_gross": risk_metrics_gross["sharpe_ratio"],
            "sharpe_net": risk_metrics_net["sharpe_ratio"],
            "sortino_net": risk_metrics_net["sortino_ratio"],
            "max_drawdown_net": risk_metrics_net["max_drawdown"],
            "var_95_net": risk_metrics_net["var_95_historical"],
            "cvar_95_net": risk_metrics_net["cvar_95"],
            "beta": risk_metrics_net.get("beta", 1.0),
            "tracking_error": risk_metrics_net.get("tracking_error", 0.0),
            "information_ratio": risk_metrics_net.get("information_ratio", 0.0)
        }

        return {
            "summary": summary_metrics,
            "performance_df": perf_df,
            "weights_df": weights_df,
            "risk_metrics_net": risk_metrics_net,
            "risk_metrics_gross": risk_metrics_gross,
            "benchmark_returns": bmk_returns
        }
