import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.connection import init_db, SessionLocal
from src.database.crud import get_price_matrix, save_backtest_run, get_all_assets
from src.features.returns import calculate_simple_returns, calculate_correlation_matrix, calculate_covariance_matrix
from src.risk.metrics import calculate_drawdown_series
from src.backtesting.engine import WalkForwardBacktester
from src.backtesting.performance import PerformanceReporter
from src.regimes.detector import RollingVolatilityRegimeDetector
from src.regimes.segmentation import RegimePerformanceAnalyzer
from src.statistics.hypothesis import ResearchHypothesisTester
from src.statistics.bootstrap import StationaryBlockBootstrap
from src.visualization import (
    plot_cumulative_equity_curves,
    plot_rolling_volatility,
    plot_drawdown_curves,
    plot_correlation_matrix,
    plot_risk_return_scatter,
    plot_weights_evolution,
    plot_risk_contributions,
    plot_sharpe_and_sortino,
    plot_turnover_and_costs,
    plot_fee_sensitivity,
    plot_regime_timeline,
    plot_regime_sharpe_comparison,
    plot_h1_diversification,
    plot_bootstrap_sharpe_distribution,
    plot_tail_risk_comparison
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def run_research_experiments(config_path: str = "configs/default.yaml"):
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    figures_dir = config["paths"]["figures_dir"]
    tables_dir = config["paths"]["tables_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize DB & Fetch Data
    init_db()
    db = SessionLocal()
    try:
        assets = get_all_assets(db)
        symbols = [a["symbol"] for a in config["data"]["assets"]]
        benchmark = config["data"]["benchmark"]
        prices_df = get_price_matrix(db, symbols=symbols + [benchmark])
    finally:
        db.close()

    if prices_df.empty:
        # Fallback to reading from processed CSV directly
        csv_path = os.path.join(config["paths"]["processed_data_dir"], "prices_clean.csv")
        prices_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    asset_symbols = [s for s in symbols if s in prices_df.columns and s != benchmark]
    sector_mapping = {a["symbol"]: a["sector"] for a in config["data"]["assets"]}

    logger.info(f"Running experiments across {len(asset_symbols)} assets and benchmark {benchmark}...")

    # --------------------------------------------------------------------------
    # 1. Walk-Forward Backtesting Across All 4 Strategies
    # --------------------------------------------------------------------------
    strategies = ["equal_weight", "min_variance", "mean_variance", "risk_parity"]
    backtest_results = {}

    for strat in strategies:
        logger.info(f"Executing Walk-Forward Backtest for: {strat}...")
        bt = WalkForwardBacktester(
            strategy_name=strat,
            prices=prices_df,
            symbols=asset_symbols,
            benchmark_symbol=benchmark,
            estimation_window=config["backtest"]["estimation_window"],
            rebalance_frequency=config["backtest"]["rebalance_frequency"],
            initial_capital=config["backtest"]["initial_capital"],
            transaction_cost_bps=config["costs"]["transaction_cost_bps"],
            slippage_bps=config["costs"]["slippage_bps"],
            risk_free_rate=config["risk_free_rate"],
            sector_mapping=sector_mapping
        )
        backtest_results[strat] = bt.run()

        # Persist to database
        db = SessionLocal()
        try:
            res = backtest_results[strat]
            p_df = res["performance_df"]
            save_backtest_run(
                db=db,
                name=f"Walk-Forward {strat.title()}",
                strategy=strat,
                start_date=p_df.index[0].date(),
                end_date=p_df.index[-1].date(),
                rebalance_freq=config["backtest"]["rebalance_frequency"],
                initial_capital=config["backtest"]["initial_capital"],
                transaction_cost_bps=config["costs"]["transaction_cost_bps"],
                slippage_bps=config["costs"]["slippage_bps"],
                weights_df=res["weights_df"],
                returns_df=p_df,
                risk_metrics_dict=res["risk_metrics_net"]
            )
        finally:
            db.close()

    # --------------------------------------------------------------------------
    # 2. Transaction Fee Sensitivity Sweep
    # --------------------------------------------------------------------------
    logger.info("Running Transaction Fee Sensitivity Sweep...")
    fee_levels = [0.0, 5.0, 10.0, 15.0, 25.0, 50.0]
    fee_sweep_cagrs = {s: [] for s in strategies}

    for fee in fee_levels:
        for strat in strategies:
            bt_sweep = WalkForwardBacktester(
                strategy_name=strat,
                prices=prices_df,
                symbols=asset_symbols,
                benchmark_symbol=benchmark,
                estimation_window=config["backtest"]["estimation_window"],
                rebalance_frequency=config["backtest"]["rebalance_frequency"],
                transaction_cost_bps=fee,
                slippage_bps=0.0,
                risk_free_rate=config["risk_free_rate"],
                sector_mapping=sector_mapping
            )
            res_sweep = bt_sweep.run()
            fee_sweep_cagrs[strat].append(res_sweep["summary"]["cagr_net"])

    # --------------------------------------------------------------------------
    # 3. Market Regime Segmentation
    # --------------------------------------------------------------------------
    logger.info("Executing Market Regime Detection & Performance Segmentation...")
    bmk_rets = calculate_simple_returns(prices_df[benchmark])
    detector = RollingVolatilityRegimeDetector(window=config["regimes"]["volatility_window"])
    regime_df = detector.fit_predict(bmk_rets)

    regime_performance_by_strat = {}
    for strat in strategies:
        strat_net_rets = backtest_results[strat]["performance_df"]["net_return"]
        regime_performance_by_strat[strat] = RegimePerformanceAnalyzer.analyze_regime_performance(
            strat_net_rets, regime_df["regime"], risk_free_rate=config["risk_free_rate"]
        )

    # --------------------------------------------------------------------------
    # 4. Formal Hypothesis Testing (H1, H2, H3, H4)
    # --------------------------------------------------------------------------
    logger.info("Executing Formal Hypothesis Testing Suite...")
    
    # H1: Diversification
    h1_results = ResearchHypothesisTester.test_h1_diversification(prices_df[asset_symbols], subsets_per_k=50)

    # H2: Risk Parity in High Vol
    h2_results = ResearchHypothesisTester.test_h2_risk_parity(
        rp_returns=backtest_results["risk_parity"]["performance_df"]["net_return"],
        ew_returns=backtest_results["equal_weight"]["performance_df"]["net_return"],
        regime_series=regime_df["regime"],
        risk_free_rate=config["risk_free_rate"]
    )

    # H3: Regimes Impact
    strat_returns_by_reg = {}
    for s in strategies:
        strat_returns_by_reg[s] = {}
        s_rets = backtest_results[s]["performance_df"]["net_return"]
        aligned = pd.concat([s_rets, regime_df["regime"]], axis=1).dropna()
        aligned.columns = ["ret", "reg"]
        for r in ["LOW_VOL", "NORMAL_VOL", "HIGH_VOL"]:
            strat_returns_by_reg[s][r] = aligned[aligned["reg"] == r]["ret"]
    h3_results = ResearchHypothesisTester.test_h3_market_regimes(strat_returns_by_reg)

    # H4: Optimization Edge (MVO vs 1/N)
    h4_results = ResearchHypothesisTester.test_h4_optimization_vs_benchmark(
        opt_net_returns=backtest_results["mean_variance"]["performance_df"]["net_return"],
        ew_net_returns=backtest_results["equal_weight"]["performance_df"]["net_return"],
        risk_free_rate=config["risk_free_rate"]
    )

    # Bootstrap distribution for Sharpe difference
    bootstrap = StationaryBlockBootstrap(n_bootstraps=1000)
    mvo_rets = backtest_results["mean_variance"]["performance_df"]["net_return"]
    ew_rets = backtest_results["equal_weight"]["performance_df"]["net_return"]
    aligned_be = pd.concat([mvo_rets, ew_rets], axis=1).dropna()
    resamples = bootstrap.generate_resamples(np.arange(len(aligned_be)))
    boot_diffs = []
    for b in range(1000):
        idx = resamples[b].astype(int)
        sr_m = (aligned_be.iloc[idx, 0].mean() * 252.0 - 0.02) / max(aligned_be.iloc[idx, 0].std() * np.sqrt(252.0), 1e-6)
        sr_e = (aligned_be.iloc[idx, 1].mean() * 252.0 - 0.02) / max(aligned_be.iloc[idx, 1].std() * np.sqrt(252.0), 1e-6)
        boot_diffs.append(sr_m - sr_e)
    boot_diffs = np.array(boot_diffs)

    # --------------------------------------------------------------------------
    # 5. Publication Tables Generation
    # --------------------------------------------------------------------------
    logger.info("Compiling LaTeX and Markdown Empirical Tables...")
    summary_table = PerformanceReporter.generate_strategy_comparison_table(backtest_results)
    cost_table = PerformanceReporter.generate_cost_drag_analysis(backtest_results)

    summary_table.to_csv(os.path.join(tables_dir, "table1_strategy_performance.csv"))
    summary_table.to_latex(os.path.join(tables_dir, "table1_strategy_performance.tex"))
    cost_table.to_csv(os.path.join(tables_dir, "table2_cost_drag.csv"))
    cost_table.to_latex(os.path.join(tables_dir, "table2_cost_drag.tex"))

    # --------------------------------------------------------------------------
    # 6. Generate 15 High-Resolution Publication Figures
    # --------------------------------------------------------------------------
    logger.info("Generating 15 Publication-Grade Visualizations in paper/figures/...")

    # Fig 1: Cumulative equity
    equity_dict = {s: backtest_results[s]["performance_df"]["portfolio_value"] for s in strategies}
    plot_cumulative_equity_curves(
        pd.DataFrame(equity_dict),
        backtest_results["equal_weight"]["benchmark_returns"],
        os.path.join(figures_dir, "fig1_cumulative_returns.png")
    )

    # Fig 2: Rolling volatility
    returns_dict = {s: backtest_results[s]["performance_df"]["net_return"] for s in strategies}
    plot_rolling_volatility(returns_dict, window=63, save_path=os.path.join(figures_dir, "fig2_rolling_volatility.png"))

    # Fig 3: Drawdowns
    dd_dict = {s: calculate_drawdown_series(backtest_results[s]["performance_df"]["net_return"])["drawdown"] for s in strategies}
    plot_drawdown_curves(dd_dict, os.path.join(figures_dir, "fig3_underwater_drawdowns.png"))

    # Fig 4: Correlation heatmap
    asset_rets = calculate_simple_returns(prices_df[asset_symbols])
    corr_df = calculate_correlation_matrix(asset_rets)
    plot_correlation_matrix(corr_df, os.path.join(figures_dir, "fig4_correlation_heatmap.png"))

    # Fig 5: Risk vs Return Scatter
    asset_vols = asset_rets.std() * np.sqrt(252.0)
    asset_means = asset_rets.mean() * 252.0
    strat_vols = {s: backtest_results[s]["summary"]["volatility_net"] for s in strategies}
    strat_rets = {s: backtest_results[s]["summary"]["cagr_net"] for s in strategies}
    plot_risk_return_scatter(asset_vols, asset_means, strat_vols, strat_rets, os.path.join(figures_dir, "fig5_risk_return_scatter.png"))

    # Fig 6: Portfolio weights evolution (Mean-Variance)
    plot_weights_evolution(backtest_results["mean_variance"]["weights_df"], "mean_variance", os.path.join(figures_dir, "fig6_portfolio_weights_evolution.png"))

    # Fig 7: Percentage Risk Contribution
    cov_full = calculate_covariance_matrix(asset_rets, annualize=True)
    rp_opt = backtest_results["risk_parity"]
    last_weights_rp = rp_opt["weights_df"].iloc[-1].values
    last_weights_ew = np.full(len(asset_symbols), 1.0 / len(asset_symbols))
    
    vol_rp = np.sqrt(last_weights_rp.T @ cov_full @ last_weights_rp)
    prc_rp = (last_weights_rp * (cov_full @ last_weights_rp) / vol_rp**2) * 100.0
    vol_ew = np.sqrt(last_weights_ew.T @ cov_full @ last_weights_ew)
    prc_ew = (last_weights_ew * (cov_full @ last_weights_ew) / vol_ew**2) * 100.0
    
    rc_df = pd.DataFrame({"Equal Weight (% TRC)": prc_ew, "Risk Parity (% TRC)": prc_rp}, index=asset_symbols)
    plot_risk_contributions(rc_df, os.path.join(figures_dir, "fig7_percentage_risk_contribution.png"))

    # Fig 8: Sharpe & Sortino Comparison
    plot_sharpe_and_sortino(summary_table, os.path.join(figures_dir, "fig8_annual_returns_bar.png"))

    # Fig 9: Turnover vs Costs
    plot_turnover_and_costs(cost_table, os.path.join(figures_dir, "fig9_sharpe_comparison.png"))

    # Fig 10: Fee sensitivity curve
    plot_fee_sensitivity(fee_levels, fee_sweep_cagrs, os.path.join(figures_dir, "fig10_cost_vs_turnover.png"))

    # Fig 11: Regime timeline
    plot_regime_timeline(regime_df, os.path.join(figures_dir, "fig11_fee_sensitivity_curve.png"))

    # Fig 12: Regime Sharpe comparison
    plot_regime_sharpe_comparison(regime_performance_by_strat, os.path.join(figures_dir, "fig12_regime_segmentation_timeline.png"))

    # Fig 13: H1 Diversification curve
    sim_h1_records = []
    np.random.seed(42)
    for k in [2, 3, 5, 8, 12, 16, min(20, len(asset_symbols))]:
        for _ in range(50):
            sub_cols = np.random.choice(asset_symbols, size=k, replace=False)
            ann_v = float(asset_rets[sub_cols].mean(axis=1).std() * np.sqrt(252.0))
            sim_h1_records.append({"k": k, "volatility": ann_v})
    plot_h1_diversification(
        pd.DataFrame(sim_h1_records),
        h1_results["slope_beta_1"],
        h1_results["intercept_beta_0"],
        os.path.join(figures_dir, "fig13_regime_conditional_sharpe.png")
    )

    # Fig 14: Bootstrap Sharpe difference distribution
    plot_bootstrap_sharpe_distribution(
        boot_diffs,
        h4_results["net_sharpe_diff"],
        h4_results["bootstrap_ci_95"][0],
        h4_results["bootstrap_ci_95"][1],
        os.path.join(figures_dir, "fig14_h1_diversification_curve.png")
    )

    # Fig 15: Tail risk comparison
    plot_tail_risk_comparison(summary_table, os.path.join(figures_dir, "fig15_bootstrap_sharpe_distribution.png"))

    # Save comprehensive results JSON
    experiment_payload = {
        "strategies": {s: backtest_results[s]["summary"] for s in strategies},
        "hypotheses": {
            "H1": h1_results,
            "H2": h2_results,
            "H3": h3_results,
            "H4": h4_results
        },
        "fee_sensitivity": {
            "fee_levels_bps": fee_levels,
            "net_cagrs": fee_sweep_cagrs
        }
    }

    def convert_numpy(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(os.path.join(results_dir, "experiment_results.json"), "w") as f:
        json.dump(experiment_payload, f, indent=2, default=convert_numpy)

    logger.info("=================================================================")
    logger.info("  RESEARCH EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
    logger.info(f"  Summary Metrics Table saved to {tables_dir}/table1_strategy_performance.csv")
    logger.info(f"  15 Publication Figures generated in {figures_dir}/")
    logger.info(f"  {h1_results['conclusion']}")
    logger.info(f"  {h2_results['conclusion']}")
    logger.info(f"  {h3_results['conclusion']}")
    logger.info(f"  {h4_results['conclusion']}")
    logger.info("=================================================================")

    return experiment_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete portfolio research experiment suite.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run_research_experiments(args.config)
