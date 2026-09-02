import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Academic publication aesthetic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def plot_cumulative_equity_curves(
    equity_df: pd.DataFrame, 
    benchmark_series: Optional[pd.Series], 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in equity_df.columns:
        ax.plot(equity_df.index, equity_df[col] / equity_df[col].iloc[0], label=col.replace("_", " ").title(), linewidth=1.8)
        
    if benchmark_series is not None:
        norm_bmk = (1.0 + benchmark_series).cumprod()
        norm_bmk = norm_bmk / norm_bmk.iloc[0]
        ax.plot(norm_bmk.index, norm_bmk, label="S&P 500 (SPY)", color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title("Figure 1: Out-of-Sample Cumulative Portfolio Equity (Rebased to $1.00)")
    ax.set_ylabel("Growth of $1.00")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 1 to {save_path}")

def plot_rolling_volatility(
    returns_dict: Dict[str, pd.Series], 
    window: int, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rets in returns_dict.items():
        roll_vol = rets.rolling(window=window).std() * np.sqrt(252.0) * 100.0
        ax.plot(roll_vol.index, roll_vol, label=name.replace("_", " ").title(), linewidth=1.5)
        
    ax.set_title(f"Figure 2: Rolling {window}-Day Annualized Volatility (%)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 2 to {save_path}")

def plot_drawdown_curves(
    drawdowns_dict: Dict[str, pd.Series], 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, dd in drawdowns_dict.items():
        ax.plot(dd.index, dd * 100.0, label=name.replace("_", " ").title(), linewidth=1.4)
        
    ax.set_title("Figure 3: Historical Peak-to-Trough Drawdown Curves (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 3 to {save_path}")

def plot_correlation_matrix(
    corr_df: pd.DataFrame, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_df, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-0.2, 
        vmax=1.0, 
        ax=ax, 
        cbar_kws={"label": "Pearson Correlation"}
    )
    ax.set_title("Figure 4: Cross-Asset Correlation Matrix")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 4 to {save_path}")

def plot_risk_return_scatter(
    asset_vols: pd.Series, 
    asset_returns: pd.Series, 
    strategy_vols: Dict[str, float], 
    strategy_returns: Dict[str, float], 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(asset_vols * 100, asset_returns * 100, color="gray", alpha=0.6, s=60, label="Individual Assets")
    for sym in asset_vols.index:
        ax.annotate(sym, (asset_vols[sym] * 100 + 0.2, asset_returns[sym] * 100 + 0.1), fontsize=8, alpha=0.8)
        
    colors = {"equal_weight": "#2563EB", "min_variance": "#059669", "mean_variance": "#D97706", "risk_parity": "#DC2626"}
    for name, vol in strategy_vols.items():
        ret = strategy_returns[name]
        ax.scatter(vol * 100, ret * 100, color=colors.get(name, "purple"), s=140, marker="*", label=name.replace("_", " ").title())
        ax.annotate(name.replace("_", " ").title(), (vol * 100 + 0.3, ret * 100 + 0.2), fontweight="bold", fontsize=9)

    ax.set_title("Figure 5: Annualized Risk vs. Return (Assets & Portfolios)")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Net Return (%)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 5 to {save_path}")

def plot_weights_evolution(
    weights_df: pd.DataFrame, 
    strategy_name: str, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(10, 5))
    weights_df.plot(kind="area", stacked=True, ax=ax, colormap="tab20", alpha=0.85)
    ax.set_title(f"Figure 6: Portfolio Allocation Evolution Over Time ({strategy_name.replace('_', ' ').title()})")
    ax.set_ylabel("Portfolio Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=8)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 6 to {save_path}")

def plot_risk_contributions(
    risk_contrib_df: pd.DataFrame, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(10, 5))
    risk_contrib_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Figure 7: Percentage Total Risk Contribution (% TRC) by Asset")
    ax.set_ylabel("Risk Contribution (%)")
    ax.set_xlabel("Asset")
    ax.axhline(100.0 / len(risk_contrib_df), color="black", linestyle="--", label="Target Equal Risk Contribution")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 7 to {save_path}")

def plot_sharpe_and_sortino(
    metrics_summary_df: pd.DataFrame, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics_summary_df))
    width = 0.35
    
    ax.bar(x - width/2, metrics_summary_df["Net Sharpe"], width, label="Net Sharpe Ratio", color="#3B82F6")
    ax.bar(x + width/2, metrics_summary_df["Net Sortino"], width, label="Net Sortino Ratio", color="#10B981")
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_summary_df.index, rotation=15)
    ax.set_title("Figure 8: Net Sharpe & Sortino Ratio Comparison")
    ax.set_ylabel("Ratio")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 8 to {save_path}")

def plot_turnover_and_costs(
    cost_summary_df: pd.DataFrame, 
    save_path: str
):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    
    x = np.arange(len(cost_summary_df))
    width = 0.35
    
    ax1.bar(x - width/2, cost_summary_df["Annual Turnover (%)"], width, label="Annual Turnover (%)", color="#8B5CF6")
    ax2.bar(x + width/2, cost_summary_df["Return Drag (bps)"], width, label="Return Drag (bps)", color="#EF4444")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(cost_summary_df.index, rotation=15)
    ax1.set_ylabel("Annual Turnover (%)", color="#8B5CF6")
    ax2.set_ylabel("Return Drag (Basis Points)", color="#EF4444")
    ax1.set_title("Figure 9: Portfolio Turnover vs. Transaction Cost Return Drag")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 9 to {save_path}")

def plot_fee_sensitivity(
    fee_levels: List[float], 
    strategy_net_cagrs: Dict[str, List[float]], 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, cagrs in strategy_net_cagrs.items():
        ax.plot(fee_levels, [c * 100 for c in cagrs], marker="o", label=name.replace("_", " ").title())
        
    ax.set_title("Figure 10: Net CAGR Degradation Across Transaction Cost Levels")
    ax.set_xlabel("Total Trading Cost (Basis Points)")
    ax.set_ylabel("Net CAGR (%)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 10 to {save_path}")

def plot_regime_timeline(
    regime_df: pd.DataFrame, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(regime_df.index, regime_df["realized_volatility"] * 100.0, color="#1E293B", linewidth=1.5, label="63-Day Realized Volatility (%)")
    
    low_mask = regime_df["regime"] == "LOW_VOL"
    high_mask = regime_df["regime"] == "HIGH_VOL"
    
    ax.fill_between(regime_df.index, 0, 70, where=low_mask, color="#10B981", alpha=0.15, label="Low Volatility Regime")
    ax.fill_between(regime_df.index, 0, 70, where=high_mask, color="#EF4444", alpha=0.20, label="High Volatility Regime")
    
    ax.set_title("Figure 11: Market Volatility Regimes Over Time")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 11 to {save_path}")

def plot_regime_sharpe_comparison(
    regime_results_dict: Dict[str, pd.DataFrame], 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(9, 5))
    regimes = ["LOW_VOL", "NORMAL_VOL", "HIGH_VOL"]
    x = np.arange(len(regimes))
    width = 0.2
    
    for i, (strat_name, df) in enumerate(regime_results_dict.items()):
        sharpes = [df.loc[r, "Sharpe Ratio"] if r in df.index else 0.0 for r in regimes]
        ax.bar(x + (i - 1.5) * width, sharpes, width, label=strat_name.replace("_", " ").title())
        
    ax.set_xticks(x)
    ax.set_xticklabels(["Low Volatility", "Normal Volatility", "High Volatility"])
    ax.set_title("Figure 12: Strategy Sharpe Ratio by Market Volatility Regime")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 12 to {save_path}")

def plot_h1_diversification(
    sim_data: pd.DataFrame, 
    slope: float, 
    intercept: float, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sim_data["k"], sim_data["volatility"] * 100.0, alpha=0.4, color="#3B82F6", label="Random Sub-Portfolios")
    
    # Plot asymptotic decay fit
    k_smooth = np.linspace(sim_data["k"].min(), sim_data["k"].max(), 200)
    fit_vol = (intercept + slope * (1.0 / np.sqrt(k_smooth))) * 100.0
    ax.plot(k_smooth, fit_vol, color="#DC2626", linewidth=2.0, label=r"Fitted: $\beta_0 + \beta_1 / \sqrt{k}$")
    
    ax.set_title(r"Figure 13: H1 Diversification — Portfolio Volatility vs Universe Size $k$")
    ax.set_xlabel("Number of Assets in Portfolio (k)")
    ax.set_ylabel("Annualized Portfolio Volatility (%)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 13 to {save_path}")

def plot_bootstrap_sharpe_distribution(
    diff_samples: np.ndarray, 
    pt_diff: float, 
    ci_lower: float, 
    ci_upper: float, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(diff_samples, kde=True, color="#6366F1", ax=ax)
    ax.axvline(pt_diff, color="blue", linewidth=2, label=f"Point Estimate ({pt_diff:.3f})")
    ax.axvline(ci_lower, color="red", linestyle="--", label=f"95% CI Lower ({ci_lower:.3f})")
    ax.axvline(ci_upper, color="red", linestyle="--", label=f"95% CI Upper ({ci_upper:.3f})")
    ax.axvline(0.0, color="black", linestyle=":", label="Zero Difference Baseline")
    
    ax.set_title(r"Figure 14: Bootstrap Distribution of Sharpe Difference ($\Delta SR = SR_{MVO} - SR_{1/N}$)")
    ax.set_xlabel(r"$\Delta$ Sharpe Ratio")
    ax.legend(frameon=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 14 to {save_path}")

def plot_tail_risk_comparison(
    tail_risk_df: pd.DataFrame, 
    save_path: str
):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tail_risk_df))
    width = 0.35
    
    ax.bar(x - width/2, tail_risk_df["Historical VaR 95 (%)"], width, label="Historical VaR (95%)", color="#F59E0B")
    ax.bar(x + width/2, tail_risk_df["CVaR 95 (%)"], width, label="Expected Shortfall CVaR (95%)", color="#B91C1C")
    
    ax.set_xticks(x)
    ax.set_xticklabels(tail_risk_df.index, rotation=15)
    ax.set_title("Figure 15: Tail Risk Comparison Across Portfolio Strategies")
    ax.set_ylabel("Loss at 95% Confidence (%)")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 15 to {save_path}")
