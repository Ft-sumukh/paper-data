import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.connection import SessionLocal, init_db
from src.database.models import Asset
from src.database.crud import get_price_matrix, get_all_assets
from src.features.returns import (
    calculate_simple_returns, calculate_covariance_matrix, 
    calculate_correlation_matrix, calculate_wealth_index
)
from src.risk.metrics import calculate_all_risk_metrics, calculate_drawdown_series
from src.optimization import get_optimizer
from src.backtesting.engine import WalkForwardBacktester
from src.backtesting.performance import PerformanceReporter
from src.regimes.detector import RollingVolatilityRegimeDetector
from src.regimes.segmentation import RegimePerformanceAnalyzer
from src.statistics.hypothesis import ResearchHypothesisTester
from src.statistics.bootstrap import StationaryBlockBootstrap

# Page configuration
st.set_page_config(
    page_title="Financial Market Risk & Portfolio Optimization Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: 700; color: #1E3A8A; margin-bottom: 5px; }
    .sub-header { font-size: 16px; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 4px solid #3B82F6; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_market_data():
    db = SessionLocal()
    try:
        assets = get_all_assets(db)
        prices = get_price_matrix(db)
        return assets, prices
    finally:
        db.close()

try:
    assets, prices_df = load_market_data()
except Exception as e:
    st.error(f"Error loading database: {e}. Please ensure data is ingested via scripts/init_db.py.")
    st.stop()

if prices_df.empty:
    st.warning("Database contains no price records. Run `python scripts/download_data.py` and `python scripts/init_db.py`.")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module:",
    ["📊 Overview & Universe", "⚖️ Portfolio Optimization", "🛡️ Risk Management", "📈 Walk-Forward Backtesting", "🌪️ Market Regimes", "🔬 Research & Hypotheses"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Universe Parameters")
symbols_all = [c for c in prices_df.columns if c != "SPY"]
selected_symbols = st.sidebar.multiselect("Active Assets:", symbols_all, default=symbols_all[:12])
rf_rate = st.sidebar.number_input("Risk-Free Rate (Annual):", min_value=0.0, max_value=0.10, value=0.02, step=0.005)

if len(selected_symbols) < 2:
    st.warning("Please select at least 2 assets in the sidebar.")
    st.stop()

active_prices = prices_df[selected_symbols]
active_returns = calculate_simple_returns(active_prices)

# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
if page == "📊 Overview & Universe":
    st.markdown("<div class='main-header'>Financial Market Risk & Portfolio Optimization Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Asset Universe Dynamics, Historical Normalized Performance, and Cross-Asset Correlation</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Assets", len(selected_symbols))
    with col2:
        st.metric("Trading Days", len(active_prices))
    with col3:
        st.metric("Start Date", str(active_prices.index[0].date()))
    with col4:
        st.metric("End Date", str(active_prices.index[-1].date()))

    st.markdown("### Normalized Price Trajectories (Rebased to $100)")
    norm_prices = (active_prices / active_prices.iloc[0]) * 100.0
    st.line_chart(norm_prices)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Cross-Asset Correlation Heatmap")
        corr_matrix = calculate_correlation_matrix(active_returns)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, vmin=-0.2, vmax=1.0)
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("### Asset Risk-Return Scatter (Annualized)")
        ann_rets = active_returns.mean() * 252.0 * 100
        ann_vols = active_returns.std() * np.sqrt(252.0) * 100
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(ann_vols, ann_rets, color="#2563EB", s=80, alpha=0.8)
        for sym in selected_symbols:
            ax2.annotate(sym, (ann_vols[sym] + 0.3, ann_rets[sym] + 0.2), fontsize=9)
        ax2.set_xlabel("Annualized Volatility (%)")
        ax2.set_ylabel("Annualized Return (%)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

# ==============================================================================
# 2. PORTFOLIO OPTIMIZATION PAGE
# ==============================================================================
elif page == "⚖️ Portfolio Optimization":
    st.markdown("<div class='main-header'>Convex Portfolio Optimization Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Optimal Asset Allocation, Risk Budgeting, and Constraint Management</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        strat = st.selectbox("Strategy:", ["Equal Weight (1/N)", "Minimum Variance", "Mean-Variance (Max Sharpe)", "Risk Parity (ERC)"])
    with c2:
        max_w = st.slider("Maximum Asset Weight:", min_value=0.10, max_value=1.0, value=0.35, step=0.05)
    with c3:
        max_sec_w = st.slider("Maximum Sector Exposure:", min_value=0.20, max_value=1.0, value=0.40, step=0.05)

    strat_map = {
        "Equal Weight (1/N)": "equal_weight",
        "Minimum Variance": "min_variance",
        "Mean-Variance (Max Sharpe)": "mean_variance",
        "Risk Parity (ERC)": "risk_parity"
    }

    # Retrieve sectors
    sector_mapping = {a.symbol: a.sector for a in assets if a.symbol in selected_symbols}
    mu = active_returns.mean().values * 252.0
    cov = calculate_covariance_matrix(active_returns, annualize=True)

    opt = get_optimizer(
        strategy_name=strat_map[strat],
        risk_free_rate=rf_rate,
        max_weight=max_w,
        max_sector_weight=max_sec_w
    )
    res = opt.optimize(mu, cov, selected_symbols, sector_mapping)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Expected Annual Return", f"{res.expected_return * 100:.2f}%")
    with col_m2:
        st.metric("Annualized Volatility", f"{res.volatility * 100:.2f}%")
    with col_m3:
        st.metric("Sharpe Ratio", f"{res.sharpe_ratio:.2f}")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("### Optimal Portfolio Weights (%)")
        w_df = pd.DataFrame({
            "Asset": list(res.symbol_weights.keys()),
            "Weight (%)": [w * 100.0 for w in res.symbol_weights.values()]
        }).set_index("Asset")
        st.bar_chart(w_df)

    with col_p2:
        st.markdown("### Percentage Risk Contribution (% TRC)")
        rc_df = pd.DataFrame({
            "Asset": selected_symbols,
            "Risk Contribution (%)": res.percentage_risk_contributions * 100.0
        }).set_index("Asset")
        st.bar_chart(rc_df)

# ==============================================================================
# 3. RISK MANAGEMENT PAGE
# ==============================================================================
elif page == "🛡️ Risk Management":
    st.markdown("<div class='main-header'>Tail Risk & Quantitative Risk Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Value at Risk (VaR), Expected Shortfall (CVaR), and Historical Peak-to-Trough Drawdowns</div>", unsafe_allow_html=True)

    # Calculate equal weight portfolio returns as standard illustration
    port_rets = active_returns.mean(axis=1)
    bmk_rets = calculate_simple_returns(prices_df["SPY"]).reindex(port_rets.index).dropna() if "SPY" in prices_df.columns else None

    metrics = calculate_all_risk_metrics(port_rets, bmk_rets, risk_free_rate=rf_rate)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Historical VaR (95%)", f"{metrics['var_95_historical'] * 100:.2f}%")
    with r2:
        st.metric("Parametric VaR (95%)", f"{metrics['var_95_parametric'] * 100:.2f}%")
    with r3:
        st.metric("Cornish-Fisher VaR (95%)", f"{metrics['var_95_cornish_fisher'] * 100:.2f}%")
    with r4:
        st.metric("Expected Shortfall / CVaR (95%)", f"{metrics['cvar_95'] * 100:.2f}%")

    st.markdown("### Historical Drawdown Curve")
    dd_df = calculate_drawdown_series(port_rets)
    st.area_chart(dd_df["drawdown"] * 100.0)

    st.markdown("### Daily Return Distribution with 95% Tail Risk Cutoffs")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(port_rets * 100, bins=50, kde=True, color="#3B82F6", ax=ax)
    ax.axvline(-metrics['var_95_historical'] * 100, color="red", linestyle="--", label=f"Historical VaR 95% ({metrics['var_95_historical']*100:.2f}%)")
    ax.axvline(-metrics['cvar_95'] * 100, color="darkred", linestyle=":", linewidth=2, label=f"Expected Shortfall CVaR ({metrics['cvar_95']*100:.2f}%)")
    ax.set_xlabel("Daily Portfolio Return (%)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ==============================================================================
# 4. WALK-FORWARD BACKTESTING PAGE
# ==============================================================================
elif page == "📈 Walk-Forward Backtesting":
    st.markdown("<div class='main-header'>Out-of-Sample Walk-Forward Backtester</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Sequential Realistic Simulation with Turnover Tracking, Trading Fees, and Slippage</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        rebal_freq = st.selectbox("Rebalance Frequency:", ["monthly", "quarterly", "weekly"])
    with b2:
        fee_bps = st.number_input("Transaction Fee (bps):", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    with b3:
        slip_bps = st.number_input("Slippage (bps):", min_value=0.0, max_value=50.0, value=5.0, step=1.0)

    if st.button("Run Multi-Strategy Walk-Forward Backtest", type="primary"):
        with st.spinner("Executing walk-forward optimization across historical windows..."):
            strategies = ["equal_weight", "min_variance", "mean_variance", "risk_parity"]
            results = {}
            for strat in strategies:
                bt = WalkForwardBacktester(
                    strategy_name=strat,
                    prices=prices_df,
                    symbols=selected_symbols,
                    benchmark_symbol="SPY",
                    rebalance_frequency=rebal_freq,
                    transaction_cost_bps=fee_bps,
                    slippage_bps=slip_bps,
                    risk_free_rate=rf_rate
                )
                results[strat] = bt.run()

            st.markdown("### Cumulative Portfolio Value ($ Equity Curves)")
            equity_dict = {}
            for s, r in results.items():
                equity_dict[s.replace("_", " ").title()] = r["performance_df"]["portfolio_value"]
            equity_df = pd.DataFrame(equity_dict)
            st.line_chart(equity_df)

            st.markdown("### Multi-Strategy Performance Comparison Table")
            summary_table = PerformanceReporter.generate_strategy_comparison_table(results)
            st.dataframe(summary_table, use_container_width=True)

            st.markdown("### Transaction Cost Drag Analysis")
            cost_table = PerformanceReporter.generate_cost_drag_analysis(results)
            st.dataframe(cost_table, use_container_width=True)

# ==============================================================================
# 5. MARKET REGIMES PAGE
# ==============================================================================
elif page == "🌪️ Market Regimes":
    st.markdown("<div class='main-header'>Market Volatility Regimes & Conditional Dynamics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Regime Segmentation (Low, Normal, High Volatility) and Regime-Dependent Asset Performance</div>", unsafe_allow_html=True)

    bmk_rets = calculate_simple_returns(prices_df["SPY"]) if "SPY" in prices_df.columns else active_returns.mean(axis=1)
    detector = RollingVolatilityRegimeDetector(window=63)
    regime_df = detector.fit_predict(bmk_rets)

    st.markdown("### Benchmark Realized Volatility & Regime Classification")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(regime_df.index, regime_df["realized_volatility"] * 100, label="63-day Realized Vol (%)", color="#1F2937")
    
    # Highlight high vol
    high_vol_mask = regime_df["regime"] == "HIGH_VOL"
    ax.fill_between(regime_df.index, 0, 60, where=high_vol_mask, color="red", alpha=0.2, label="High Volatility Regime")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend(loc="upper left")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Equal-Weight Performance Segmented by Volatility Regime")
    port_rets = active_returns.mean(axis=1)
    regime_perf = RegimePerformanceAnalyzer.analyze_regime_performance(port_rets, regime_df["regime"], risk_free_rate=rf_rate)
    st.dataframe(regime_perf, use_container_width=True)

# ==============================================================================
# 6. RESEARCH & HYPOTHESES PAGE
# ==============================================================================
elif page == "🔬 Research & Hypotheses":
    st.markdown("<div class='main-header'>Statistical Hypothesis Testing & Research Evidence</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Rigorous Statistical Testing of Hypotheses H1, H2, H3, and H4 with Asymptotic and Bootstrap Confidence Intervals</div>", unsafe_allow_html=True)

    h_tab1, h_tab2, h_tab3, h_tab4 = st.tabs([
        "H1: Diversification", "H2: Risk Parity", "H3: Market Regimes", "H4: Optimization Edge"
    ])

    with h_tab1:
        st.markdown("#### H1 — Diversification Hypothesis")
        st.markdown("> **Hypothesis**: Increasing the number of sufficiently uncorrelated assets reduces portfolio volatility asymptotically following $\\sigma_p \\propto 1/\\sqrt{k}$.")
        if st.button("Run H1 Statistical Test"):
            with st.spinner("Running 500 sub-portfolio simulations..."):
                h1_res = ResearchHypothesisTester.test_h1_diversification(active_prices, subsets_per_k=50)
                st.success(h1_res["conclusion"])
                vols_k = pd.DataFrame(list(h1_res["mean_volatility_by_k"].items()), columns=["Universe Size (k)", "Mean Volatility"])
                st.line_chart(vols_k.set_index("Universe Size (k)"))

    with h_tab2:
        st.markdown("#### H2 — Risk Parity Superiority in Volatile Regimes")
        st.markdown("> **Hypothesis**: Risk-parity allocation provides superior risk-adjusted performance compared with equal-weight allocation under high-volatility conditions.")
        st.info("Execute walk-forward backtest to view Jobson-Korkie and Wilcoxon tests across regimes.")

    with h_tab3:
        st.markdown("#### H3 — Market Regimes Impact")
        st.markdown("> **Hypothesis**: Portfolio construction methods exhibit statistically different performance characteristics across volatility regimes.")
        st.info("ANOVA & Kruskal-Wallis tests confirm whether Sharpe ratio distributions differ significantly across regimes.")

    with h_tab4:
        st.markdown("#### H4 — Optimization Edge Over 1/N Benchmark")
        st.markdown("> **Hypothesis**: Optimization-based portfolio construction produces statistically meaningful differences in risk-adjusted performance vs 1/N after transaction costs.")
        st.info("Stationary Block Bootstrap (1,000 iterations) evaluates the 95% Confidence Interval of the Sharpe Ratio difference.")
