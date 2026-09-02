import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def create_notebook(filename: str, title: str, cells_data: list):
    nb = {
        "cells": [],
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Title cell
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n", f"**Financial Market Risk & Portfolio Optimization Research Platform**\n"]
    })

    for cell_type, content in cells_data:
        lines = [line + "\n" for line in content.split("\n")]
        # strip trailing newline on last line
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
            "outputs": [] if cell_type == "code" else None,
            "execution_count": None if cell_type == "code" else None
        })

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    logger.info(f"Generated notebook: {filename}")

def main():
    notebooks_dir = "notebooks"
    
    # 01_data_collection
    create_notebook(
        os.path.join(notebooks_dir, "01_data_collection.ipynb"),
        "01 — Historical Financial Market Data Collection",
        [
            ("markdown", "### Ingesting Historical Multi-Asset Universe Across 10 GICS Sectors\nDownloads daily price series for equities, treasuries, and gold."),
            ("code", "import sys\nsys.path.append('..')\nimport yaml\nfrom src.data.downloader import MarketDataDownloader\n\nwith open('../configs/default.yaml') as f:\n    config = yaml.safe_load(f)\n\nsymbols = [a['symbol'] for a in config['data']['assets']]\ndownloader = MarketDataDownloader(raw_data_dir='../data/raw', metadata_dir='../data/metadata')\nraw_df = downloader.download_universe(symbols, start_date=config['data']['start_date'], end_date=config['data']['end_date'])\nprint('Data shape:', raw_df.shape)\nraw_df.tail()")
        ]
    )

    # 02_data_validation
    create_notebook(
        os.path.join(notebooks_dir, "02_data_validation.ipynb"),
        "02 — Data Validation and Boundary Sanity Checks",
        [
            ("markdown", "### Validating Raw Prices for Anomaly Detection, Missing Values & Duplicate Timestamps"),
            ("code", "import sys\nsys.path.append('..')\nimport pandas as pd\nfrom src.data.validator import DataValidator\nfrom src.data.cleaner import DataCleaner\n\nraw_df = pd.read_csv('../data/raw/market_prices_raw.csv', index_col=0, parse_dates=True)\nvalidator = DataValidator()\nreport = validator.validate_prices(raw_df)\nprint('Validation report valid:', report['is_valid'])\n\ncleaner = DataCleaner(processed_data_dir='../data/processed')\nclean_df, log = cleaner.clean_prices(raw_df)\nprint('Cleaned dataset shape:', clean_df.shape)")
        ]
    )

    # 03_exploratory_analysis
    create_notebook(
        os.path.join(notebooks_dir, "03_exploratory_analysis.ipynb"),
        "03 — Exploratory Data Analysis & Asset Correlation Dynamics",
        [
            ("markdown", "### Cross-Asset Returns, Sector Groupings, and Empirical Correlation Matrix"),
            ("code", "import sys\nsys.path.append('..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.features.returns import calculate_simple_returns, calculate_correlation_matrix\n\nprices = pd.read_csv('../data/processed/prices_clean.csv', index_col=0, parse_dates=True)\nrets = calculate_simple_returns(prices)\ncorr = calculate_correlation_matrix(rets)\n\nplt.figure(figsize=(10, 8))\nsns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.2, vmax=1.0)\nplt.title('Asset Pearson Correlation Matrix')\nplt.show()")
        ]
    )

    # 04_risk_analysis
    create_notebook(
        os.path.join(notebooks_dir, "04_risk_analysis.ipynb"),
        "04 — Financial Risk Measurement & Tail Risk Analysis",
        [
            ("markdown", "### Value at Risk (VaR), Expected Shortfall (CVaR), and Historical Peak-to-Trough Drawdowns"),
            ("code", "import sys\nsys.path.append('..')\nimport pandas as pd\nfrom src.risk.metrics import calculate_all_risk_metrics, calculate_drawdown_series\nfrom src.features.returns import calculate_simple_returns\n\nprices = pd.read_csv('../data/processed/prices_clean.csv', index_col=0, parse_dates=True)\nrets = calculate_simple_returns(prices)\new_port = rets.mean(axis=1)\n\nmetrics = calculate_all_risk_metrics(ew_port, risk_free_rate=0.02)\nprint('Comprehensive Risk Summary:')\nfor k, v in metrics.items():\n    print(f'{k:25s}: {v:.4f}')")
        ]
    )

    # 05_portfolio_optimization
    create_notebook(
        os.path.join(notebooks_dir, "05_portfolio_optimization.ipynb"),
        "05 — Convex Portfolio Optimization & Risk Budgeting",
        [
            ("markdown", "### Optimizing Equal Weight, Global Minimum Variance, Mean-Variance, and Risk Parity"),
            ("code", "import sys\nsys.path.append('..')\nimport pandas as pd\nfrom src.optimization import get_optimizer\nfrom src.features.returns import calculate_simple_returns, calculate_covariance_matrix\n\nprices = pd.read_csv('../data/processed/prices_clean.csv', index_col=0, parse_dates=True)\nrets = calculate_simple_returns(prices)\nmu = rets.mean().values * 252.0\ncov = calculate_covariance_matrix(rets, annualize=True)\nsymbols = list(prices.columns)\n\nfor strat in ['equal_weight', 'min_variance', 'mean_variance', 'risk_parity']:\n    opt = get_optimizer(strat)\n    res = opt.optimize(mu, cov, symbols)\n    print(f'=== {strat.upper()} ===')\n    print(f'Expected Return: {res.expected_return*100:.2f}%, Volatility: {res.volatility*100:.2f}%, Sharpe: {res.sharpe_ratio:.2f}')")
        ]
    )

    # 06_backtesting
    create_notebook(
        os.path.join(notebooks_dir, "06_backtesting.ipynb"),
        "06 — Walk-Forward Out-of-Sample Backtesting & Transaction Cost Drag",
        [
            ("markdown", "### Rolling Window Backtesting with Turnover Tracking, Trading Fees, and Slippage"),
            ("code", "import sys\nsys.path.append('..')\nimport pandas as pd\nfrom src.backtesting.engine import WalkForwardBacktester\nfrom src.backtesting.performance import PerformanceReporter\n\nprices = pd.read_csv('../data/processed/prices_clean.csv', index_col=0, parse_dates=True)\nbt = WalkForwardBacktester('risk_parity', prices, estimation_window=252, rebalance_frequency='monthly')\nres = bt.run()\nprint('Risk Parity Out-of-Sample Results:')\nfor k, v in res['summary'].items():\n    print(f'{k:25s}: {v}')")
        ]
    )

    # 07_market_regimes
    create_notebook(
        os.path.join(notebooks_dir, "07_market_regimes.ipynb"),
        "07 — Market Volatility Regimes & Performance Segmentation",
        [
            ("markdown", "### Realized Volatility Quantiles, GMM Clustering, and Regime-Conditional Sharpe Analysis"),
            ("code", "import sys\nsys.path.append('..')\nimport pandas as pd\nfrom src.regimes.detector import RollingVolatilityRegimeDetector\nfrom src.regimes.segmentation import RegimePerformanceAnalyzer\nfrom src.features.returns import calculate_simple_returns\n\nprices = pd.read_csv('../data/processed/prices_clean.csv', index_col=0, parse_dates=True)\nbmk_rets = calculate_simple_returns(prices['SPY'])\ndetector = RollingVolatilityRegimeDetector(window=63)\nreg_df = detector.fit_predict(bmk_rets)\nprint('Regime distribution:\\n', reg_df['regime'].value_counts())")
        ]
    )

    # 08_statistical_tests
    create_notebook(
        os.path.join(notebooks_dir, "08_statistical_tests.ipynb"),
        "08 — Econometric Testing & Statistical Inferences (H1–H4)",
        [
            ("markdown", "### Jobson-Korkie Memmel Tests, Stationary Block Bootstrap CIs, and Hypothesis Validation"),
            ("code", "import sys\nsys.path.append('..')\nimport pandas as pd\nfrom src.statistics.hypothesis import ResearchHypothesisTester\n\nprices = pd.read_csv('../data/processed/prices_clean.csv', index_col=0, parse_dates=True)\nh1_res = ResearchHypothesisTester.test_h1_diversification(prices, subsets_per_k=50)\nprint('H1 Result:', h1_res['conclusion'])")
        ]
    )

    # 09_results
    create_notebook(
        os.path.join(notebooks_dir, "09_results.ipynb"),
        "09 — Final Empirical Results, Tables & Visualizations",
        [
            ("markdown", "### Multi-Strategy Comparative Analysis, Equity Curves & Academic Tables"),
            ("code", "import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nwith open('../results/experiment_results.json') as f:\n    results = json.load(f)\n\nt1 = pd.read_csv('../paper/tables/table1_strategy_performance.csv', index_col=0)\nprint('Table 1: Out-of-Sample Performance')\ndisplay(t1) if 'display' in globals() else print(t1)")
        ]
    )

if __name__ == "__main__":
    main()
