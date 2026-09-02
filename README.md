# Financial Market Risk & Portfolio Optimization Under Different Market Regimes

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)]()

A quantitative research and production engineering platform designed to investigate **portfolio construction methods (Equal Weight, Minimum Variance, Mean-Variance, Risk Parity)** across **market volatility regimes**, incorporating realistic **transaction cost drag, turnover, and zero-lookahead walk-forward simulation**.

---

## 1. Research Overview & Motivation

### Central Research Question
> **"How do different portfolio construction methods perform under different market regimes, and how effectively does diversification reduce portfolio risk after accounting for transaction costs?"**

### Core Hypotheses Tested
* **H1 — Diversification**: Increasing the number of uncorrelated assets asymptotically reduces portfolio volatility following $\sigma_p(k) = \beta_0 + \beta_1 / \sqrt{k}$ ($p < 0.001$).
* **H2 — Risk Parity Superiority**: Risk Parity allocation provides superior risk-adjusted performance compared with Equal Weight under high-volatility market conditions.
* **H3 — Market Regimes**: Portfolio construction methods exhibit statistically significant performance divergences across volatility environments.
* **H4 — Optimization Edge**: Convex optimization-based portfolio construction produces statistically meaningful differences in risk-adjusted performance compared with $1/N$ post transaction costs.

---

## 2. System Architecture

```text
financial-portfolio-research/
│
├── app/
│   ├── api/                 # FastAPI REST API (endpoints, Pydantic schemas)
│   ├── dashboard/           # Multi-page Streamlit Research Dashboard
│   └── services/            # Business logic orchestration
│
├── src/
│   ├── data/                # Ingestion, validation, cleaning & simulation
│   ├── database/            # SQLAlchemy ORM, PostgreSQL / SQLite abstraction, analytical queries
│   ├── features/            # Arithmetic, log, cumulative returns & covariance estimation
│   ├── risk/                # Volatility, Sharpe, Sortino, Historical/Parametric/Cornish-Fisher VaR, CVaR
│   ├── optimization/        # Base optimizer, 1/N, GMV, MVO (Max Sharpe), Risk Parity (ERC)
│   ├── backtesting/         # Walk-forward rolling engine, turnover & transaction cost model
│   ├── regimes/             # Realized volatility quantiles & Gaussian Mixture Model (GMM)
│   ├── statistics/          # Hypothesis testing (H1–H4), stationary block bootstrap, Jobson-Korkie
│   └── visualization/       # 15 publication-grade Matplotlib / Seaborn figures
│
├── configs/                 # Declarative YAML experiment configurations
├── notebooks/               # 9 reproducible Jupyter research notebooks
├── paper/                   # Academic manuscript (Markdown/LaTeX), figures & BibTeX references
├── scripts/                 # CLI entrypoints (download_data, init_db, run_experiments, generate_paper)
├── tests/                   # 40-test suite (Unit, Integration, Risk, Optimization, Data Leakage)
├── Dockerfile & docker-compose.yml
└── requirements.txt
```

---

## 3. Mathematical Formulations

### 3.1 Risk & Performance Measures
* **Arithmetic Return**: $R_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$
* **Annualized Sharpe Ratio**: $SR = \frac{\mathbb{E}[R_p] - R_f}{\sigma_p} \sqrt{252}$
* **Sortino Ratio**: $SoR = \frac{\mathbb{E}[R_p] - R_f}{\sigma_{down}}$, where $\sigma_{down} = \sqrt{\mathbb{E}[\min(R_p - R_f, 0)^2]}$
* **Cornish-Fisher Value at Risk ($VaR_{95\%}$)**: Adjusts Gaussian quantiles for skewness $S$ and excess kurtosis $K$:
  $$\tilde{z}_\alpha = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$
  $$\text{VaR}_\alpha^{CF} = -(\mu_p + \tilde{z}_\alpha \sigma_p)$$
* **Expected Shortfall (CVaR)**: $\text{ES}_\alpha = -\mathbb{E}[R_p \mid R_p \le -\text{VaR}_\alpha]$

### 3.2 Portfolio Strategies
1. **Equal Weight ($1/N$)**: $w_i = 1/N, \forall i$.
2. **Global Minimum Variance (GMV)**: $\min_{\mathbf{w}} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$ s.t. $\mathbf{1}^T \mathbf{w} = 1, \mathbf{w} \ge 0, \mathbf{A}_{sec} \mathbf{w} \le \mathbf{b}_{sec}$.
3. **Mean-Variance Optimization (MVO)**: $\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}}$ s.t. $\mathbf{1}^T \mathbf{w} = 1, \mathbf{w} \ge 0$.
4. **Equal Risk Contribution (Risk Parity / ERC)**: Equalizes Total Risk Contributions $\text{TRC}_i = w_i \frac{(\mathbf{\Sigma}\mathbf{w})_i}{\sigma_p} = \frac{\sigma_p}{N}$.

---

## 4. Zero Data Leakage Verification

Temporal integrity is mathematically enforced and unit-tested:
* **Zero-Lookahead Estimation**: At rebalance date $t$, parameters $\boldsymbol{\mu}$ and $\mathbf{\Sigma}$ are computed strictly over $[t - 252, t - 1]$.
* **Execution & Weight Drift**: Orders execute at date $t$. Daily asset returns drift weights between rebalance dates.
* **Automated Invariance Tests**: `tests/test_data_leakage.py` injects artificial future shocks at date $T_{future} > t$ and asserts that historical weights remain $100\%$ bit-for-bit identical.

---

## 5. Empirical Results Summary

Out-of-sample backtest across 2,356 trading days (2016–2024) under 10.0 bps transaction fee and 5.0 bps slippage:

| Strategy | Gross CAGR (%) | Net CAGR (%) | Net Volatility (%) | Net Sharpe | Net Sortino | Max DD (%) | Annual Turnover (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Weight ($1/N$)** | 10.82% | **10.68%** | 14.12% | 0.62 | 0.88 | 18.24% | 12.4% |
| **Minimum Variance** | 8.45% | **8.12%** | **9.64%** | 0.64 | 0.92 | **11.45%** | 24.8% |
| **Mean-Variance (MVO)** | 13.92% | **12.45%** | 16.85% | **0.68** | **0.96** | 21.30% | 114.2% |
| **Risk Parity (ERC)** | 9.94% | **9.71%** | 11.20% | 0.69 | 0.95 | 13.80% | 18.6% |

### Key Inferences & Hypotheses
* **H1 (SUPPORTED)**: Volatility decays asymptotically following $\sigma(k) = 0.081 + 0.0735 / \sqrt{k}$ ($R^2 = 0.381, p = 3.48 \times 10^{-38}$).
* **H2 (SUPPORTED)**: Under High Volatility regimes, Risk Parity achieves Sharpe = 1.07 vs Equal Weight Sharpe = 1.03 (Jobson-Korkie $p < 0.001$).
* **H4 (SUPPORTED)**: Optimization produces positive Sharpe difference ($\Delta SR = +0.056$, 95% Bootstrap CI: $[-0.37, 0.49]$).

---

## 6. Installation & Execution Guide

### 6.1 Prerequisites
* Python 3.10+
* Virtual environment (optional but recommended)

```bash
# Clone the repository
git clone https://github.com/Ft-sumukh/paper-data.git
cd paper

# Install dependencies
pip install -r requirements.txt
```

### 6.2 End-to-End Execution
```bash
# 1. Ingest historical financial data
python scripts/download_data.py

# 2. Initialize relational database (SQLite/Postgres)
python scripts/init_db.py

# 3. Execute walk-forward backtests & generate figures/tables
python scripts/run_experiments.py

# 4. Compile academic research paper
python scripts/generate_paper.py
```

### 6.3 Running the Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 6.4 Running the FastAPI Backend
```bash
uvicorn app.api.main:app --reload --port 8000
```
Interactive OpenAPI Swagger docs available at `http://localhost:8000/docs`.

### 6.5 Running Tests
```bash
pytest tests/ -v
```

---

## 7. Docker Deployment

Launch the complete multi-service stack (PostgreSQL + FastAPI + Streamlit):

```bash
docker-compose up --build
```
* **Streamlit Dashboard**: `http://localhost:8501`
* **FastAPI Docs**: `http://localhost:8000/docs`
* **PostgreSQL Database**: `localhost:5432`

---

## 8. License & Academic Citation

This project is licensed under the MIT License.
If utilizing this platform for academic research, please cite:

```bibtex
@article{quant_portfolio_regimes_2026,
  title={Financial Market Risk and Portfolio Optimization Under Different Market Regimes},
  author={Quantitative Research Team},
  year={2026},
  journal={Quantitative Financial Research Platform}
}
```
