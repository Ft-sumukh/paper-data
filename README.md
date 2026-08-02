# Short-Horizon Price Prediction from Limit Order Book Dynamics

This repository contains a complete, research-grade machine learning project suitable for an undergraduate research study or a strong quantitative portfolio project. 

## Project Title
**"Short-Horizon Price Prediction from Limit Order Book Dynamics: A Comparative Study of Order-Flow Imbalance, LSTM, and Transformer Models"**

---

## 1. Directory Structure

```
.
├── requirements.txt         # Package dependencies
├── config.yaml              # Global project and model configurations
├── run_experiments.py       # Main orchestration script
├── README.md                # This instructions file
├── src/                     # Source modules
│   ├── data_loader.py       # Data fetching, synthetic simulation, and splitting
│   ├── features.py          # Microstructure & LOB feature engineering
│   ├── sequence.py          # Sliding window dataset and standardizing loader
│   ├── baselines.py         # Baseline classifiers (Majority, Random, LogReg, RF)
│   ├── models.py            # PyTorch models (LSTM and Transformer Encoder)
│   ├── train.py             # Early-stopping neural optimizer and evaluator
│   ├── backtest.py          # Event-driven backtester with transaction costs
│   ├── stats.py             # Validation tests (Bootstrap CI, McNemar's, DM)
│   ├── interpret.py         # Coefficient and Permutation feature importance
│   ├── visualizations.py    # Generator for all 15 publication-grade figures
│   └── paper_generator.py   # Research paper text compiler
├── tests/                   # PyTest unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_backtest.py
├── results/                 # YAML metrics file and saved PyTorch checkpoints
├── figures/                 # Saved PNG high-resolution figures (1 to 15)
└── paper/                   # Compiled Markdown research paper draft
```

---

## 2. Installation and Setup

1. **Install Python Packages**:
   Ensure you have Python 3.10+ installed. Install the requirements using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Unit Tests**:
   Verify the modules are working correctly by executing:
   ```bash
   pytest
   ```

---

## 3. Running the Empirical Experiments

To run all experiments, grid sensitivity sweeps, backtesting transaction cost sensitivity evaluations, statistical significance tests, and interpretability evaluations, execute the following script:

```bash
python run_experiments.py
```

This script will automatically:
1. Load or download the raw NASDAQ TSLA Limit Order Book sample.
2. Compute microstructure features (OFI, spreads, returns, imbalances).
3. Chronologically split the dataset to avoid look-ahead bias and data leakage.
4. Train baseline classifiers (Majority Class, Random class, Logistic Regression on Level-1 OFI, Logistic Regression on Full features, and Random Forest).
5. Train deep sequence-based networks (LSTM and Transformer Encoder) in PyTorch using validation-based early stopping.
6. Perform an event-driven backtest incorporating bid-ask spreads, trading fees, and slippage.
7. Conduct systematic ablation studies (features and architectures).
8. Compute statistical tests (McNemar's test, Diebold-Mariano forecasting comparison, and bootstrap confidence intervals).
9. Output 15 publication-quality figures to `figures/`.
10. Compile the final academic paper draft incorporating all exact empirical results to `paper/paper_draft.md`.

---

## 4. Key Configurations (`config.yaml`)

- Modify target horizon $k$ under `data -> target_horizons` and `data -> default_horizon`.
- Adjust the classification stationary threshold $\theta$ under `data -> stationary_threshold`.
- Configure model structures and optimization parameters under `lstm` and `transformer`.
- Vary execution transaction fee $\phi$ and slippage impact $\sigma$ under `backtest`.

---

## 5. Statistical and Economic Significance
- Point estimates and 95% bootstrap confidence intervals are computed for macro F1-score and Net Profitability.
- McNemar's test determines if model correct classification ratios differ significantly.
- Diebold-Mariano tests evaluate forecast Brier scores to check if deep learning prediction errors are statistically smaller than linear baselines.
- The event-driven backtest compares predictive accuracy against economic returns after accounting for transaction costs.
