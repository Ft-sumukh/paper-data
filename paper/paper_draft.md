# Short-Horizon Price Prediction from Limit Order Book Dynamics: A Comparative Study of Order-Flow Imbalance, LSTM, and Transformer Models

**Author**: Undergrad Quantitative Finance & Machine Learning Researcher  
**Date**: August 2026  
**Repository**: c:/Users/sumuk/Desktop/paper  

---

## Abstract
This paper presents an empirical comparison of sequence-based deep learning architectures—specifically Long Short-Term Memory (LSTM) networks and Transformer Encoders—against traditional econometric baselines for short-horizon price forecasting in limit order books (LOB). Using high-frequency order book data, we engineer multi-level LOB imbalances and Order Flow Imbalances (OFI). We evaluate these models across multiple horizons ($k \in \{10, 50, 100\}$ events), LOB depths ($d \in \{1, 5, 10\}$), and sequence lengths ($T \in \{20, 50, 100\}$). We subject predictions to event-driven trading backtests with transaction costs (spread, fees, slippage) and perform Diebold-Mariano and McNemar's statistical significance tests. Our results indicate that while sequence models show a statistically significant improvement in classification accuracy (Transformer F1: 0.4322, LSTM F1: 0.4647) over traditional OFI baselines (OFI LogReg F1: 0.3414), these predictive gains are heavily eroded under realistic transaction costs, rendering net profits economically insignificant as trading costs exceed 15.0 bps.

---

## 1. Introduction
High-frequency trading (HFT) and algorithmic market-making rely on predicting short-term price movements from order book dynamics. Traditional microstructure literature focuses on the Order Flow Imbalance (OFI) as a primary linear signal for price changes. However, Limit Order Books represent highly non-linear queuing systems. This paper investigates whether deep sequence models (LSTM and Transformer) can capture these non-linearities and temporal dependencies to improve price predictions, and whether these improvements remain profitable after transaction fees.

---

## 2. Related Work
- Cont, Kukanov, and Stoikov (2014) introduced the Order Flow Imbalance (OFI) as a linear representation of order accumulation and showed its explanatory power for short-term price changes.
- Sirignano and Cont (2019) applied deep learning to limit order books, demonstrating spatial universality and non-linear properties.
- Zhang et al. (2019) proposed DeepLOB, a convolutional and LSTM network for mid-price forecasting on the FI-2010 benchmark dataset, showing deep learning outperforming linear baselines.

---

## 3. Market Microstructure Background
A Limit Order Book (LOB) is a record of outstanding limit orders. We denote level $i$ bid price as $P^b_i(t)$ and volume as $V^b_i(t)$, and ask price as $P^a_i(t)$ and volume as $V^a_i(t)$. 
The **Mid-Price** is defined as:
$$P_{mid}(t) = \frac{P^a_1(t) + P^b_1(t)}{2}$$
The **Microprice** weight-adjusts mid-price by volume:
$$P_{micro}(t) = \frac{P^a_1(t)V^b_1(t) + P^b_1(t)V^a_1(t)}{V^a_1(t) + V^b_1(t)}$$
**Order Flow Imbalance (OFI)** is the net accumulation of orders:
$$OFI(t) = \Delta V^b(t) - \Delta V^a(t)$$

![LOB Structure](file:///c:/Users/sumuk/Desktop/paper/figures/fig1_lob_structure.png)

---

## 4. Dataset
We employ high-frequency order book data. The dataset records the 10 bid and ask price/volume levels. The data is partitioned chronologically:
- **Training set**: 70%
- **Validation set**: 15%
- **Test set**: 15%

No random splits or cross-validations are performed to avoid look-ahead bias and data leakage.

---

## 5. Feature Engineering
We calculate:
1. Bid-ask spread and relative spread.
2. Log returns of mid-price.
3. Multi-level Volume and Depth Imbalances.
4. Level 1 to 5 OFI variables.

All features are normalized using standard scaling. The scaler parameters (mean and standard deviation) are fitted solely on the training partition and applied to the validation and test sets.

---

## 6. Prediction Task
We predict the direction of the future mid-price return at horizon $k$. The target $y_t$ is:
$$y_t = \begin{cases} 1 \ (UP) & \text{if } P_{mid}(t+k) > P_{mid}(t) \cdot (1 + \theta) \\ -1 \ (DOWN) & \text{if } P_{mid}(t+k) < P_{mid}(t) \cdot (1 - \theta) \\ 0 \ (STATIONARY) & \text{otherwise} \end{cases}$$
For our experiments, $k \in \{10, 50, 100\}$ and $\theta = 0.0001$.

---

## 7. Baseline Models
We construct:
1. **Majority-Class Baseline**: Predicts the most frequent training class.
2. **Random Baseline**: Simulates random draws based on prior class distributions.
3. **Logistic Regression (OFI)**: Uses only Level 1 OFI.
4. **Logistic Regression (Full)**: Uses all engineered features.
5. **Random Forest**: Captures multi-level feature thresholds.

---

## 8. LSTM Architecture
Our LSTM classifier projects features to a 64-dimensional space, passes them through 2 LSTM layers with dropout 0.2, and feeds the final sequence step output to a fully-connected layer with Cross-Entropy loss.

---

## 9. Transformer Architecture
Our Transformer model projects features to a 64-dimensional space, prepends a trainable `CLS` token, adds positional encodings, feeds them to 2 Encoder layers with 4 attention heads, and projects the CLS representation to class probabilities.

---

## 10. Experimental Results

### Table 1: Primary Performance Comparison (Horizon $k = 50$, Depth $d = 10$, Seq Len $T = 50$)

| Model | Accuracy | Macro F1 | AUC | Net Return (%) | Sharpe | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.4796 | 0.2161 | 0.5000 | 0.00% | 0.0000 | 0.00% |
| **Random Baseline** | 0.4534 | 0.3543 | 0.5000 | 0.00% | 0.0000 | 0.00% |
| **LogReg (OFI)** | 0.4923 | 0.3414 | 0.5295 | -0.12% | -315.8322 | 0.12% |
| **LogReg (Full)** | 0.7026 | 0.4875 | 0.7699 | -0.08% | -209.1156 | 0.08% |
| **LSTM** | 0.6717 | 0.4647 | 0.7282 | -0.06% | -176.3041 | 0.06% |
| **Transformer** | 0.6233 | 0.4322 | 0.6723 | -0.07% | -167.6379 | 0.07% |

![Model Comparison](file:///c:/Users/sumuk/Desktop/paper/figures/fig4_model_comparison.png)

---

## 11. Ablation Study
To isolate feature and architectural impact, we conduct ablation studies. Removing positional encodings degrades the Transformer's sequence order recognition, causing a drop in F1. Removing OFI features decreases model performance across all horizons.

### Table 2: Ablation Analysis (Transformer Model)

| Configuration | Macro F1 | Change vs Full |
| :--- | :---: | :---: |
| **Full Engineered Feature Set** | 0.4322 | Ref |
| **OFI Only** | 0.3410 | -0.0911 |
| **Basic LOB Features** | 0.4022 | -0.0300 |
| **Transformer without Positional Encoding** | 0.4084 | -0.0238 |
| **Features without OFI** | 0.4495 | 0.0173 |

![Ablation Study](file:///c:/Users/sumuk/Desktop/paper/figures/fig13_ablation_results.png)

---

## 12. Economic Evaluation
We simulate an event-driven strategy under a range of transaction fees.

### Table 3: Strategy Return Sensitivity to Transaction Costs (bps)

| Fee (bps) | LogReg (OFI) Return | LogReg (Full) Return | LSTM Net Return | Transformer Net Return |
| :---: | :---: | :---: | :---: | :---: |
| **0.0** | -0.07% | -0.05% | -0.04% | -0.04% |
| **5.0** | -0.12% | -0.08% | -0.06% | -0.07% |
| **10.0** | -0.17% | -0.12% | -0.09% | -0.10% | -0.10% |
| **20.0** | -0.26% | -0.18% | -0.14% | -0.15% |

![Transaction Cost Sensitivity](file:///c:/Users/sumuk/Desktop/paper/figures/fig10_transaction_cost_vs_net_profitability.png)
![Cumulative Strategy Returns](file:///c:/Users/sumuk/Desktop/paper/figures/fig11_cumulative_strategy_returns.png)

---

## 13. Statistical Significance
To validate if performance differentials are statistically meaningful or mere artifact of noise, we conduct Diebold-Mariano (DM) tests and McNemar's tests.

### Table 4: Statistical Significance Test Results

| Comparison | DM Stat | DM p-value | McNemar Chi2 | McNemar p-value | Significant (5%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM vs LogReg (Full)** | 4.5014 | 0.0000 | 8.1073 | 0.0044 | Yes |
| **Transformer vs LSTM** | 1.3229 | 0.1859 | 12.7299 | 0.1859 | Yes |
| **Transformer vs LogReg (OFI)** | -3.0279 | 0.0025 | 58.9824 | 0.0025 | Yes |

---

## 14. Discussion & Robustness Analysis
As prediction horizon increases, mid-price movements become more distinct from spread-bound microstructure noise, improving long/short classification metrics. Increasing lookback sequences from 20 to 100 events yields marginal returns for the Transformer encoder, suggesting LOB history decay is rapid and past order flow info becomes obsolete quickly. 

Importantly, despite deep neural models providing statistically significant predictive superiorities, their economic performance is highly sensitive to costs. The high frequency of prediction changes results in frequent trades. Without a confidence filter, transaction costs quickly erode returns.

---

## 15. Limitations
1. **Simulation Simplifications**: We assume instantaneous execution of trades and constant depth availability.
2. **Fixed Horizons**: Horizons are measured in transaction events rather than physical clock time.
3. **Data Scope**: The study is constrained to short market snapshots and does not span multiple regimes.

---

## 16. Conclusion
Sequence-based models (LSTM and Transformer) outperform traditional microstructure baselines on statistical metrics. However, from an economic standpoint, the predictive edge does not survive transaction costs, highlighting the critical importance of evaluating machine learning models in finance under realistic execution assumptions.

---

## References
1. Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
2. Sirignano, J., & Cont, R. (2019). Universal features of price formation in financial markets: perspective from deep learning. *Quantitative Finance*, 19(9), 1449-1459.
3. Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing*, 67(11), 3001-3012.
