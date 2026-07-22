import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Use standard professional styling
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def plot_lob_structure(save_path: str):
    """Figure 1: Limit Order Book structure diagram"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define arbitrary order book levels
    asks_p = [100.1, 100.2, 100.3, 100.4, 100.5]
    asks_v = [150, 90, 240, 110, 300]
    
    bids_p = [99.9, 99.8, 99.7, 99.6, 99.5]
    bids_v = [80, 200, 120, 250, 180]
    
    ax.barh(asks_p, asks_v, color='tomato', height=0.08, label='Ask Volume', alpha=0.8)
    ax.barh(bids_p, bids_v, color='seagreen', height=0.08, label='Bid Volume', alpha=0.8)
    
    # Mid-price and microprice lines
    ax.axhline(100.0, color='blue', linestyle='--', linewidth=1.5, label='Mid-Price (100.00)')
    ax.axhline(99.965, color='purple', linestyle='-.', linewidth=1.5, label='Microprice (99.965)')
    
    ax.set_xlabel('Volume (Shares)')
    ax.set_ylabel('Price ($)')
    ax.set_title('Limit Order Book (LOB) Depth Structure Schema')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 1 to {save_path}")

def plot_ofi_over_time(test_df: pd.DataFrame, ofi_series: pd.Series, save_path: str):
    """Figure 2: Order Flow Imbalance over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Plot mid-price
    mid_price = (test_df["ask_price_1"].values + test_df["bid_price_1"].values) / 2.0 / 10000.0
    steps = np.arange(min(500, len(mid_price)))
    mid_slice = mid_price[:len(steps)]
    ofi_slice = ofi_series.values[:len(steps)]
    
    ax1.plot(steps, mid_slice, color='black', linewidth=1.5, label='Mid Price')
    ax1.set_ylabel('Mid Price ($)')
    ax1.set_title('Short-Horizon Mid-Price and Order Flow Imbalance (OFI)')
    ax1.legend(loc='upper left')
    
    # Plot OFI
    ax2.bar(steps, ofi_slice, color=np.where(ofi_slice >= 0, 'seagreen', 'tomato'), alpha=0.7, label='OFI (L1)')
    ax2.set_xlabel('LOB Events')
    ax2.set_ylabel('OFI Value')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 2 to {save_path}")

def plot_class_distribution(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, save_path: str):
    """Figure 3: Class distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ['DOWN', 'STATIONARY', 'UP']
    
    def count_classes(y):
        counts = [np.sum(y == 0), np.sum(y == 1), np.sum(y == 2)]
        return counts
        
    train_c = count_classes(y_train)
    val_c = count_classes(y_val)
    test_c = count_classes(y_test)
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, train_c, width, label='Train', color='#4F81BD')
    ax.bar(x, val_c, width, label='Val', color='#C0504D')
    ax.bar(x + width, test_c, width, label='Test', color='#9BBB59')
    
    ax.set_ylabel('Count')
    ax.set_title('Target Class Distribution Across Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 3 to {save_path}")

def plot_model_comparison(metrics_dict: Dict[str, Dict[str, Any]], save_path: str):
    """Figure 4: Model performance comparison"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["accuracy"] for m in models]
    f1s = [metrics_dict[m]["macro_f1"] for m in models]
    aucs = [metrics_dict[m].get("auc", 0.5) for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, accuracies, width, label='Accuracy', color='#3366cc')
    ax.bar(x, f1s, width, label='Macro F1', color='#dc3912')
    ax.bar(x + width, aucs, width, label='ROC-AUC', color='#ff9900')
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Classification Performance by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 4 to {save_path}")

def plot_confusion_matrices(cms: Dict[str, np.ndarray], save_path: str):
    """Figure 5: Confusion matrices"""
    models = list(cms.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
        
    labels = ['DOWN', 'STAT', 'UP']
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        cm = cms[model_name]
        
        # Normalize CM
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)
        
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title(f'{model_name}')
        
        # Show values
        thresh = 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=9)
                
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 5 to {save_path}")

def plot_roc_curves(rocs: Dict[str, Tuple[np.ndarray, np.ndarray, float]], save_path: str):
    """Figure 6: ROC curves"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for model_name, (fpr, tpr, auc_val) in rocs.items():
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_val:.3f})', linewidth=1.5)
        
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 6 to {save_path}")

def plot_horizon_vs_performance(horizons: List[int], results: Dict[str, List[float]], save_path: str):
    """Figure 7: Prediction horizon vs performance"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model_name, f1_list in results.items():
        ax.plot(horizons, f1_list, marker='o', label=model_name, linewidth=2)
        
    ax.set_xlabel('Prediction Horizon (Events)')
    ax.set_ylabel('Macro F1-Score')
    ax.set_title('Model Performance across Prediction Horizons')
    ax.set_xticks(horizons)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 7 to {save_path}")

def plot_depth_vs_performance(depths: List[int], results: Dict[str, List[float]], save_path: str):
    """Figure 8: LOB depth vs performance"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model_name, f1_list in results.items():
        ax.plot(depths, f1_list, marker='s', label=model_name, linewidth=2)
        
    ax.set_xlabel('LOB Feature Depth (Levels)')
    ax.set_ylabel('Macro F1-Score')
    ax.set_title('Effect of Limit Order Book Depth on Prediction F1')
    ax.set_xticks(depths)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 8 to {save_path}")

def plot_seq_len_vs_performance(seq_lengths: List[int], results: Dict[str, List[float]], save_path: str):
    """Figure 9: Sequence length vs performance"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model_name, f1_list in results.items():
        ax.plot(seq_lengths, f1_list, marker='^', label=model_name, linewidth=2)
        
    ax.set_xlabel('Input Sequence Length (Lookback Events)')
    ax.set_ylabel('Macro F1-Score')
    ax.set_title('Impact of Input Sequence Length on Neural Predictors')
    ax.set_xticks(seq_lengths)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 9 to {save_path}")

def plot_fee_vs_profitability(fees: List[float], results: Dict[str, List[float]], save_path: str):
    """Figure 10: Transaction cost vs net profitability"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model_name, returns_list in results.items():
        # returns_list contains net return %
        ax.plot([f * 10000 for f in fees], [r * 100 for r in returns_list], marker='o', label=model_name, linewidth=2)
        
    ax.axhline(0.0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trading Fee (Basis Points / bps)')
    ax.set_ylabel('Net Return (%)')
    ax.set_title('Trading Net Returns under varying Transaction Costs')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 10 to {save_path}")

def plot_cumulative_returns(portfolio_dfs: Dict[str, pd.DataFrame], save_path: str):
    """Figure 11: Cumulative strategy returns"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize portfolio values to start at $100,000 (standardized)
    first_key = list(portfolio_dfs.keys())[0]
    n_points = len(portfolio_dfs[first_key])
    
    # Calculate Buy-and-Hold cumulative return of Mid Price
    mid_prices = portfolio_dfs[first_key]["mid_price"].values
    bh_cum = mid_prices / mid_prices[0] * 100000.0
    
    ax.plot(bh_cum, color='grey', linestyle=':', label='Buy & Hold Mid-Price', alpha=0.7)
    
    colors = {'Logistic Regression (OFI)': '#999999', 'Logistic Regression (Full)': '#3366cc', 'LSTM': '#dc3912', 'Transformer': '#ff9900'}
    for model_name, df in portfolio_dfs.items():
        color = colors.get(model_name, None)
        ax.plot(df["portfolio_value"].values, label=model_name, linewidth=1.5, color=color)
        
    ax.set_xlabel('Test LOB Events')
    ax.set_ylabel('Portfolio Equity ($)')
    ax.set_title('Cumulative Trading Strategy Equity Curves (Test Set)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 11 to {save_path}")

def plot_drawdown_curves(portfolio_dfs: Dict[str, pd.DataFrame], save_path: str):
    """Figure 12: Drawdown curves"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = {'Logistic Regression (OFI)': '#999999', 'Logistic Regression (Full)': '#3366cc', 'LSTM': '#dc3912', 'Transformer': '#ff9900'}
    for model_name, df in portfolio_dfs.items():
        color = colors.get(model_name, None)
        ax.fill_between(np.arange(len(df)), -df["drawdown"].values * 100.0, 0, label=model_name, alpha=0.3, color=color)
        
    ax.set_xlabel('Test LOB Events')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Historical Strategy Drawdowns (Mark-to-Market)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 12 to {save_path}")

def plot_ablation_results(ablation_scores: Dict[str, float], save_path: str):
    """Figure 13: Ablation study results"""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    scenarios = list(ablation_scores.keys())
    f1s = list(ablation_scores.values())
    
    colors = ['#4F81BD' if 'Full' in s else '#C0504D' for s in scenarios]
    
    bars = ax.barh(scenarios, f1s, color=colors, height=0.6, alpha=0.8)
    
    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                ha='left', va='center', fontsize=9)
                
    ax.set_xlabel('Macro F1-Score')
    ax.set_title('Ablation Analysis: Model F1 under Alternate Subsets')
    ax.set_xlim(0, max(f1s) + 0.1)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 13 to {save_path}")

def plot_loss_curves(train_losses: List[float], val_losses: List[float], save_path: str):
    """Figure 14: Training/validation loss curves"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o', linewidth=1.5)
    ax.plot(epochs, val_losses, label='Val Loss', color='red', marker='x', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Loss Trajectory during Model Optimization')
    ax.set_xticks(epochs)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 14 to {save_path}")

def plot_feature_importance(importance_dict: Dict[str, float], title: str, save_path: str):
    """Figure 15: Feature importance / model interpretation"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Limit to top 15 features
    top_features = list(importance_dict.keys())[:15]
    top_importances = [importance_dict[f] for f in top_features]
    
    y_pos = np.arange(len(top_features))
    
    ax.barh(y_pos, top_importances, color='teal', align='center', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel('Importance Score (F1 Degradation / Weight)')
    ax.set_title(f'Feature Importance: {title}')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 15 to {save_path}")
