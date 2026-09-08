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

def plot_confidence_vs_error_rate(cal_df: pd.DataFrame, save_path: str):
    """Figure 1: Confidence vs. Error Rate across confidence bins."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(cal_df))
    width = 0.4

    ax1.bar(x, cal_df["prediction_count"], width=width, color="#CBD5E1", alpha=0.7, label="Prediction Count")
    ax2.plot(x, cal_df["error_rate"] * 100.0, color="#DC2626", marker="o", linewidth=2.0, label="Error Rate (%)")
    ax2.plot(x, (1.0 - cal_df["average_confidence"]) * 100.0, color="#4F46E5", linestyle="--", linewidth=1.5, label="Expected Error (1 - Conf)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(cal_df["confidence_bin"])
    ax1.set_xlabel("Model Confidence Bin")
    ax1.set_ylabel("Prediction Volume (Count)", color="#475569")
    ax2.set_ylabel("Error Rate (%)", color="#DC2626")
    ax1.set_title("Figure 1: Model Prediction Confidence vs. Empirical Error Rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 1 to {save_path}")

def plot_reliability_diagram(cal_df: pd.DataFrame, ece: float, save_path: str):
    """Figure 2: Reliability Diagram with Expected Calibration Error."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration (y = x)")
    
    # Midpoints of bins
    midpoints = (cal_df["bin_lower"] + cal_df["bin_upper"]) / 2.0
    ax.bar(midpoints, cal_df["empirical_accuracy"], width=0.08, alpha=0.6, color="#2563EB", edgecolor="#1D4ED8", label="Empirical Accuracy")
    ax.scatter(midpoints, cal_df["average_confidence"], color="#DC2626", s=50, zorder=5, label="Mean Confidence")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Figure 2: Reliability Diagram (ECE = {ece:.4f})")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 2 to {save_path}")

def plot_error_rate_by_regime(regime_df: pd.DataFrame, save_path: str):
    """Figure 3: Error rate by market regime."""
    fig, ax = plt.subplots(figsize=(9, 5))
    df_sorted = regime_df.sort_values(by="lstm_error_rate" if "lstm_error_rate" in regime_df else "error_rate", ascending=True)

    y_labels = df_sorted["market_regime"].str.replace("_", " ").str.title()
    rates = df_sorted["lstm_error_rate"] if "lstm_error_rate" in df_sorted else df_sorted["error_rate"]
    
    bars = ax.barh(y_labels, rates * 100.0, color="#475569", height=0.6)
    
    # Highlight highest error regime
    if len(bars) > 0:
        bars[-1].set_color("#DC2626")

    ax.set_xlabel("Empirical Prediction Error Rate (%)")
    ax.set_title("Figure 3: Predictive Error Rate Across Market Microstructure Regimes")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%", (w + 0.5, bar.get_y() + bar.get_height()/2), va="center", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 3 to {save_path}")

def plot_lstm_vs_transformer_robustness(robustness_df: pd.DataFrame, save_path: str):
    """Figure 4: Head-to-head LSTM vs Transformer error rates by regime."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(robustness_df))
    width = 0.35

    ax.bar(x - width/2, robustness_df["lstm_error_rate"] * 100.0, width, label="LSTM Error Rate (%)", color="#3B82F6")
    ax.bar(x + width/2, robustness_df["trans_error_rate"] * 100.0, width, label="Transformer Error Rate (%)", color="#10B981")

    ax.set_xticks(x)
    ax.set_xticklabels(robustness_df["market_regime"].str.replace("_", " ").str.title(), rotation=20, ha="right")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Figure 4: Sequence Model Robustness (LSTM vs. Transformer) by Market Regime")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 4 to {save_path}")

def plot_feature_trajectories(trajectories: Dict[str, pd.DataFrame], feature_name: str, save_path: str, title: str):
    """Figures 5-9: Trajectory of a feature across t-20 to t+20 window around predictions."""
    fig, ax = plt.subplots(figsize=(9, 5))

    styles = {
        "Correct": {"color": "#10B981", "linestyle": "-", "linewidth": 2.0},
        "All Errors": {"color": "#F59E0B", "linestyle": "--", "linewidth": 2.0},
        "High-Confidence Errors": {"color": "#DC2626", "linestyle": "-", "linewidth": 2.5}
    }

    t_lags = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])

    for label, vals in trajectories.items():
        st = styles.get(label, {"color": "gray", "linestyle": ":", "linewidth": 1.5})
        ax.plot(t_lags, vals, label=label, **st)

    ax.axvline(0, color="black", linestyle=":", alpha=0.8, label="Prediction Event (t)")
    ax.set_xlabel("Event Relative Time (t - k to t + k)")
    ax.set_ylabel(feature_name)
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved trajectory plot to {save_path}")

def plot_error_heatmap(heatmap_df: pd.DataFrame, x_label: str, y_label: str, title: str, save_path: str):
    """Figure 10: 2D Error Rate Heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_df * 100.0, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Error Rate (%)"}, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 10 to {save_path}")

def plot_nse_opening_errors(opening_df: pd.DataFrame, save_path: str):
    """Figure 11: Opening-period vs rest-of-session error rates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(opening_df))

    bars = ax.bar(x, opening_df["error_rate"] * 100.0, color=["#DC2626", "#F97316", "#FBBF24", "#3B82F6"], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(opening_df["time_window"])
    ax.set_ylabel("Prediction Error Rate (%)")
    ax.set_title("Figure 11: NSE Prediction Error Concentration (Market Open vs. Session)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width()/2, h + 0.5), ha="center", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 11 to {save_path}")

def plot_nse_condition_errors(cond_df: pd.DataFrame, save_path: str):
    """Figure 12: NSE error rates by market condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    df_sorted = cond_df.sort_values(by="error_rate", ascending=True)

    bars = ax.barh(df_sorted["condition"], df_sorted["error_rate"] * 100.0, color="#6366F1", height=0.6)
    ax.set_xlabel("Error Rate (%)")
    ax.set_title("Figure 12: NSE Model Failure Rates Across Technical & Microstructure Conditions")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%", (w + 0.5, bar.get_y() + bar.get_height()/2), va="center", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 12 to {save_path}")

def plot_high_confidence_examples(cases_df: pd.DataFrame, save_path: str):
    """Figure 13: Summary panel of high-confidence failure cases."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sub = cases_df.head(6)
    y = np.arange(len(sub))
    labels = [f"#{row.get('event_id', i)}: Pred {row.get('predicted_class', '?')} vs Act {row.get('actual_class', '?')}" for i, row in sub.iterrows()]

    confs = sub["model_confidence"] * 100.0 if "model_confidence" in sub else np.full(len(sub), 85.0)
    bars = ax.barh(y, confs, color="#EF4444", height=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Model Confidence (%)")
    ax.set_title("Figure 13: High-Confidence Failure Case Studies (Severity Inspection)")
    ax.set_xlim(0, 105)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar, (_, r) in zip(bars, sub.iterrows()):
        w = bar.get_width()
        reg = r.get("market_regime", "Regime")
        ax.annotate(f"{w:.1f}% ({reg})", (w + 1, bar.get_y() + bar.get_height()/2), va="center", fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 13 to {save_path}")

def plot_correct_vs_incorrect_distributions(df: pd.DataFrame, save_path: str):
    """Figure 14: Confidence distributions of correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    corr_conf = df[df["is_correct"]]["confidence"]
    err_conf = df[~df["is_correct"]]["confidence"]

    sns.kdeplot(corr_conf, ax=ax, label=f"Correct Predictions (Mean: {corr_conf.mean():.2f})", color="#10B981", fill=True, alpha=0.3)
    sns.kdeplot(err_conf, ax=ax, label=f"Incorrect Predictions (Mean: {err_conf.mean():.2f})", color="#EF4444", fill=True, alpha=0.3)

    ax.set_xlabel("Model Predicted Confidence (Max Probability)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 14: Density Distribution of Confidence (Correct vs. Incorrect Predictions)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved Figure 14 to {save_path}")
