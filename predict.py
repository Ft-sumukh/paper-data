import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_lob_data, split_data
from src.features import engineer_features, create_labels
from src.sequence import create_dataloaders
from src.baselines import train_logistic_regression, evaluate_predictions
from src.models import LSTMClassifier, TransformerClassifier
from src.train import train_model, predict_loader

# Import portfolio & regime modules
from src.features.returns import calculate_simple_returns, calculate_covariance_matrix
from src.regimes.detector import RollingVolatilityRegimeDetector
from src.optimization import get_optimizer

def run_prediction_demo():
    print("\n" + "="*85)
    print("      FINANCIAL MARKET PREDICTION ENGINE — INFERENCE & MODEL PREDICTIONS")
    print("="*85)

    # --------------------------------------------------------------------------
    # PART 1: Limit Order Book (LOB) Short-Horizon Price Direction Prediction
    # --------------------------------------------------------------------------
    print("\n[1/2] Loading LOB Microstructure Data & Running Price Direction Prediction Models...")
    
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Load data
    df_raw = load_lob_data(cfg)
    df_features = engineer_features(df_raw, levels_to_use=cfg["features"]["default_depth"])
    df_labeled = create_labels(
        df_features, 
        horizon=cfg["data"]["default_horizon"], 
        stationary_threshold=cfg["data"]["stationary_threshold"],
        binary=cfg["data"]["binary_classification"]
    )

    feature_cols = [c for c in df_labeled.columns if c not in ["timestamp", "mid_price", "target_return", "label"]]
    train_df, val_df, test_df = split_data(df_labeled, cfg)

    # 2. Sequence dataloaders for Neural Models
    seq_len = cfg["sequence"]["default_length"]
    train_X, train_y = train_df[feature_cols].values, train_df["label"].values
    val_X, val_y = val_df[feature_cols].values, val_df["label"].values
    test_X, test_y = test_df[feature_cols].values, test_df["label"].values

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_X, train_y, val_X, val_y, test_X, test_y,
        sequence_length=seq_len, batch_size=64
    )

    input_dim = len(feature_cols)
    num_classes = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Initialize & Train lightweight LSTM for demo
    print("  -> Training LSTM Model (2 layers, hidden=64)...")
    lstm_model = LSTMClassifier(
        input_dim=input_dim,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    train_model(
        lstm_model, train_loader, val_loader,
        epochs=3, lr=0.001, patience=2,
        device=device, save_path="results/temp_lstm_demo.pt"
    )

    # 5. Initialize & Train Transformer Model
    print("  -> Training Transformer Model (2 layers, heads=4, dim=64)...")
    trans_model = TransformerClassifier(
        input_dim=input_dim,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=num_classes,
        seq_len=seq_len,
        dropout=0.1
    ).to(device)

    train_model(
        trans_model, train_loader, val_loader,
        epochs=3, lr=0.0005, patience=2,
        device=device, save_path="results/temp_trans_demo.pt"
    )

    # 6. Generate Out-of-Sample Predictions
    print("\n  -> Generating Out-of-Sample Predictions on Test Events (Horizon k=50)...")
    y_pred_lstm, y_prob_lstm = predict_loader(lstm_model, test_loader, device=device)
    y_pred_trans, y_prob_trans = predict_loader(trans_model, test_loader, device=device)

    # Ensure UTF-8 stdout encoding on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Labels: 0: DOWN, 1: STATIONARY, 2: UP
    class_map = {0: "DOWN (-1)", 1: "STATIONARY (0)", 2: "UP (+1)"}
    symbol_map = {0: "[DOWN]", 1: "[FLAT]", 2: "[ UP ]"}

    # Align sequence indices with test dataframe
    test_aligned_df = test_df.iloc[seq_len:].reset_index(drop=True)
    n_preds = min(len(y_pred_lstm), len(test_aligned_df))

    # Display a sample of predictions across test set
    sample_indices = [5, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    sample_indices = [i for i in sample_indices if i < n_preds]

    records = []
    print("\n" + "-"*110)
    print(f"{'Event':<7} | {'Mid-Price':<10} | {'Spread (bps)':<12} | {'LSTM Signal':<12} | {'LSTM Probs [D, S, U]':<24} | {'Trans Signal':<12} | {'Actual Target':<14} | {'Match'}")
    print("-"*110)

    for idx in sample_indices:
        mid_p = test_aligned_df.loc[idx, "mid_price"]
        spread_bps = (test_aligned_df.loc[idx, "relative_spread"] if "relative_spread" in test_aligned_df.columns else 0.0005) * 10000.0
        
        pred_l = int(y_pred_lstm[idx])
        probs_l = [round(float(p), 2) for p in y_prob_lstm[idx]]
        prob_str_l = f"[{probs_l[0]:.2f}, {probs_l[1]:.2f}, {probs_l[2]:.2f}]"
        
        pred_t = int(y_pred_trans[idx])
        actual = int(test_aligned_df.loc[idx, "label"])
        
        is_correct = "MATCH" if pred_l == actual else "MISS "
        
        print(f"#{idx:<6} | ${mid_p:<9.2f} | {spread_bps:<12.1f} | {symbol_map[pred_l]}   | {prob_str_l:<24} | {symbol_map[pred_t]}   | {class_map[actual]:<14} | {is_correct}")

        records.append({
            "event_index": idx,
            "mid_price": round(mid_p, 2),
            "relative_spread_bps": round(spread_bps, 1),
            "lstm_prediction": class_map[pred_l],
            "lstm_probs_down_flat_up": prob_str_l,
            "transformer_prediction": class_map[pred_t],
            "actual_label": class_map[actual],
            "lstm_correct": (pred_l == actual)
        })

    print("-"*110)

    # Clean up temp checkpoint files
    for tmp in ["results/temp_lstm_demo.pt", "results/temp_trans_demo.pt"]:
        if os.path.exists(tmp):
            os.remove(tmp)

    # --------------------------------------------------------------------------
    # PART 2: Portfolio Market Regime & Optimal Weight Allocation Prediction
    # --------------------------------------------------------------------------
    print("\n[2/2] Running Multi-Asset Market Regime & Portfolio Allocation Predictions...")

    # Load clean multi-asset prices
    prices_path = "data/processed/prices_clean.csv"
    if os.path.exists(prices_path):
        prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        active_assets = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "TLT", "GLD"]
        active_assets = [a for a in active_assets if a in prices_df.columns]
        
        # 1. Market Volatility Regime Prediction
        if "SPY" in prices_df.columns:
            bmk_rets = calculate_simple_returns(prices_df["SPY"])
            detector = RollingVolatilityRegimeDetector(window=63)
            regime_df = detector.fit_predict(bmk_rets)
            current_vol = regime_df["realized_volatility"].iloc[-1] * 100.0
            current_regime = regime_df["regime"].iloc[-1]
            print(f"\n  > Current Benchmark (SPY) 63-Day Realized Volatility : {current_vol:.2f}%")
            print(f"  > Predicted Active Market Regime                   : [{current_regime}]")

        # 2. Next-Period Optimal Portfolio Allocation Prediction
        sub_prices = prices_df[active_assets].iloc[-252:]
        rets = calculate_simple_returns(sub_prices)
        mu = rets.mean().values * 252.0
        cov = calculate_covariance_matrix(rets, annualize=True)

        opt_mvo = get_optimizer("mean_variance", risk_free_rate=0.02)
        opt_rp = get_optimizer("risk_parity", risk_free_rate=0.02)
        res_mvo = opt_mvo.optimize(mu, cov, active_assets)
        res_rp = opt_rp.optimize(mu, cov, active_assets)

        print("\n  > PREDICTED OPTIMAL PORTFOLIO WEIGHTS FOR NEXT REBALANCE PERIOD:")
        print("  " + "-"*65)
        print(f"  {'Asset':<8} | {'Sector':<18} | {'Mean-Variance (MVO)':<20} | {'Risk Parity (ERC)'}")
        print("  " + "-"*65)
        
        sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials", 
            "JNJ": "Healthcare", "XOM": "Energy", "TLT": "Fixed Income", "GLD": "Commodities"
        }
        for a in active_assets:
            w_m = res_mvo.symbol_weights.get(a, 0.0) * 100.0
            w_r = res_rp.symbol_weights.get(a, 0.0) * 100.0
            sec = sectors.get(a, "Other")
            print(f"  {a:<8} | {sec:<18} | {w_m:>18.2f}% | {w_r:>16.2f}%")
        print("  " + "-"*65)
        print(f"  Expected Return Forecast : MVO = {res_mvo.expected_return*100:.2f}% | Risk Parity = {res_rp.expected_return*100:.2f}%")
        print(f"  Portfolio Volatility      : MVO = {res_mvo.volatility*100:.2f}% | Risk Parity = {res_rp.volatility*100:.2f}%")
        print(f"  Forecast Sharpe Ratio     : MVO = {res_mvo.sharpe_ratio:.2f}   | Risk Parity = {res_rp.sharpe_ratio:.2f}")

    # Save summary predictions CSV
    pred_df = pd.DataFrame(records)
    pred_df.to_csv("results/predictions_sample.csv", index=False)
    print("\n" + "="*85)
    print("  PREDICTIONS EXECUTED SUCCESSFULLY — Output saved to results/predictions_sample.csv")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_prediction_demo()
