import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class RobustnessAnalyzer:
    """
    Evaluates and compares the robustness of sequence models (LSTM vs. Transformer)
    under distinct market regimes and stress conditions.
    """

    @staticmethod
    def compare_regime_robustness(
        df_merged: pd.DataFrame,
        regime_col: str = "primary_regime"
    ) -> pd.DataFrame:
        """
        Calculates error rates and accuracy for LSTM and Transformer across market regimes.
        Expects columns: [regime_col, 'is_correct_lstm', 'is_correct_trans'].
        """
        required_cols = [regime_col, "is_correct_lstm", "is_correct_trans"]
        for c in required_cols:
            if c not in df_merged.columns:
                raise ValueError(f"Missing column '{c}' in merged comparison DataFrame.")

        records = []
        regimes = df_merged[regime_col].unique()

        for r in sorted(regimes):
            sub = df_merged[df_merged[regime_col] == r]
            n_events = len(sub)
            if n_events == 0:
                continue

            acc_lstm = float(sub["is_correct_lstm"].mean())
            err_lstm = 1.0 - acc_lstm

            acc_trans = float(sub["is_correct_trans"].mean())
            err_trans = 1.0 - acc_trans

            delta_acc = acc_trans - acc_lstm
            delta_err = err_trans - err_lstm

            # Determine robustness leader
            if abs(delta_acc) < 0.015:
                advantage = "COMPARABLE"
            elif delta_acc > 0:
                advantage = "TRANSFORMER_SUPERIOR"
            else:
                advantage = "LSTM_SUPERIOR"

            # Both failed check
            both_failed_pct = float((~sub["is_correct_lstm"] & ~sub["is_correct_trans"]).mean())

            records.append({
                "market_regime": r,
                "event_count": n_events,
                "lstm_accuracy": round(acc_lstm, 4),
                "lstm_error_rate": round(err_lstm, 4),
                "trans_accuracy": round(acc_trans, 4),
                "trans_error_rate": round(err_trans, 4),
                "delta_accuracy": round(delta_acc, 4),
                "both_failed_rate": round(both_failed_pct, 4),
                "robustness_advantage": advantage
            })

        df_out = pd.DataFrame(records)
        return df_out

    @staticmethod
    def evaluate_condition_breakdowns(
        df_merged: pd.DataFrame,
        condition_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluates specific boolean conditions (e.g. cond_high_volatility, cond_price_reversal).
        """
        if condition_cols is None:
            condition_cols = [
                c for c in df_merged.columns 
                if c.startswith("cond_") or c.startswith("is_") or "shock" in c or "reversal" in c
            ]

        records = []
        for cond in condition_cols:
            if cond not in df_merged.columns:
                continue
            sub = df_merged[df_merged[cond] == True]
            if len(sub) < 3:
                continue

            acc_lstm = float(sub["is_correct_lstm"].mean())
            acc_trans = float(sub["is_correct_trans"].mean())

            records.append({
                "condition": cond.replace("cond_", "").replace("_", " ").title(),
                "sample_size": len(sub),
                "lstm_error_rate": round(1.0 - acc_lstm, 4),
                "transformer_error_rate": round(1.0 - acc_trans, 4),
                "error_reduction": round((1.0 - acc_lstm) - (1.0 - acc_trans), 4),
                "superior_model": "Transformer" if acc_trans > acc_lstm else ("LSTM" if acc_lstm > acc_trans else "Tie")
            })

        return pd.DataFrame(records)
