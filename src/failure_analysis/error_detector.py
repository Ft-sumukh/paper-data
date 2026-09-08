import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ErrorDetector:
    """
    Automated Failure Analysis and Prediction Error Categorization Engine.
    Partitions predictions into objective diagnostic categories without manual selection:
    - All Incorrect Predictions
    - High-Confidence Incorrect Predictions (Confidence >= threshold, default 0.80)
    - Low-Confidence Incorrect Predictions (Confidence < threshold, default 0.50)
    - High-Confidence Correct Predictions
    - Cross-Model Divergence Categories (LSTM vs. Transformer)
    """

    def __init__(self, high_conf_threshold: float = 0.80, low_conf_threshold: float = 0.50):
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

    def analyze_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches a predictions DataFrame with comprehensive diagnostic failure flags.
        Expects columns: ['actual', 'prediction', 'confidence'] or class probabilities.
        """
        df_out = df.copy()

        # Ensure correct boolean match
        if "actual" in df_out.columns and "prediction" in df_out.columns:
            df_out["is_correct"] = df_out["actual"] == df_out["prediction"]
            df_out["is_error"] = ~df_out["is_correct"]
        elif "actual_class" in df_out.columns and "predicted_class" in df_out.columns:
            df_out["is_correct"] = df_out["actual_class"] == df_out["predicted_class"]
            df_out["is_error"] = ~df_out["is_correct"]
        else:
            raise ValueError("Predictions dataframe must contain actual and prediction columns.")

        # Confidence handling
        if "confidence" not in df_out.columns:
            if "probability" in df_out.columns:
                df_out["confidence"] = df_out["probability"]
            elif "prob_max" in df_out.columns:
                df_out["confidence"] = df_out["prob_max"]
            else:
                df_out["confidence"] = 0.50

        df_out["confidence"] = df_out["confidence"].astype(float)

        # Categorize confidence error tiers
        df_out["is_high_conf_error"] = df_out["is_error"] & (df_out["confidence"] >= self.high_conf_threshold)
        df_out["is_low_conf_error"] = df_out["is_error"] & (df_out["confidence"] < self.low_conf_threshold)
        df_out["is_high_conf_correct"] = df_out["is_correct"] & (df_out["confidence"] >= self.high_conf_threshold)

        # Error Category Labeling
        def categorize_error(row):
            if row["is_correct"]:
                return "CORRECT_HIGH_CONF" if row["confidence"] >= self.high_conf_threshold else "CORRECT_STANDARD"
            if row["confidence"] >= self.high_conf_threshold:
                return "HIGH_CONF_ERROR"
            elif row["confidence"] < self.low_conf_threshold:
                return "LOW_CONF_ERROR"
            else:
                return "MODERATE_CONF_ERROR"

        df_out["error_category"] = df_out.apply(categorize_error, axis=1)
        return df_out

    def compare_models(
        self, 
        df_lstm: pd.DataFrame, 
        df_trans: pd.DataFrame, 
        on_col: str = "event_id"
    ) -> pd.DataFrame:
        """
        Cross-model failure comparison between LSTM and Transformer.
        Categorizes events into:
        - BOTH_CORRECT
        - BOTH_INCORRECT
        - LSTM_INCORRECT_TRANSFORMER_CORRECT
        - TRANSFORMER_INCORRECT_LSTM_CORRECT
        """
        m_lstm = self.analyze_predictions(df_lstm)
        m_trans = self.analyze_predictions(df_trans)

        merged = pd.merge(
            m_lstm, 
            m_trans, 
            on=on_col, 
            suffixes=("_lstm", "_trans")
        )

        def determine_divergence(row):
            lstm_corr = row["is_correct_lstm"]
            trans_corr = row["is_correct_trans"]
            if lstm_corr and trans_corr:
                return "BOTH_CORRECT"
            elif not lstm_corr and not trans_corr:
                return "BOTH_INCORRECT"
            elif not lstm_corr and trans_corr:
                return "LSTM_INCORRECT_TRANS_CORRECT"
            else:
                return "TRANS_INCORRECT_LSTM_CORRECT"

        merged["model_divergence"] = merged.apply(determine_divergence, axis=1)
        return merged

    def summarize_failure_metrics(self, df_analyzed: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns summary diagnostic statistics.
        """
        total = len(df_analyzed)
        if total == 0:
            return {}

        n_errors = int(df_analyzed["is_error"].sum())
        n_high_conf_errors = int(df_analyzed["is_high_conf_error"].sum())
        n_low_conf_errors = int(df_analyzed["is_low_conf_error"].sum())
        n_high_conf_correct = int(df_analyzed["is_high_conf_correct"].sum())

        return {
            "total_predictions": total,
            "total_errors": n_errors,
            "overall_accuracy": round((total - n_errors) / total, 4),
            "overall_error_rate": round(n_errors / total, 4),
            "high_confidence_errors": n_high_conf_errors,
            "high_confidence_error_rate": round(n_high_conf_errors / max(1, n_high_conf_errors + n_high_conf_correct), 4),
            "low_confidence_errors": n_low_conf_errors,
            "high_confidence_correct": n_high_conf_correct,
            "mean_confidence": round(float(df_analyzed["confidence"].mean()), 4),
            "mean_confidence_errors": round(float(df_analyzed[df_analyzed["is_error"]]["confidence"].mean()), 4) if n_errors > 0 else 0.0,
            "mean_confidence_correct": round(float(df_analyzed[df_analyzed["is_correct"]]["confidence"].mean()), 4) if (total - n_errors) > 0 else 0.0,
        }
