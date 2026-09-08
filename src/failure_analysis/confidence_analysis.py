import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ConfidenceAnalyzer:
    """
    Evaluates whether model confidence scores are well-calibrated
    and determines if higher confidence reliably corresponds to higher predictive accuracy.
    """

    CONFIDENCE_BINS = [
        (0.0, 0.50, "0–50%"),
        (0.50, 0.60, "50–60%"),
        (0.60, 0.70, "60–70%"),
        (0.70, 0.80, "70–80%"),
        (0.80, 0.90, "80–90%"),
        (0.90, 1.001, "90–100%")
    ]

    @classmethod
    def compute_calibration_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups predictions into standard confidence bins and calculates
        empirical accuracy, error rate, average confidence, and calibration gaps.
        Expects columns: 'confidence' (float 0..1), 'is_correct' (bool).
        """
        if "confidence" not in df.columns or "is_correct" not in df.columns:
            raise ValueError("DataFrame must contain 'confidence' and 'is_correct' columns.")

        records = []
        n_total = len(df)

        for low, high, label in cls.CONFIDENCE_BINS:
            mask = (df["confidence"] >= low) & (df["confidence"] < high)
            subset = df[mask]
            count = len(subset)

            if count > 0:
                acc = float(subset["is_correct"].mean())
                err_rate = 1.0 - acc
                avg_conf = float(subset["confidence"].mean())
                cal_gap = abs(acc - avg_conf)
                n_errors = int((~subset["is_correct"]).sum())
                n_correct = int(subset["is_correct"].sum())
            else:
                acc = 0.0
                err_rate = 0.0
                avg_conf = (low + min(high, 1.0)) / 2.0
                cal_gap = 0.0
                n_errors = 0
                n_correct = 0

            records.append({
                "confidence_bin": label,
                "bin_lower": low,
                "bin_upper": min(high, 1.0),
                "prediction_count": count,
                "count_percentage": round((count / max(1, n_total)) * 100.0, 2),
                "correct_count": n_correct,
                "error_count": n_errors,
                "empirical_accuracy": round(acc, 4),
                "error_rate": round(err_rate, 4),
                "average_confidence": round(avg_conf, 4),
                "calibration_gap": round(cal_gap, 4)
            })

        return pd.DataFrame(records)

    @classmethod
    def calculate_expected_calibration_error(cls, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE):
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        MCE = max_m |acc(B_m) - conf(B_m)|
        """
        cal_df = cls.compute_calibration_table(df)
        total_n = cal_df["prediction_count"].sum()

        if total_n == 0:
            return {"ece": 0.0, "mce": 0.0}

        weighted_gaps = (cal_df["prediction_count"] / total_n) * cal_df["calibration_gap"]
        ece = float(weighted_gaps.sum())
        mce = float(cal_df["calibration_gap"].max())

        return {
            "expected_calibration_error": round(ece, 4),
            "maximum_calibration_error": round(mce, 4),
            "is_overconfident": bool(
                cal_df[cal_df["prediction_count"] > 0]["average_confidence"].mean() > 
                cal_df[cal_df["prediction_count"] > 0]["empirical_accuracy"].mean()
            )
        }

    @classmethod
    def extract_high_confidence_failures(
        cls, 
        df: pd.DataFrame, 
        threshold: float = 0.80
    ) -> pd.DataFrame:
        """
        Isolates high-confidence failures (confidence >= threshold, but is_correct == False).
        These represent the most critical systematic failures for a quantitative model.
        """
        mask = (df["confidence"] >= threshold) & (~df["is_correct"])
        high_conf_fails = df[mask].copy().sort_values(by="confidence", ascending=False)
        return high_conf_fails
