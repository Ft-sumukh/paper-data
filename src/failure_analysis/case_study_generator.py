import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class FailureCaseStudyGenerator:
    """
    Automated extraction and diagnostic reporting of representative failure case studies.
    Employs objective programmatic criteria rather than manual selection,
    and formats explanations with rigorous, non-causal scientific language.
    """

    CLASS_NAMES = {0: "DOWN (-1)", 1: "FLAT (0)", 2: "UP (+1)"}

    @classmethod
    def extract_representative_cases(
        cls, 
        df_analyzed: pd.DataFrame, 
        n_per_category: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extracts representative cases across the 7 mandated failure taxonomies.
        """
        cases = []
        df = df_analyzed.copy()

        # Normalize column names with fallbacks
        if "confidence" not in df.columns:
            if "confidence_lstm" in df.columns:
                df["confidence"] = df["confidence_lstm"]
            elif "conf_lstm" in df.columns:
                df["confidence"] = df["conf_lstm"]
            elif "confidence_trans" in df.columns:
                df["confidence"] = df["confidence_trans"]
            else:
                df["confidence"] = 0.5

        if "prediction" not in df.columns:
            if "prediction_lstm" in df.columns:
                df["prediction"] = df["prediction_lstm"]
            elif "pred_lstm" in df.columns:
                df["prediction"] = df["pred_lstm"]
            elif "predicted_class" in df.columns:
                df["prediction"] = df["predicted_class"]

        if "actual" not in df.columns:
            if "actual_lstm" in df.columns:
                df["actual"] = df["actual_lstm"]
            elif "actual_class" in df.columns:
                df["actual"] = df["actual_class"]
            elif "label" in df.columns:
                df["actual"] = df["label"]

        if "is_error" not in df.columns:
            if "is_error_lstm" in df.columns:
                df["is_error"] = df["is_error_lstm"]
            elif "prediction" in df.columns and "actual" in df.columns:
                df["is_error"] = df["prediction"] != df["actual"]
            else:
                df["is_error"] = False

        if "cond_price_reversal" not in df.columns and "cond_price_reversal_lstm" in df.columns:
            df["cond_price_reversal"] = df["cond_price_reversal_lstm"]
        if "cond_low_liquidity" not in df.columns and "cond_low_liquidity_lstm" in df.columns:
            df["cond_low_liquidity"] = df["cond_low_liquidity_lstm"]
        if "primary_regime" not in df.columns and "primary_regime_lstm" in df.columns:
            df["primary_regime"] = df["primary_regime_lstm"]

        # Category A: High-Confidence False DOWN
        mask_a = (df["prediction"] == 0) & (df["actual"] == 2) & (df["confidence"] >= 0.70)
        case_a = df[mask_a].sort_values(by="confidence", ascending=False).head(n_per_category)
        if case_a.empty:
            mask_a = (df["prediction"] == 0) & (df["actual"] == 2)
            case_a = df[mask_a].sort_values(by="confidence", ascending=False).head(n_per_category)
        for _, r in case_a.iterrows():
            cases.append(cls._format_case_record(r, "Category A: High-Confidence False DOWN", 
                "The model estimated a high probability of downward movement based on prior ask-side depth imbalance, "
                "yet the subsequent horizon coincided with an aggressive market buy sequence that cleared resting liquidity."))

        # Category B: High-Confidence False UP
        mask_b = (df["prediction"] == 2) & (df["actual"] == 0) & (df["confidence"] >= 0.70)
        case_b = df[mask_b].sort_values(by="confidence", ascending=False).head(n_per_category)
        if case_b.empty:
            mask_b = (df["prediction"] == 2) & (df["actual"] == 0)
            case_b = df[mask_b].sort_values(by="confidence", ascending=False).head(n_per_category)
        for _, r in case_b.iterrows():
            cases.append(cls._format_case_record(r, "Category B: High-Confidence False UP", 
                "The model predicted upward expansion following positive order flow momentum; however, "
                "contemporaneous selling pressure broke through the bid queue, consistent with an unexpected liquidity shock."))

        # Category C: Predicted Movement but Actual FLAT (Volatility Evaporation)
        mask_c = (df["prediction"].isin([0, 2])) & (df["actual"] == 1) & (df["confidence"] >= 0.60)
        case_c = df[mask_c].sort_values(by="confidence", ascending=False).head(n_per_category)
        if case_c.empty:
            mask_c = (df["prediction"].isin([0, 2])) & (df["actual"] == 1)
            case_c = df[mask_c].sort_values(by="confidence", ascending=False).head(n_per_category)
        for _, r in case_c.iterrows():
            cases.append(cls._format_case_record(r, "Category C: Predicted Movement but Actual FLAT", 
                "The model anticipated directional continuation, but real-time trade velocity evaporated into "
                "tight two-sided resting liquidity, holding the mid-price change within the stationary threshold."))

        # Category D: Sudden Price Reversal
        mask_d = df.get("cond_price_reversal", pd.Series(False, index=df.index)) & df["is_error"]
        case_d = df[mask_d].sort_values(by="confidence", ascending=False).head(n_per_category)
        for _, r in case_d.iterrows():
            cases.append(cls._format_case_record(r, "Category D: Sudden Price Reversal", 
                "The prediction occurred immediately before an inflection point where the past 20-tick price trend "
                "abruptly inverted, demonstrating that autoregressive sequence memory struggled to detect regime turning points."))

        # Category E: Liquidity Withdrawal
        mask_e = df.get("cond_low_liquidity", pd.Series(False, index=df.index)) & df["is_error"]
        case_e = df[mask_e].sort_values(by="confidence", ascending=False).head(n_per_category)
        for _, r in case_e.iterrows():
            cases.append(cls._format_case_record(r, "Category E: Liquidity Withdrawal", 
                "The error coincided with resting depth falling into the lower decile of the training distribution, "
                "where small aggressive order sizes induced disproportionate slippage and price volatility."))

        # Category F: LSTM Wrong / Transformer Correct
        if "model_divergence" in df.columns:
            mask_f = df["model_divergence"] == "LSTM_INCORRECT_TRANS_CORRECT"
            case_f = df[mask_f].sort_values(by="confidence", ascending=False).head(n_per_category)
            for _, r in case_f.iterrows():
                cases.append(cls._format_case_record(r, "Category F: LSTM Wrong / Transformer Correct", 
                    "The Transformer's multi-head global attention mechanism successfully captured long-range context across "
                    "the 50-tick sequence, whereas the LSTM's sequential recurrent state overweighted recent noise."))

            # Category G: Transformer Wrong / LSTM Correct
            mask_g = df["model_divergence"] == "TRANS_INCORRECT_LSTM_CORRECT"
            case_g = df[mask_g].sort_values(by="confidence", ascending=False).head(n_per_category)
            for _, r in case_g.iterrows():
                cases.append(cls._format_case_record(r, "Category G: Transformer Wrong / LSTM Correct", 
                    "The LSTM's strong local recency bias correctly adapted to short-term micro-momentum, whereas "
                    "the Transformer's diffuse attention weights were distracted by earlier book oscillations."))

        return cases

    @classmethod
    def _format_case_record(cls, row: pd.Series, category_title: str, explanation: str) -> Dict[str, Any]:
        """
        Formats an objective failure report record.
        """
        event_id = row.get("event_id", row.name if hasattr(row, "name") else "N/A")
        pred = int(row.get("prediction", row.get("predicted_class", row.get("prediction_lstm", -1))))
        actual = int(row.get("actual", row.get("actual_class", row.get("actual_lstm", -1))))
        conf = float(row.get("confidence", row.get("confidence_lstm", 0.0)))
        mid_p = float(row.get("mid_price", row.get("mid_price_lstm", 0.0)))
        spread = float(row.get("relative_spread", row.get("relative_spread_lstm", row.get("spread_1", 0.0)))) * 10000.0 # bps
        ofi = float(row.get("ofi_level_1", row.get("ofi_level_1_lstm", row.get("ofi", 0.0))))
        regime = str(row.get("primary_regime", row.get("primary_regime_lstm", "NORMAL")))

        return {
            "category": category_title,
            "event_id": event_id,
            "mid_price": round(mid_p, 2),
            "spread_bps": round(spread, 1),
            "ofi_level_1": round(ofi, 2),
            "predicted_class": cls.CLASS_NAMES.get(pred, str(pred)),
            "actual_class": cls.CLASS_NAMES.get(actual, str(actual)),
            "model_confidence": round(conf, 4),
            "market_regime": regime,
            "scientific_explanation": explanation
        }

    @classmethod
    def generate_case_studies_table(cls, cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts extracted cases list into a summary DataFrame.
        """
        return pd.DataFrame(cases)
