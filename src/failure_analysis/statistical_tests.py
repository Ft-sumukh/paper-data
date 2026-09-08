import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class FailureSignificanceTester:
    """
    Econometric and statistical hypothesis testing for regime-dependent failure rates
    and model robustness differentials.
    """

    @staticmethod
    def compute_odds_ratio(a: int, b: int, c: int, d: int) -> Tuple[float, float, float]:
        """
        Computes odds ratio and 95% Woolf confidence interval with 0.5 continuity correction.
        """
        a_c = a + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else a
        b_c = b + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else b
        c_c = c + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else c
        d_c = d + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else d

        odds_ratio = (a_c * d_c) / (b_c * c_c)
        se_log_or = np.sqrt(1.0/a_c + 1.0/b_c + 1.0/c_c + 1.0/d_c)
        log_or = np.log(odds_ratio)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        return float(odds_ratio), float(ci_lower), float(ci_upper)

    @classmethod
    def test_regime_failure_differential(
        cls,
        df: pd.DataFrame,
        regime_col: str,
        target_regime: str,
        baseline_regime: str = "NORMAL",
        error_col: str = "is_error"
    ) -> Dict[str, Any]:
        """
        Performs a 2x2 contingency test (Chi-square or Fisher exact) comparing error frequency
        in target_regime vs. baseline_regime.
        Contingency matrix:
                    Error    Correct
        Target        a        b
        Baseline      c        d
        """
        df_target = df[df[regime_col] == target_regime]
        df_base = df[df[regime_col] == baseline_regime]

        n_target = len(df_target)
        n_base = len(df_base)

        if n_target < 2 or n_base < 2:
            return {
                "target_regime": target_regime,
                "baseline_regime": baseline_regime,
                "p_value": 1.0,
                "odds_ratio": 1.0,
                "ci_lower": 1.0,
                "ci_upper": 1.0,
                "is_significant": False,
                "notes": "Insufficient sample size."
            }

        a = int(df_target[error_col].sum())
        b = int(n_target - a)
        c = int(df_base[error_col].sum())
        d = int(n_base - c)

        table = [[a, b], [c, d]]

        # Chi-squared test with Yates correction
        try:
            chi2, p_val, dof, _ = stats.chi2_contingency(table, correction=True)
        except Exception:
            chi2, p_val = 0.0, 1.0

        # Odds Ratio and 95% Woolf Confidence Interval
        odds_ratio, ci_lower, ci_upper = cls.compute_odds_ratio(a, b, c, d)

        # Effect size: Cramér's V
        n_total = a + b + c + d
        cramers_v = np.sqrt(chi2 / max(1, n_total)) if n_total > 0 else 0.0

        target_err_rate = a / max(1, n_target)
        base_err_rate = c / max(1, n_base)

        return {
            "target_regime": target_regime,
            "baseline_regime": baseline_regime,
            "target_sample_size": n_target,
            "baseline_sample_size": n_base,
            "target_error_rate": round(target_err_rate, 4),
            "baseline_error_rate": round(base_err_rate, 4),
            "chi2_statistic": round(float(chi2), 4),
            "p_value": float(p_val),
            "odds_ratio": round(float(odds_ratio), 4),
            "ci_95": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
            "cramers_v": round(float(cramers_v), 4),
            "is_significant": bool(p_val < 0.05)
        }

    @staticmethod
    def test_mcnemar_model_divergence(
        df: pd.DataFrame,
        correct_col_1: str = "is_correct_lstm",
        correct_col_2: str = "is_correct_trans"
    ) -> Dict[str, Any]:
        """
        McNemar's test for paired nominal data comparing error discordance
        between LSTM (Model 1) and Transformer (Model 2).
        Contingency:
                        Model 2 Correct    Model 2 Incorrect
        Model 1 Correct         n00                n01 (b)
        Model 1 Incorrect       n10 (c)            n11
        """
        if correct_col_1 in df.columns and correct_col_2 in df.columns:
            b = int(((df[correct_col_1] == True) & (df[correct_col_2] == False)).sum())
            c = int(((df[correct_col_1] == False) & (df[correct_col_2] == True)).sum())
        elif "model_divergence" in df.columns:
            b = int((df["model_divergence"] == "TRANS_INCORRECT_LSTM_CORRECT").sum())
            c = int((df["model_divergence"] == "LSTM_INCORRECT_TRANS_CORRECT").sum())
        else:
            raise KeyError(f"Neither '{correct_col_1}' nor 'model_divergence' found in DataFrame.")

        if (b + c) == 0:
            return {
                "mcnemar_statistic": 0.0,
                "p_value": 1.0,
                "model_1_only_correct": 0,
                "model_2_only_correct": 0,
                "is_significant": False,
                "conclusion": "No discordant predictions between models."
            }

        # McNemar statistic with continuity correction: (|b - c| - 1)^2 / (b + c)
        stat = (abs(b - c) - 1.0)**2 / (b + c)
        p_val = stats.chi2.sf(stat, df=1)

        return {
            "mcnemar_statistic": round(float(stat), 4),
            "p_value": float(p_val),
            "lstm_only_correct": b,
            "trans_only_correct": c,
            "discordant_count": b + c,
            "is_significant": bool(p_val < 0.05),
            "conclusion": (
                f"Statistically significant discordance (p = {p_val:.4e}). "
                f"Transformer succeeded where LSTM failed in {c} cases vs {b} cases."
                if p_val < 0.05 else "Differences between models are not statistically significant at alpha = 0.05."
            )
        }

    @classmethod
    def test_all_regimes(
        cls, 
        df: pd.DataFrame, 
        regime_col: str = "primary_regime",
        error_col: str = "is_error",
        baseline_regime: str = "NORMAL"
    ) -> pd.DataFrame:
        """
        Runs statistical tests across all detected regimes against the baseline regime.
        """
        regimes = [r for r in df[regime_col].unique() if r != baseline_regime]
        results = []

        for reg in regimes:
            res = cls.test_regime_failure_differential(
                df=df,
                regime_col=regime_col,
                target_regime=reg,
                baseline_regime=baseline_regime,
                error_col=error_col
            )
            results.append({
                "Target Regime": res["target_regime"],
                "Baseline Regime": res["baseline_regime"],
                "Target N": res["target_sample_size"],
                "Target Err Rate": f"{res['target_error_rate']*100:.2f}%",
                "Baseline Err Rate": f"{res['baseline_error_rate']*100:.2f}%",
                "Odds Ratio": res["odds_ratio"],
                "95% CI": f"[{res['ci_95'][0]}, {res['ci_95'][1]}]",
                "p-value": f"{res['p_value']:.4e}" if res["p_value"] < 0.001 else f"{res['p_value']:.4f}",
                "Significant (p < 0.05)": "YES" if res["is_significant"] else "NO"
            })

        return pd.DataFrame(results)
