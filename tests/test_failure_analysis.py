import pytest
import numpy as np
import pandas as pd
from src.failure_analysis.error_detector import ErrorDetector
from src.failure_analysis.regime_detector import MarketConditionDetector, MarketRegimeDetector
from src.failure_analysis.confidence_analysis import ConfidenceAnalyzer
from src.failure_analysis.robustness_analysis import RobustnessAnalyzer
from src.failure_analysis.statistical_tests import FailureSignificanceTester
from src.failure_analysis.case_study_generator import FailureCaseStudyGenerator

@pytest.fixture
def mock_prediction_df():
    """Generates synthetic predictions with varying confidence and accuracy."""
    n = 200
    np.random.seed(42)
    actual = np.random.choice([0, 1, 2], size=n, p=[0.25, 0.50, 0.25])
    # 70% accuracy
    pred = actual.copy()
    error_idx = np.random.choice(n, size=int(n * 0.3), replace=False)
    pred[error_idx] = (pred[error_idx] + 1) % 3
    conf = np.random.uniform(0.40, 0.95, size=n)

    df = pd.DataFrame({
        "event_id": np.arange(n),
        "mid_price": 100.0 + np.cumsum(np.random.normal(0, 0.1, n)),
        "relative_spread": np.random.exponential(0.0005, n),
        "spread_1": 0.05,
        "ofi_level_1": np.random.normal(0, 50, n),
        "depth_level_1": np.random.uniform(50, 500, n),
        "rolling_vol_20": np.random.exponential(0.0003, n),
        "past_return_20": np.random.normal(0, 0.001, n),
        "future_return_20": np.random.normal(0, 0.001, n),
        "actual": actual,
        "prediction": pred,
        "confidence": conf,
    })
    return df

@pytest.fixture
def mock_lob_train_df():
    """Generates synthetic LOB training data for threshold calibration."""
    n = 500
    np.random.seed(123)
    return pd.DataFrame({
        "rolling_vol_20": np.random.exponential(0.0002, n),
        "depth_level_1": np.random.uniform(100, 1000, n),
        "ofi_level_1": np.random.normal(0, 40, n),
        "relative_spread": np.random.exponential(0.0004, n),
        "past_return_20": np.random.normal(0, 0.0008, n),
        "mid_price": 100.0 + np.cumsum(np.random.normal(0, 0.05, n))
    })

def test_error_detector_analysis(mock_prediction_df):
    detector = ErrorDetector(high_conf_threshold=0.80, low_conf_threshold=0.50)
    analyzed = detector.analyze_predictions(mock_prediction_df)

    assert "is_error" in analyzed.columns
    assert "is_correct" in analyzed.columns
    assert "is_high_conf_error" in analyzed.columns
    assert "error_category" in analyzed.columns

    # Verify high confidence error logic
    high_conf_errs = analyzed[analyzed["is_high_conf_error"]]
    assert (high_conf_errs["confidence"] >= 0.80).all()
    assert (high_conf_errs["is_error"]).all()

    # Verify metrics summary
    summary = detector.summarize_failure_metrics(analyzed)
    assert "total_predictions" in summary
    assert "overall_accuracy" in summary
    assert "overall_error_rate" in summary
    assert summary["total_predictions"] == len(mock_prediction_df)

def test_model_divergence_comparison(mock_prediction_df):
    detector = ErrorDetector()
    df_lstm = mock_prediction_df.copy()
    df_trans = mock_prediction_df.copy()
    # Vary transformer predictions
    df_trans["prediction"] = (df_trans["prediction"] + 1) % 3

    merged = detector.compare_models(df_lstm, df_trans, on_col="event_id")
    assert "model_divergence" in merged.columns
    allowed = {"BOTH_CORRECT", "BOTH_INCORRECT", "LSTM_INCORRECT_TRANS_CORRECT", "TRANS_INCORRECT_LSTM_CORRECT"}
    assert set(merged["model_divergence"].unique()).issubset(allowed)

def test_regime_detector_lob_calibration(mock_lob_train_df, mock_prediction_df):
    detector = MarketConditionDetector()
    thresholds = detector.fit_thresholds(mock_lob_train_df)

    assert "volatility_spike" in thresholds
    assert "liquidity_low" in thresholds
    assert "ofi_shock" in thresholds
    assert "spread_expansion" in thresholds
    assert thresholds["liquidity_low"] < thresholds["liquidity_high"]

    classified = detector.classify_lob_regimes(mock_prediction_df)
    assert "cond_high_volatility" in classified.columns
    assert "cond_low_liquidity" in classified.columns
    assert "cond_price_reversal" in classified.columns
    assert "primary_regime" in classified.columns
    assert not classified["primary_regime"].isna().any()

def test_regime_detector_nse():
    detector = MarketConditionDetector()
    row = pd.Series({
        "actual_open": 105.0,
        "actual_high": 106.0,
        "actual_low": 104.0,
        "actual_close": 105.5,
        "rsi_14": 75.0
    })
    conds = detector.classify_nse_conditions(row, prev_close=100.0)
    assert "GAP_UP" in conds
    assert "OVERBOUGHT_CONTINUATION" in conds

def test_confidence_calibration(mock_prediction_df):
    detector = ErrorDetector()
    df_analyzed = detector.analyze_predictions(mock_prediction_df)

    calib_table = ConfidenceAnalyzer.compute_calibration_table(df_analyzed)
    assert len(calib_table) == 6
    assert "confidence_bin" in calib_table.columns
    assert "empirical_accuracy" in calib_table.columns
    assert "average_confidence" in calib_table.columns
    assert "prediction_count" in calib_table.columns

    ece_res = ConfidenceAnalyzer.calculate_expected_calibration_error(df_analyzed)
    assert "expected_calibration_error" in ece_res
    assert "maximum_calibration_error" in ece_res
    assert ece_res["expected_calibration_error"] >= 0.0

    high_conf_fails = ConfidenceAnalyzer.extract_high_confidence_failures(df_analyzed, threshold=0.80)
    assert (high_conf_fails["confidence"] >= 0.80).all()
    assert (high_conf_fails["is_error"]).all()

def test_robustness_analysis(mock_prediction_df):
    detector = MarketConditionDetector()
    df_regimed = detector.classify_lob_regimes(mock_prediction_df)

    error_detector = ErrorDetector()
    df_lstm = error_detector.analyze_predictions(df_regimed)
    df_trans = error_detector.analyze_predictions(df_regimed)

    df_merged = error_detector.compare_models(df_lstm, df_trans, on_col="event_id")
    for col in df_regimed.columns:
        if col not in df_merged.columns:
            df_merged[col] = df_regimed[col].values

    t1 = RobustnessAnalyzer.compare_regime_robustness(df_merged, regime_col="primary_regime")
    assert "market_regime" in t1.columns
    assert "lstm_accuracy" in t1.columns
    assert "trans_accuracy" in t1.columns
    assert "delta_accuracy" in t1.columns

def test_statistical_significance(mock_prediction_df):
    detector = MarketConditionDetector()
    df_regimed = detector.classify_lob_regimes(mock_prediction_df)
    error_detector = ErrorDetector()
    df_analyzed = error_detector.analyze_predictions(df_regimed)

    # Test odds ratio computation
    odds, ci_low, ci_high = FailureSignificanceTester.compute_odds_ratio(30, 70, 10, 90)
    assert odds > 1.0
    assert ci_low > 1.0

    # Test regime differential
    diff_res = FailureSignificanceTester.test_regime_failure_differential(
        df_analyzed,
        regime_col="primary_regime",
        target_regime="HIGH_VOLATILITY",
        baseline_regime="NORMAL"
    )
    assert "p_value" in diff_res
    assert "odds_ratio" in diff_res

    # McNemar test
    discordant_df = pd.DataFrame({
        "model_divergence": ["LSTM_INCORRECT_TRANS_CORRECT"] * 25 + ["TRANS_INCORRECT_LSTM_CORRECT"] * 10
    })
    mcnemar = FailureSignificanceTester.test_mcnemar_model_divergence(discordant_df)
    assert mcnemar["trans_only_correct"] == 25
    assert mcnemar["lstm_only_correct"] == 10
    assert mcnemar["p_value"] < 0.05

def test_case_study_generator(mock_prediction_df):
    detector = MarketConditionDetector()
    df_regimed = detector.classify_lob_regimes(mock_prediction_df)
    error_detector = ErrorDetector()
    df_lstm = error_detector.analyze_predictions(df_regimed)
    df_trans = error_detector.analyze_predictions(df_regimed)
    df_merged = error_detector.compare_models(df_lstm, df_trans, on_col="event_id")
    for col in df_regimed.columns:
        if col not in df_merged.columns:
            df_merged[col] = df_regimed[col].values

    cases = FailureCaseStudyGenerator.extract_representative_cases(df_merged, n_per_category=1)
    assert len(cases) > 0
    c_df = FailureCaseStudyGenerator.generate_case_studies_table(cases)
    assert "category" in c_df.columns
    assert "predicted_class" in c_df.columns
    assert "actual_class" in c_df.columns
    assert "scientific_explanation" in c_df.columns
