import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)

class MarketConditionDetector:
    """
    Rule-based Market Condition and Microstructure Regime Detector.
    Calibrates regime thresholds strictly on in-sample training data distributions
    to avoid lookahead bias and data leakage.
    """

    def __init__(
        self,
        volatility_quantile: float = 0.95,
        liquidity_low_quantile: float = 0.10,
        liquidity_high_quantile: float = 0.90,
        ofi_shock_quantile: float = 0.95,
        spread_expansion_quantile: float = 0.95,
        trend_quantile: float = 0.90
    ):
        self.volatility_quantile = volatility_quantile
        self.liquidity_low_quantile = liquidity_low_quantile
        self.liquidity_high_quantile = liquidity_high_quantile
        self.ofi_shock_quantile = ofi_shock_quantile
        self.spread_expansion_quantile = spread_expansion_quantile
        self.trend_quantile = trend_quantile

        self.thresholds: Dict[str, float] = {}
        self.is_calibrated = False

    def fit_thresholds(self, df_train: pd.DataFrame) -> Dict[str, float]:
        """
        Learns percentile thresholds strictly from training set statistics.
        """
        thresholds = {}

        # 1. Volatility threshold (rolling std of returns)
        if "mid_return_1" in df_train.columns:
            roll_vol = df_train["mid_return_1"].rolling(20, min_periods=5).std().fillna(0.0)
            thresholds["volatility_spike"] = float(np.nanpercentile(roll_vol, self.volatility_quantile * 100))
        elif "volatility" in df_train.columns:
            thresholds["volatility_spike"] = float(np.nanpercentile(df_train["volatility"], self.volatility_quantile * 100))
        else:
            thresholds["volatility_spike"] = 0.001

        # 2. Liquidity (Depth) thresholds
        if "bid_vol_1" in df_train.columns and "ask_vol_1" in df_train.columns:
            depth = df_train["bid_vol_1"] + df_train["ask_vol_1"]
            thresholds["liquidity_low"] = float(np.nanpercentile(depth, self.liquidity_low_quantile * 100))
            thresholds["liquidity_high"] = float(np.nanpercentile(depth, self.liquidity_high_quantile * 100))
        elif "depth_level_1" in df_train.columns:
            thresholds["liquidity_low"] = float(np.nanpercentile(df_train["depth_level_1"], self.liquidity_low_quantile * 100))
            thresholds["liquidity_high"] = float(np.nanpercentile(df_train["depth_level_1"], self.liquidity_high_quantile * 100))
        elif "volume" in df_train.columns:
            thresholds["liquidity_low"] = float(np.nanpercentile(df_train["volume"], self.liquidity_low_quantile * 100))
            thresholds["liquidity_high"] = float(np.nanpercentile(df_train["volume"], self.liquidity_high_quantile * 100))
        else:
            thresholds["liquidity_low"] = 100.0
            thresholds["liquidity_high"] = 5000.0

        # 3. OFI Shock threshold
        if "ofi_level_1" in df_train.columns:
            ofi_diff = df_train["ofi_level_1"].diff().abs().fillna(0.0)
            thresholds["ofi_shock"] = float(np.nanpercentile(ofi_diff, self.ofi_shock_quantile * 100))
        else:
            thresholds["ofi_shock"] = 50.0

        # 4. Spread Expansion threshold
        if "relative_spread" in df_train.columns:
            thresholds["spread_expansion"] = float(np.nanpercentile(df_train["relative_spread"], self.spread_expansion_quantile * 100))
        elif "spread_1" in df_train.columns:
            thresholds["spread_expansion"] = float(np.nanpercentile(df_train["spread_1"], self.spread_expansion_quantile * 100))
        else:
            thresholds["spread_expansion"] = 0.001

        # 5. Trend return threshold
        if "mid_price" in df_train.columns:
            ret_20 = df_train["mid_price"].pct_change(20).abs().fillna(0.0)
            thresholds["strong_trend"] = float(np.nanpercentile(ret_20, self.trend_quantile * 100))
        else:
            thresholds["strong_trend"] = 0.005

        self.thresholds = thresholds
        self.is_calibrated = True
        logger.info(f"Calibrated market condition thresholds from training data: {thresholds}")
        return thresholds

    fit_lob_thresholds = fit_thresholds

    def classify_lob_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tags each event row in the test DataFrame with boolean condition flags
        and assigns a mutually exclusive primary_regime based on hierarchical priority:
        1. REVERSAL
        2. OFI_SHOCK
        3. LOW_LIQUIDITY
        4. SPREAD_EXPANSION
        5. HIGH_VOLATILITY
        6. NORMAL
        """
        if not self.is_calibrated:
            logger.warning("MarketConditionDetector not calibrated on training set. Fitting on current input data.")
            self.fit_thresholds(df)

        df_out = df.copy()

        # Compute relevant condition features if missing
        if "volatility_20" not in df_out.columns:
            if "rolling_vol_20" in df_out.columns:
                df_out["volatility_20"] = df_out["rolling_vol_20"]
            elif "mid_return_1" in df_out.columns:
                df_out["volatility_20"] = df_out["mid_return_1"].rolling(20, min_periods=5).std().bfill()
            else:
                df_out["volatility_20"] = 0.0005

        if "total_depth_1" not in df_out.columns:
            if "bid_vol_1" in df_out.columns and "ask_vol_1" in df_out.columns:
                df_out["total_depth_1"] = df_out["bid_vol_1"] + df_out["ask_vol_1"]
            elif "depth_level_1" in df_out.columns:
                df_out["total_depth_1"] = df_out["depth_level_1"]
            elif "volume" in df_out.columns:
                df_out["total_depth_1"] = df_out["volume"]
            else:
                df_out["total_depth_1"] = 500.0

        if "ofi_diff" not in df_out.columns and "ofi_level_1" in df_out.columns:
            df_out["ofi_diff"] = df_out["ofi_level_1"].diff().abs().bfill()

        if "past_ret_20" not in df_out.columns and "mid_price" in df_out.columns:
            df_out["past_ret_20"] = df_out["mid_price"].pct_change(20).bfill()

        if "future_ret_20" not in df_out.columns and "mid_price" in df_out.columns:
            df_out["future_ret_20"] = df_out["mid_price"].pct_change(20).shift(-20).bfill()

        # 1. Individual condition flags
        df_out["cond_high_volatility"] = df_out["volatility_20"] >= self.thresholds.get("volatility_spike", 0.001)
        df_out["cond_low_liquidity"] = df_out["total_depth_1"] <= self.thresholds.get("liquidity_low", 100.0)
        df_out["cond_high_liquidity"] = df_out["total_depth_1"] >= self.thresholds.get("liquidity_high", 5000.0)
        
        spread_col = "relative_spread" if "relative_spread" in df_out.columns else "spread_1"
        df_out["cond_spread_expansion"] = df_out[spread_col] >= self.thresholds.get("spread_expansion", 0.001)
        df_out["cond_ofi_shock"] = df_out.get("ofi_diff", 0.0) >= self.thresholds.get("ofi_shock", 50.0)

        # Reversal conditions: sign of past 20-tick return opposite to future 20-tick return
        past_ret = df_out.get("past_ret_20", 0.0)
        future_ret = df_out.get("future_ret_20", 0.0)
        strong_trend_thresh = self.thresholds.get("strong_trend", 0.002)
        
        df_out["cond_strong_uptrend"] = past_ret >= strong_trend_thresh
        df_out["cond_strong_downtrend"] = past_ret <= -strong_trend_thresh
        df_out["cond_price_reversal"] = (
            ((past_ret > strong_trend_thresh * 0.5) & (future_ret < -strong_trend_thresh * 0.5)) |
            ((past_ret < -strong_trend_thresh * 0.5) & (future_ret > strong_trend_thresh * 0.5))
        )
        df_out["cond_momentum_reversal"] = df_out["cond_price_reversal"] & df_out["cond_ofi_shock"]

        # Assign mutually exclusive primary regime for tabular grouping (priority ordered)
        def assign_primary_regime(row):
            if row["cond_price_reversal"]:
                return "PRICE_REVERSAL"
            if row["cond_high_volatility"]:
                return "HIGH_VOLATILITY"
            if row["cond_ofi_shock"]:
                return "ORDER_FLOW_SHOCK"
            if row["cond_low_liquidity"]:
                return "LOW_LIQUIDITY"
            if row["cond_spread_expansion"]:
                return "SPREAD_EXPANSION"
            if row["cond_strong_uptrend"]:
                return "STRONG_UPTREND"
            if row["cond_strong_downtrend"]:
                return "STRONG_DOWNTREND"
            if row["cond_high_liquidity"]:
                return "HIGH_LIQUIDITY"
            return "NORMAL"

        df_out["primary_regime"] = df_out.apply(assign_primary_regime, axis=1)
        return df_out

    @staticmethod
    def classify_nse_conditions(
        row: pd.Series,
        prev_close: float,
        atr_pct_threshold: float = 5.0
    ) -> List[str]:
        """
        Classifies rule-based market conditions for an equity day/session.
        """
        conditions = []
        open_p = row.get("actual_open", row.get("Open", prev_close))
        high_p = row.get("actual_high", row.get("High", open_p))
        low_p = row.get("actual_low", row.get("Low", open_p))
        close_p = row.get("actual_close", row.get("Close", open_p))
        rsi = row.get("rsi_14", row.get("rsi", 50.0))

        gap_pct = ((open_p - prev_close) / prev_close) * 100.0
        intraday_ret = ((close_p - open_p) / open_p) * 100.0
        range_pct = ((high_p - low_p) / prev_close) * 100.0

        if gap_pct >= 1.0:
            conditions.append("GAP_UP")
        elif gap_pct <= -1.0:
            conditions.append("GAP_DOWN")

        if range_pct >= atr_pct_threshold:
            conditions.append("HIGH_VOLATILITY_EXPANSION")

        if rsi >= 70.0:
            if close_p > prev_close:
                conditions.append("OVERBOUGHT_CONTINUATION")
            else:
                conditions.append("OVERBOUGHT_REVERSAL")
        elif rsi <= 30.0:
            if close_p < prev_close:
                conditions.append("OVERSOLD_BREAKDOWN")
            else:
                conditions.append("OVERSOLD_BOUNCE")
        else:
            conditions.append("NEUTRAL_RSI")

        if (gap_pct > 0.5 and intraday_ret < -1.0) or (gap_pct < -0.5 and intraday_ret > 1.0):
            conditions.append("INTRADAY_REVERSAL")

        if not conditions:
            conditions.append("NORMAL")

        return conditions

MarketRegimeDetector = MarketConditionDetector
