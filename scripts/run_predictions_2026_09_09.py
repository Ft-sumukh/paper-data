import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf

# Set up paths
sys.path.insert(0, os.path.abspath("."))
from src.failure_analysis import MarketConditionDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

INSTRUMENTS = {
    "MOREPENLAB": {
        "symbol": "MOREPENLAB.NS",
        "name": "Morepen Laboratories Ltd",
        "category": "Smallcap Pharma / API"
    },
    "OMAXE": {
        "symbol": "OMAXE.NS",
        "name": "Omaxe Ltd",
        "category": "Smallcap Real Estate & Infra"
    },
    "ATHERENERG": {
        "symbol": "ATHERENERG.NS",
        "name": "Ather Energy Ltd",
        "category": "Midcap EV Technology"
    },
    "HFCL": {
        "symbol": "HFCL.NS",
        "name": "HFCL Ltd",
        "category": "Telecom Equipment & Optical Fiber"
    },
    "WELCORP": {
        "symbol": "WELCORP.NS",
        "name": "Welspun Corp Ltd",
        "category": "Large-Midcap Pipe & Infrastructure"
    },
    "PCJEWELLER": {
        "symbol": "PCJEWELLER.NS",
        "name": "PC Jeweller Ltd",
        "category": "Consumer Cyclical / Gems & Jewellery"
    },
    "NIFTY50": {
        "symbol": "^NSEI",
        "name": "NIFTY 50 Benchmark Index",
        "category": "Broad Market Benchmark"
    }
}

def analyze_and_predict(target_date="2026-09-09"):
    regime_detector = MarketConditionDetector()
    results = {}

    print("\n" + "=" * 95)
    print(f"  QUANTITATIVE MODEL PREDICTION ENGINE — NSE TARGET DATE: {target_date}")
    print("=" * 95)

    for key, meta in INSTRUMENTS.items():
        symbol = meta["symbol"]
        logger.info(f"Analyzing {key} ({symbol})...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")

        if df.empty or len(df) < 25:
            logger.warning(f"Insufficient data for {symbol}")
            continue

        # Basic prices
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(latest["Close"])
        prev_close = float(prev["Close"])
        high = float(latest["High"])
        low = float(latest["Low"])
        volume = int(latest["Volume"])
        chg_pct = ((close - prev_close) / prev_close) * 100.0
        session_date = df.index[-1].strftime("%Y-%m-%d")

        # Technical Indicators
        sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
        sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
        sma200 = float(df["Close"].rolling(min(200, len(df))).mean().iloc[-1])

        # MACD (12, 26, 9)
        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1])
        sig_val = float(signal.iloc[-1])
        hist_val = macd_val - sig_val

        # RSI (14)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # ATR (14)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift(1)).abs()
        tr3 = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr / close) * 100.0

        # Bollinger Bands (20, 2)
        bb_mid = sma20
        bb_std = float(df["Close"].rolling(20).std().iloc[-1])
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std

        # Volume Profile
        vol_mean = float(df["Volume"].rolling(20).mean().iloc[-1])
        vol_ratio = float(volume / vol_mean) if vol_mean > 0 else 1.0

        # Pivot Points (Floor trader pivots)
        pivot = (high + low + close) / 3.0
        r1 = (2.0 * pivot) - low
        s1 = (2.0 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        # Failure Analysis Regime Detection
        cond_row = pd.Series({
            "actual_open": float(latest["Open"]),
            "actual_high": high,
            "actual_low": low,
            "actual_close": close,
            "rsi_14": rsi
        })
        regimes = regime_detector.classify_nse_conditions(cond_row, prev_close=prev_close, atr_pct_threshold=4.5)

        # -------------------------------------------------------------
        # Algorithmic Directional Forecasting Engine
        # -------------------------------------------------------------
        bull_score = 0
        bear_score = 0

        # Price relative to Moving Averages
        if close > sma20: bull_score += 1.5
        else: bear_score += 1.5

        if close > sma50: bull_score += 1.0
        else: bear_score += 1.0

        # MACD momentum
        if macd_val > sig_val and hist_val > 0: bull_score += 2.0
        elif macd_val < sig_val and hist_val < 0: bear_score += 2.0
        elif hist_val > 0: bull_score += 1.0
        else: bear_score += 1.0

        # RSI Overbought/Oversold & Momentum Squeeze
        if rsi > 78.0:
            bear_score += 2.0 # High overbought mean-reversion pull
            if vol_ratio > 1.8: bull_score += 1.0 # institutional momentum squeeze offset
        elif rsi > 60.0:
            bull_score += 1.5 # Healthy bullish continuation
        elif rsi < 35.0:
            bull_score += 1.5 # Oversold bounce opportunity
        elif rsi < 45.0:
            bear_score += 1.5 # Weak downward drift

        # Volume confirmation
        if chg_pct > 0 and vol_ratio > 1.2: bull_score += 1.5
        elif chg_pct < 0 and vol_ratio > 1.2: bear_score += 1.5

        # Bollinger Band squeeze or exhaustion
        if close >= bb_upper and rsi > 70:
            bear_score += 1.5 # Exhaustion at upper band
        elif close <= bb_lower:
            bull_score += 1.0 # Support at lower band

        total_score = bull_score + bear_score
        confidence = round(float(max(bull_score, bear_score) / max(1.0, total_score)), 4)
        if confidence < 0.55:
            confidence = 0.55
        if confidence > 0.88:
            confidence = 0.88 # Cap to avoid overconfidence distortion

        # Determine directional bias and targets
        if bull_score > bear_score + 1.5:
            predicted_direction = "UP"
            predicted_bias = "BULLISH_MOMENTUM_CONTINUATION"
            expected_low = round(max(s1, close - 0.7 * atr), 2)
            expected_high = round(close + 1.1 * atr, 2)
        elif bear_score > bull_score + 1.5:
            predicted_direction = "DOWN"
            predicted_bias = "CORRECTIVE_CONSOLIDATION_PULLBACK"
            expected_low = round(close - 1.1 * atr, 2)
            expected_high = round(min(r1, close + 0.7 * atr), 2)
        else:
            predicted_direction = "CONSOLIDATION"
            predicted_bias = "SIDEWAYS_RANGEBOUND_CONSOLIDATION"
            expected_low = round(s1, 2)
            expected_high = round(r1, 2)

        results[key] = {
            "symbol": symbol,
            "company": meta["name"],
            "category": meta["category"],
            "last_session_date": session_date,
            "last_close": round(close, 2),
            "last_change_pct": round(chg_pct, 2),
            "session_range": [round(low, 2), round(high, 2)],
            "indicators": {
                "sma20": round(sma20, 2),
                "sma50": round(sma50, 2),
                "sma200": round(sma200, 2),
                "rsi_14": round(rsi, 2),
                "macd": round(macd_val, 2),
                "macd_signal": round(sig_val, 2),
                "macd_histogram": round(hist_val, 2),
                "atr_14": round(atr, 2),
                "atr_pct": round(atr_pct, 2),
                "volume_ratio_20d": round(vol_ratio, 2),
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2)
            },
            "pivots": {
                "pivot": round(pivot, 2),
                "support_s1": round(s1, 2),
                "support_s2": round(s2, 2),
                "resistance_r1": round(r1, 2),
                "resistance_r2": round(r2, 2)
            },
            "prediction": {
                "target_date": target_date,
                "predicted_direction": predicted_direction,
                "predicted_bias": predicted_bias,
                "confidence_score": confidence,
                "expected_trading_range": [expected_low, expected_high],
                "active_regimes": regimes,
                "opening_volatility_risk": "HIGH" if atr_pct > 4.0 else "MODERATE"
            }
        }

    # Save to json
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"nse_predictions_{target_date.replace('-', '_')}.json")
    
    payload = {
        "generated_at": session_date,
        "target_date": target_date,
        "exchange": "NSE",
        "instrument_count": len(results),
        "predictions": results
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Successfully generated predictions for {len(results)} instruments at {out_file}")
    return payload

if __name__ == "__main__":
    analyze_and_predict(target_date="2026-09-09")
