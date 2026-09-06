import os
import sys
import json
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

def verify_predictions(predictions_file: str = "results/nse_predictions_2026_09_07.json"):
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found at {predictions_file}")
        return

    with open(predictions_file, "r") as f:
        meta = json.load(f)

    target_date = meta.get("target_trade_date", "2026-09-07")
    predictions = meta.get("predictions", {})

    print("\n" + "="*95)
    print(f"      REAL-TIME PREDICTION VERIFICATION SUITE — NSE TARGET DATE: {target_date}")
    print("="*95)

    verification_records = []

    for key, pred in predictions.items():
        symbol = pred["symbol"]
        print(f"\n[Verifying {key} ({symbol})]...")
        stock = yf.Ticker(symbol)
        df = stock.history(period="5d")

        if df.empty:
            print(f"  [!] No market data returned for {symbol}. Market may not be open yet.")
            continue

        latest_dt = df.index[-1].strftime("%Y-%m-%d")
        latest_row = df.iloc[-1]
        close_price = latest_row["Close"]
        open_price = latest_row["Open"]
        high_price = latest_row["High"]
        low_price = latest_row["Low"]
        volume = int(latest_row["Volume"])

        prev_close = pred["friday_close"]
        chg_pct = ((close_price - prev_close) / prev_close) * 100.0
        actual_direction = "UP" if chg_pct > 0.05 else ("DOWN" if chg_pct < -0.05 else "FLAT")

        # Evaluate directional prediction
        pred_dir = pred["predicted_direction"]
        if "UP" in pred_dir and actual_direction == "UP":
            direction_match = "MATCH (CORRECT)"
        elif "DOWN" in pred_dir and actual_direction == "DOWN":
            direction_match = "MATCH (CORRECT)"
        elif "CONSOLIDATION" in pred["predicted_bias"] and abs(chg_pct) < 1.5:
            direction_match = "MATCH (CONSOLIDATION)"
        else:
            direction_match = "MISS (INCORRECT)"

        # Check support/resistance respect
        s1 = pred["support_s1"]
        r1 = pred["resistance_r1"]
        tested_s1 = low_price <= s1 * 1.01
        tested_r1 = high_price >= r1 * 0.99

        print(f"  * Baseline Price (Friday Close) : INR {prev_close:.2f}")
        print(f"  * Latest Session ({latest_dt})  : Open: INR {open_price:.2f} | Close: INR {close_price:.2f} ({chg_pct:+.2f}%)")
        print(f"  * Intraday Range                : Low: INR {low_price:.2f} | High: INR {high_price:.2f}")
        print(f"  * Predicted Direction           : {pred_dir} ({pred['predicted_bias']})")
        print(f"  * Actual Outcome                : {actual_direction} ({chg_pct:+.2f}%)")
        print(f"  * Directional Evaluation        : [{direction_match}]")
        print(f"  * Key Levels Tracked            : Tested S1 (INR {s1:.2f})? {'YES' if tested_s1 else 'NO'} | Tested R1 (INR {r1:.2f})? {'YES' if tested_r1 else 'NO'}")

        verification_records.append({
            "symbol": symbol,
            "company": pred["company"],
            "verification_date": latest_dt,
            "target_date": target_date,
            "baseline_close": prev_close,
            "actual_open": round(open_price, 2),
            "actual_high": round(high_price, 2),
            "actual_low": round(low_price, 2),
            "actual_close": round(close_price, 2),
            "change_pct": round(chg_pct, 2),
            "predicted_bias": pred["predicted_bias"],
            "predicted_direction": pred_dir,
            "actual_direction": actual_direction,
            "direction_match": direction_match,
            "support_s1": s1,
            "resistance_r1": r1,
            "tested_s1": tested_s1,
            "tested_r1": tested_r1,
            "volume": volume
        })

    if verification_records:
        out_df = pd.DataFrame(verification_records)
        os.makedirs("results", exist_ok=True)
        out_csv = "results/realtime_verification_log.csv"
        out_df.to_csv(out_csv, index=False)
        print("\n" + "-"*95)
        print(f"Verification completed. Log saved to {out_csv}")
        print("-"*95 + "\n")
    else:
        print("\nNo records were processed. Ensure live data feed is accessible.\n")

if __name__ == "__main__":
    verify_predictions()
