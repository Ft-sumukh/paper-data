import os
import sys
import json
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

def verify_predictions(
    predictions_file: str = "results/nse_predictions_2026_09_09.json",
    log_file: str = "results/realtime_verification_log.csv"
):
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found at {predictions_file}")
        return

    with open(predictions_file, "r") as f:
        meta = json.load(f)

    target_date = meta.get("target_date", "2026-09-09")
    predictions = meta.get("predictions", {})

    print("\n" + "=" * 95)
    print(f"      REAL-TIME PREDICTION VERIFICATION SUITE — NSE TARGET DATE: {target_date}")
    print("=" * 95)

    verification_records = []

    for key, pred in predictions.items():
        symbol = pred["symbol"]
        print(f"\n[Verifying {key} ({symbol})]...")
        stock = yf.Ticker(symbol)
        df = stock.history(period="5d")

        if df.empty:
            print(f"  [!] No market data returned for {symbol}.")
            continue

        latest_dt = df.index[-1].strftime("%Y-%m-%d")
        if latest_dt < target_date:
            print(f"  [i] Notice: Trading session for {target_date} has not closed yet. Latest date in feed is {latest_dt}.")
            print(f"      Run this script tomorrow after 15:30 IST to record actual market closes.")
            continue

        latest_row = df.iloc[-1]
        close_price = round(float(latest_row["Close"]), 2)
        open_price = round(float(latest_row["Open"]), 2)
        high_price = round(float(latest_row["High"]), 2)
        low_price = round(float(latest_row["Low"]), 2)
        volume = int(latest_row["Volume"])

        prev_close = pred["last_close"]
        chg_pct = round(((close_price - prev_close) / prev_close) * 100.0, 2)
        actual_direction = "UP" if chg_pct > 0.05 else ("DOWN" if chg_pct < -0.05 else "FLAT")

        pred_dir = pred["prediction"]["predicted_direction"]
        pred_bias = pred["prediction"]["predicted_bias"]

        # Evaluate directional prediction
        if pred_dir == "UP" and actual_direction == "UP":
            direction_match = "MATCH (CORRECT)"
        elif pred_dir == "DOWN" and actual_direction == "DOWN":
            direction_match = "MATCH (CORRECT)"
        elif "CONSOLIDATION" in pred_bias and abs(chg_pct) < 1.5:
            direction_match = "MATCH (CONSOLIDATION)"
        else:
            direction_match = "MISS (INCORRECT)"

        # Check pivot test
        s1 = pred["pivots"]["support_s1"]
        r1 = pred["pivots"]["resistance_r1"]
        tested_s1 = bool(low_price <= s1)
        tested_r1 = bool(high_price >= r1)

        print(f"  * Latest Date Evaluated: {latest_dt}")
        print(f"  * Baseline (Prior) Close: INR {prev_close:.2f}")
        print(f"  * Actual Session: Open={open_price:.2f}, High={high_price:.2f}, Low={low_price:.2f}, Close={close_price:.2f} ({chg_pct:+.2f}%)")
        print(f"  * Predicted Direction: {pred_dir} ({pred_bias})")
        print(f"  * Actual Direction:    {actual_direction}")
        print(f"  * Directional Match:   >>> {direction_match} <<<")
        print(f"  * Tested Support S1 ({s1}): {tested_s1} | Tested Resistance R1 ({r1}): {tested_r1}")

        verification_records.append({
            "symbol": symbol,
            "company": pred["company"],
            "verification_date": datetime.now().strftime("%Y-%m-%d"),
            "target_date": target_date,
            "baseline_close": prev_close,
            "actual_open": open_price,
            "actual_high": high_price,
            "actual_low": low_price,
            "actual_close": close_price,
            "change_pct": chg_pct,
            "predicted_bias": pred_bias,
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
        ver_df = pd.DataFrame(verification_records)
        print("\n" + "=" * 95)
        print("                  SUMMARY VERIFICATION TABLE")
        print("=" * 95)
        print(ver_df[["symbol", "baseline_close", "actual_close", "change_pct", "predicted_direction", "actual_direction", "direction_match"]].to_string(index=False))

        # Append or update verification log
        if os.path.exists(log_file):
            existing = pd.read_csv(log_file)
            combined = pd.concat([existing, ver_df], ignore_index=True).drop_duplicates(subset=["symbol", "target_date"], keep="last")
        else:
            combined = ver_df
        combined.to_csv(log_file, index=False)
        logger.info(f"Verification records saved to {log_file}")

if __name__ == "__main__":
    verify_predictions()
