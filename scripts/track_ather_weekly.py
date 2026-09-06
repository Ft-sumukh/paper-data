import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

WEEKLY_SETUP = {
    "symbol": "ATHERENERG.NS",
    "company": "Ather Energy Ltd",
    "week_start": "2026-09-07",
    "week_end": "2026-09-11",
    "baseline_close": 1584.00,
    "confluence_support": [1535.00, 1558.00], # 20-SMA & 23.6% Fib & Weekly PP
    "major_support_s1": 1487.50,
    "deep_support_s2": 1440.00, # 38.2% Fib
    "immediate_resistance_r1": 1640.00,
    "major_resistance_r2": 1687.50, # Weekly R1
    "swing_high_ceiling": 1744.00,
    "expected_weekly_band": [1440.00, 1728.00] # +/- 1 Weekly ATR
}

def track_ather_week():
    symbol = WEEKLY_SETUP["symbol"]
    print("\n" + "="*95)
    print(f"      ATHER ENERGY (ATHERENERG.NS) — WEEK-AHEAD DYNAMIC TRACKER")
    print(f"      Week: {WEEKLY_SETUP['week_start']} to {WEEKLY_SETUP['week_end']} | Baseline: INR {WEEKLY_SETUP['baseline_close']:.2f}")
    print("="*95)

    stock = yf.Ticker(symbol)
    df = stock.history(period="10d")

    if df.empty:
        print(f"No market data returned for {symbol}.")
        return

    # Filter for the target week
    week_start = pd.to_datetime(WEEKLY_SETUP["week_start"]).tz_localize(df.index.tz)
    recent_df = df[df.index >= (week_start - pd.Timedelta(days=3))]

    records = []
    print(f"\n{'Date':<12} | {'Open':<9} | {'High':<9} | {'Low':<9} | {'Close':<9} | {'Chg %':<8} | {'Zone Status':<28} | {'Volume'}")
    print("-" * 105)

    for dt, row in recent_df.iterrows():
        date_str = dt.strftime("%Y-%m-%d")
        close_p = row["Close"]
        open_p = row["Open"]
        high_p = row["High"]
        low_p = row["Low"]
        vol = int(row["Volume"])
        chg = ((close_p - WEEKLY_SETUP["baseline_close"]) / WEEKLY_SETUP["baseline_close"]) * 100.0

        # Assess zone status
        if low_p <= WEEKLY_SETUP["confluence_support"][1] and high_p >= WEEKLY_SETUP["confluence_support"][0]:
            zone = "INSIDE 20-SMA CONFLUENCE"
        elif close_p < WEEKLY_SETUP["confluence_support"][0]:
            zone = "BREAKDOWN (Below 20-SMA)"
        elif close_p >= WEEKLY_SETUP["major_resistance_r2"]:
            zone = "BULLISH BREAKOUT (> R1)"
        else:
            zone = "HOLDING ABOVE 20-SMA"

        print(f"{date_str:<12} | {open_p:<9.2f} | {high_p:<9.2f} | {low_p:<9.2f} | {close_p:<9.2f} | {chg:>+6.2f}% | {zone:<28} | {vol:,}")

        records.append({
            "date": date_str,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "change_from_baseline_pct": round(chg, 2),
            "zone_status": zone,
            "volume": vol
        })

    print("-" * 105)
    out_df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ather_weekly_tracker.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\n[+] Weekly progress updated in: {csv_path}\n")

if __name__ == "__main__":
    track_ather_week()
