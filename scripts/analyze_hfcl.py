import yfinance as yf
import pandas as pd
import numpy as np

stock = yf.Ticker('HFCL.NS')
df_daily = stock.history(period='1y')
df_weekly = stock.history(period='1y', interval='1wk')

if not df_daily.empty:
    last_row = df_daily.iloc[-1]
    prev_row = df_daily.iloc[-2]
    close = last_row['Close']
    prev_close = prev_row['Close']
    chg = (close - prev_close) / prev_close * 100
    
    # 20, 50, 200 SMA
    sma20 = df_daily['Close'].rolling(20).mean().iloc[-1]
    sma50 = df_daily['Close'].rolling(50).mean().iloc[-1]
    sma200 = df_daily['Close'].rolling(200).mean().iloc[-1] if len(df_daily) >= 200 else np.nan
    
    # MACD (12, 26, 9)
    exp12 = df_daily['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df_daily['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_val = macd.iloc[-1]
    sig_val = signal.iloc[-1]
    hist_val = macd_val - sig_val
    
    # RSI 14
    delta = df_daily['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    # Volume avg
    vol_mean = df_daily['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = last_row['Volume'] / vol_mean if vol_mean > 0 else 1.0

    # ATR (14)
    tr1 = df_daily['High'] - df_daily['Low']
    tr2 = (df_daily['High'] - df_daily['Close'].shift(1)).abs()
    tr3 = (df_daily['Low'] - df_daily['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    weekly_atr = atr * np.sqrt(5)

    # Weekly Pivots from previous week
    w_last = df_weekly.iloc[-2]
    w_high = w_last['High']
    w_low = w_last['Low']
    w_close = w_last['Close']
    pp = (w_high + w_low + w_close) / 3.0
    r1 = 2 * pp - w_low
    s1 = 2 * pp - w_high
    r2 = pp + (w_high - w_low)
    s2 = pp - (w_high - w_low)

    # 60-day swing & Fibonacci
    swing_high = df_daily['High'].tail(60).max()
    swing_low = df_daily['Low'].tail(60).min()
    fib_diff = swing_high - swing_low
    fib_236 = swing_high - 0.236 * fib_diff
    fib_382 = swing_high - 0.382 * fib_diff
    fib_500 = swing_high - 0.500 * fib_diff
    fib_618 = swing_high - 0.618 * fib_diff

    print("=== HFCL.NS DETAILED TECHNICAL ANALYSIS ===")
    print(f"Last Session Date     : {df_daily.index[-1].strftime('%Y-%m-%d')}")
    print(f"Last Close Price      : INR {close:.2f} ({chg:+.2f}%)")
    print(f"Session Range (Fri)   : Low INR {last_row['Low']:.2f} - High INR {last_row['High']:.2f} (Open: INR {last_row['Open']:.2f})")
    print(f"SMA 20 / SMA 50 / 200 : {sma20:.2f} / {sma50:.2f} / {sma200:.2f}")
    print(f"RSI (14)              : {rsi:.2f}")
    print(f"MACD / Signal Line    : {macd_val:.2f} / {sig_val:.2f} (Hist: {hist_val:+.2f})")
    print(f"Volume Ratio (vs 20D) : {vol_ratio:.2f}x (Volume: {int(last_row['Volume']):,})")
    print(f"ATR (14 Daily / Weekly): INR {atr:.2f} ({(atr/close)*100:.2f}%) / +/- INR {weekly_atr:.2f}")
    print(f"60-Day Swing Range    : Low INR {swing_low:.2f} -> High INR {swing_high:.2f}")
    print(f"Weekly Pivots         : PP={pp:.2f}, S1={s1:.2f}, S2={s2:.2f}, R1={r1:.2f}, R2={r2:.2f}")
    print(f"Fibonacci Retracements: 23.6%={fib_236:.2f}, 38.2%={fib_382:.2f}, 50%={fib_500:.2f}, 61.8%={fib_618:.2f}")
    print("Recent 5 Closes       :", [round(x, 2) for x in df_daily['Close'].tail(5).tolist()])
    print("Recent 5 Volumes      :", [int(x) for x in df_daily['Volume'].tail(5).tolist()])
