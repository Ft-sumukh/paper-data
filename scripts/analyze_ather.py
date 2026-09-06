import yfinance as yf
import pandas as pd
import numpy as np

t = 'ATHERENERG.NS'
stock = yf.Ticker(t)
df = stock.history(period='6mo')

if not df.empty:
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    close = last_row['Close']
    prev_close = prev_row['Close']
    chg = (close - prev_close) / prev_close * 100
    
    # 20, 50 SMA
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else np.nan
    
    # MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_val = macd.iloc[-1]
    sig_val = signal.iloc[-1]
    hist_val = macd_val - sig_val
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    # Volume avg
    vol_mean = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = last_row['Volume'] / vol_mean if vol_mean > 0 else 1.0

    # ATR (14)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    # 20-day Support / Resistance
    last20 = df.tail(20)
    res = last20['High'].max()
    sup = last20['Low'].min()

    print(f"=== {t} ===")
    print(f"Last Session Date   : {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Last Close Price    : INR {close:.2f} ({chg:+.2f}%)")
    print(f"Session Range       : Low INR {last_row['Low']:.2f} - High INR {last_row['High']:.2f} (Open: {last_row['Open']:.2f})")
    print(f"SMA 20 / SMA 50     : {sma20:.2f} / {sma50:.2f}")
    print(f"RSI (14)            : {rsi:.2f}")
    print(f"MACD / Signal       : {macd_val:.2f} / {sig_val:.2f} (Hist: {hist_val:+.2f})")
    print(f"Volume Ratio (vs 20D): {vol_ratio:.2f}x (Vol: {int(last_row['Volume']):,})")
    print(f"ATR (14) Volatility : INR {atr:.2f} ({(atr/close)*100:.2f}%)")
    print(f"20-Day Range [S - R]: INR {sup:.2f} - INR {res:.2f}")
    print("Recent 5 Closes     :", [round(x, 2) for x in df['Close'].tail(5).tolist()])
    print("Recent 5 Volumes    :", [int(x) for x in df['Volume'].tail(5).tolist()])
