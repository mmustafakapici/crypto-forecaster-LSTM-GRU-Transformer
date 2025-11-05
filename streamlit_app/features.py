import numpy as np
import pandas as pd

def ta_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def add_extra_features(df: pd.DataFrame, options: dict) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = np.log(out["Close"] / out["Close"].shift(1))
    if options.get("SMA_7"):  out["SMA_7"]  = out["Close"].rolling(7).mean()
    if options.get("SMA_21"): out["SMA_21"] = out["Close"].rolling(21).mean()
    if options.get("VOL_20"): out["VOL_20"] = out["ret1"].rolling(20).std()
    if options.get("RSI_14"): out["RSI_14"] = ta_rsi(out["Close"], 14)
    return out.dropna()

def signal_colors(sig_arr):
    cmap = { -1: "red", 0: "gray", 1: "limegreen" }
    return [cmap.get(int(s), "gray") for s in sig_arr]

def compound_from_return(r1: float, h: int) -> float:
    return float(np.exp(h * r1) - 1.0)

def to_price_from_return(last_close: float, r: float) -> float:
    return float(last_close * np.exp(r))