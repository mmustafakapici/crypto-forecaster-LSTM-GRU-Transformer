from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def download_ohlcv(ticker: str, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        # Some tickers can return multi-index columns; take 'Close' etc. from first level
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    return df

def compute_log_return(close: pd.Series) -> np.ndarray:
    close = close.astype(float).values
    r = np.log(close[1:] / close[:-1])
    return r

def build_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].astype(float).values
    return X

def time_split(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float):
    assert 0.0 < train_ratio < 1.0
    assert 0.0 < val_ratio < 1.0
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (X[:train_end], y[:train_end]), (X[train_end:val_end], y[train_end:val_end]), (X[val_end:], y[val_end:])

def scale_train_only(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler, scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.asarray(xs), np.asarray(ys)
