import numpy as np
import pandas as pd
import streamlit as st
from src.data import download_ohlcv
from features import add_extra_features   # <-- relative yerine absolute

def ticker_panel(cfg_local, tickers, extra_opt=None):
    feats = cfg_local["data"]["features"]
    add_tid = bool(cfg_local["data"].get("add_ticker_id", False))
    emb_dim = int(cfg_local["model"].get("ticker_embedding_dim", 0))
    add_tid_effective = add_tid and emb_dim > 0

    Xs, Ys, closes = [], [], []
    for idx, t in enumerate(tickers):
        df = download_ohlcv(t, cfg_local["data"]["start"], cfg_local["data"]["end"], cfg_local["data"]["interval"])
        if df is None or len(df) < (cfg_local["data"]["seq_len"] + 2):
            st.warning(f"{t}: yeterli veri yok, atlanıyor.")
            continue

        df_feat = add_extra_features(df, extra_opt) if extra_opt else df.copy()

        base_cols = ["Open","High","Low","Close","Volume"]
        ta_cols = [c for c in ["SMA_7","SMA_21","VOL_20","RSI_14"] if c in df_feat.columns]
        avail = [c for c in feats if c in df_feat.columns] + ta_cols
        X = df_feat[avail].values.astype(float)

        if add_tid_effective:
            tid = np.full((len(X), 1), float(idx), dtype=float)
            X = np.concatenate([X, tid], axis=1)

        close = df_feat["Close"].values
        y = np.log(close[1:] / close[:-1])
        X = X[1:]  # return hizalaması

        closes.append(pd.Series(close[1:], index=df_feat.index[1:]))
        Xs.append(X)
        Ys.append(y)

    if not Xs:
        return None, None, None, None

    X = np.vstack(Xs)
    y = np.concatenate(Ys)
    close_concat = pd.concat(closes, axis=0)  # DatetimeIndex korunur
    last_dt = close_concat.index[-1] if len(close_concat) else None
    return X, y, close_concat, last_dt
