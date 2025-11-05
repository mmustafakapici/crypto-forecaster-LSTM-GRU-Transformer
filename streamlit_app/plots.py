import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from features import signal_colors   # <-- relative yerine absolute

def plot_close_series(close_series: pd.Series):
    st.caption("Close (concat)")
    st.line_chart(pd.DataFrame({"Close": close_series.values}, index=close_series.index))

def plot_pred_true(preds, tgts, dates=None, signals=None, title="Pred vs True (Test) + Signals"):
    if signals is None:
        signals = np.zeros_like(preds)
    cols = signal_colors(signals)

    fig = plt.figure(figsize=(8, 3))
    if dates is not None and len(dates) == len(preds):
        plt.plot(dates, tgts,  label="True (r)", linewidth=1)
        plt.plot(dates, preds, label="Pred (r̂)", linewidth=1)
        plt.scatter(dates, preds, s=12, c=cols, alpha=0.6, label="Signal (L/F/S)")
    else:
        x = np.arange(len(preds))
        plt.plot(x, tgts,  label="True (r)", linewidth=1)
        plt.plot(x, preds, label="Pred (r̂)", linewidth=1)
        plt.scatter(x, preds, s=12, c=cols, alpha=0.6, label="Signal (L/F/S)")

    plt.title(title)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def plot_equity(equity, dates=None, title="Equity Curve (Test)"):
    fig = plt.figure(figsize=(8, 3))
    if dates is not None and len(dates) == len(equity):
        plt.plot(dates, equity, linewidth=1)
    else:
        plt.plot(equity, linewidth=1)
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
