import argparse
import numpy as np
import pandas as pd
import streamlit as st

from src.utils import get_device, set_seed
from config_loader import load_cfg
from panel import ticker_panel
from predictor import rollout_predict
from plots import plot_close_series, plot_pred_true, plot_equity
from features import compound_from_return, to_price_from_return
from ui import sidebar_controls, top_controls

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--config", default="configs/config.backtest.rocm-safe.yaml")
parser.add_argument("--rocm-safe", action="store_true")
args, _ = parser.parse_known_args()

st.set_page_config(page_title="Crypto LSTM MVP (ROCm-Safe)", layout="wide")
st.title("🪙 Crypto LSTM MVP — Streamlit Demo")

cfg = load_cfg(args.config, args.rocm_safe)
cfg_path, rocm_safe_flag, extra_opt = sidebar_controls(cfg, args)

if st.sidebar.button("Load Config"):
    cfg = load_cfg(cfg_path, rocm_safe_flag)
    st.success("Config yüklendi.")

controls = top_controls(cfg)
cfg["data"]["tickers"] = [t.strip() for t in controls["tickers"].split(",") if t.strip()]
cfg["data"]["start"] = controls["start_date"]
cfg["data"]["end"] = controls["end_date"]
cfg["data"]["features"] = [c for c in controls["features"] if c not in ("SMA_7","SMA_21","VOL_20","RSI_14")]
cfg["data"]["seq_len"] = controls["seq_len"]
cfg["strategy"]["threshold_long"]  = controls["th_long_bps"] / 1e4
cfg["strategy"]["threshold_short"] = controls["th_short_bps"] / 1e4

st.divider()
left, right = st.columns([2, 3])

with left:
    st.subheader("1) Veriyi Yükle")
    if st.button("Fetch & Prepare"):
        try:
            X, y, close_series, last_dt = ticker_panel(cfg, cfg["data"]["tickers"], extra_opt=extra_opt)
            if X is None:
                st.error("Veri bulunamadı.")
            else:
                st.session_state["X"] = X
                st.session_state["y"] = y
                st.session_state["CloseSeries"] = close_series
                st.session_state["last_dt"] = last_dt
                st.success(f"X: {X.shape}, y: {y.shape}")
        except Exception as e:
            st.exception(e)

    st.subheader("2) Predict (Rollout on Test + Multi-Horizon)")
    if st.button("Run Predict"):
        if "X" not in st.session_state:
            st.warning("Önce 'Fetch & Prepare' bas.")
        else:
            try:
                out = rollout_predict(cfg, st.session_state["X"], st.session_state["y"],
                                      close_index=st.session_state["CloseSeries"].index)
                st.session_state.update({
                    "preds": out["preds"],
                    "targets": out["targets"],
                    "returns": out["returns"],
                    "equity": out["equity"],
                    "model": out["model"],
                    "scaler": out["scaler"],
                    "te_dates": out["te_dates"],
                })

                from src.signals import gen_signal
                sig1 = gen_signal(out["preds"], cfg["strategy"]["threshold_long"], cfg["strategy"]["threshold_short"])
                st.session_state["signals"] = sig1

                last_close_all = st.session_state["CloseSeries"].iloc[-1]
                multi = {}
                for h in controls["horizons"]:
                    r_h = compound_from_return(out["preds"][-1], h)
                    multi[h] = {
                        "ret": r_h,
                        "price": to_price_from_return(last_close_all, r_h),
                        "date": (st.session_state["last_dt"] + pd.Timedelta(days=h)) if st.session_state.get("last_dt") is not None else None
                    }
                st.session_state["multi"] = multi

                st.success(f"MAE: {out['mae']:.6f} | Yön doğruluğu: {out['dir_acc']:.3f}")
            except Exception as e:
                st.exception(e)

    st.subheader("3) Mini Backtest (Test set)")
    if st.button("Run Mini Backtest"):
        if "returns" not in st.session_state:
            st.warning("Önce 'Run Predict' yap.")
        else:
            ret = st.session_state["returns"]
            sr = (np.nanmean(ret) / (np.nanstd(ret) + 1e-12)) * np.sqrt(252.0)
            st.info(f"Sharpe (test rollout): **{sr:.3f}**")

with right:
    st.subheader("Grafikler")

    if "CloseSeries" in st.session_state:
        plot_close_series(st.session_state["CloseSeries"])

    if "preds" in st.session_state and "targets" in st.session_state:
        plot_pred_true(
            st.session_state["preds"],
            st.session_state["targets"],
            dates=st.session_state.get("te_dates"),
            signals=st.session_state.get("signals"),
        )

    if "equity" in st.session_state:
        plot_equity(
            st.session_state["equity"],
            dates=st.session_state.get("te_dates"),
        )

    if "multi" in st.session_state:
        st.subheader("Horizon Tahmini (son bar referanslı approx)")
        m = st.session_state["multi"]
        cols = st.columns(len(m))
        out_mode = "Yüzdelik"
        for i, h in enumerate(sorted(m.keys())):
            with cols[i]:
                if out_mode.startswith("Yüzdelik"):
                    val = m[h]["ret"] * 100.0
                    st.metric(label=f"{h} günlük beklenen getiri", value=f"{val:+.2f}%")
                else:
                    st.metric(label=f"{h} günlük beklenen fiyat", value=f"{m[h]['price']:.2f}")
                if m[h]["date"] is not None:
                    st.caption(f"Tarih: {m[h]['date'].date()}")

set_seed(42)
device = get_device(cfg.get("device", "auto"))
st.sidebar.markdown("---")
st.sidebar.write(f"**Device:** {device.type.upper()}")