import streamlit as st

def sidebar_controls(default_cfg, args):
    st.sidebar.title("Config")
    cfg_path = st.sidebar.text_input("Config path", value=args.config)
    rocm_safe_flag = st.sidebar.checkbox(
        "ROCm-Safe",
        value=args.rocm_safe,
        help="Tek katman, dropout=0, AMP/compile kapalı."
    )

    st.sidebar.subheader("Feature Zenginleştirme")
    opt_SMA7  = st.sidebar.checkbox("SMA 7",  value=True)
    opt_SMA21 = st.sidebar.checkbox("SMA 21", value=False)
    opt_VOL20 = st.sidebar.checkbox("Volatilite 20 (σ)", value=True)
    opt_RSI14 = st.sidebar.checkbox("RSI 14", value=True)
    extra_opt = {
        "SMA_7":  opt_SMA7,
        "SMA_21": opt_SMA21,
        "VOL_20": opt_VOL20,
        "RSI_14": opt_RSI14,
    }

    st.sidebar.markdown("---")
    st.sidebar.write("**ROCm-Safe İpuçları**")
    st.sidebar.caption(
        """- `--rocm-safe` → num_layers=1, dropout=0.0, AMP/compile kapalı.
- Gerekirse CPU: `CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""`.
- gfx1030 bazen: `MIOPEN_FIND_MODE=1`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`."""
    )

    return cfg_path, rocm_safe_flag, extra_opt


def top_controls(cfg):
    col1, col2, col3 = st.columns(3)
    with col1:
        tickers = st.text_input("Tickers (virgülle)", value="BTC-USD,ETH-USD,SOL-USD")
    with col2:
        start_date = st.text_input("Start", value=cfg["data"].get("start", "2018-01-01"))
    with col3:
        end_date = st.text_input("End", value=cfg["data"].get("end", "2025-01-01"))

    features = st.multiselect(
        "Features",
        options=["Open", "High", "Low", "Close", "Volume", "SMA_7", "SMA_21", "VOL_20", "RSI_14"],
        default=cfg["data"]["features"],
    )

    seq_len = st.slider("Sequence Length (L)", min_value=10, max_value=200, value=int(cfg["data"]["seq_len"]), step=5)

    st.subheader("Tahmin Ayarları")
    colA, colB, colC = st.columns(3)
    with colA:
        horizons = st.multiselect("Horizon (gün)", options=[1, 3, 5], default=[1, 3, 5])
    with colB:
        out_mode = st.radio("Çıktı modu", options=["Yüzdelik (return)", "Fiyat"], index=0)
    with colC:
        th_long = st.slider("Long eşiği (bps)",  min_value=0, max_value=200,
                            value=int(cfg["strategy"]["threshold_long"]*1e4), step=5)
        th_short = st.slider("Short eşiği (bps)", min_value=0, max_value=200,
                             value=int(cfg["strategy"]["threshold_short"]*1e4), step=5)

    return {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "features": features,
        "seq_len": int(seq_len),
        "horizons": horizons,
        "out_mode": out_mode,
        "th_long_bps": th_long,
        "th_short_bps": th_short,
    }
