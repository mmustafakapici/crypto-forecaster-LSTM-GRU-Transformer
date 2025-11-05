import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from src.utils import get_device
from src.models import build_model_from_cfg
from src.signals import gen_signal, pnl_from_signals

def create_seq(X, y, L: int):
    xs, ys = [], []
    for i in range(len(X) - L):
        xs.append(X[i : i + L])
        ys.append(y[i + L])
    return np.asarray(xs), np.asarray(ys)

def build_and_load_model(cfg_local, input_size, device):
    model = build_model_from_cfg(input_size=input_size, cfg=cfg_local).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model

def rollout_predict(cfg_local, X, y, split_ratios=(0.7, 0.15), close_index=None):
    L = int(cfg_local["data"]["seq_len"])
    n = len(X)
    tr_end = int(n * split_ratios[0])
    va_end = int(n * (split_ratios[0] + split_ratios[1]))
    tr_end = max(tr_end, L + 1)

    X_train, y_train = X[:tr_end], y[:tr_end]
    X_val, y_val = X[tr_end:va_end], y[tr_end:va_end]
    X_test, y_test = X[va_end:], y[va_end:]
    if len(X_test) <= L:
        raise ValueError("Test bölümü çok kısa; seq_len düşür veya tarih aralığını genişlet.")

    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)

    Xs_tr, ys_tr = create_seq(X_tr, y_train, L)
    Xs_te, ys_te = create_seq(X_te, y_test, L)

    te_dates = None
    if close_index is not None:
        start = va_end + L
        stop  = start + len(ys_te)
        if stop <= len(close_index):
            te_dates = close_index[start:stop]

    input_size = Xs_tr.shape[-1] if (Xs_tr.ndim == 3 and Xs_tr.size > 0) else Xs_te.shape[-1]
    device = get_device(cfg_local.get("device", "auto"))
    model = build_and_load_model(cfg_local, input_size, device)

    bs = 256
    preds = []
    for i in range(0, len(Xs_te), bs):
        Xb = torch.tensor(Xs_te[i:i+bs], dtype=torch.float32, device=device)
        with torch.no_grad():
            pr = model(Xb).squeeze(-1).detach().cpu().numpy()
        preds.append(pr)
    preds = np.concatenate(preds)

    mae = float(np.mean(np.abs(preds - ys_te)))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(ys_te)))

    stg = cfg_local["strategy"]
    sig = gen_signal(preds, stg["threshold_long"], stg["threshold_short"])
    pnl, equity = pnl_from_signals(ys_te, sig,
                                   transaction_cost_bps=stg["transaction_cost_bps"],
                                   hold_flat=stg["hold_flat"])

    return {
        "preds": preds,
        "targets": ys_te,
        "signals": sig,
        "returns": pnl,
        "equity": equity,
        "mae": mae,
        "dir_acc": dir_acc,
        "scaler": scaler,
        "model": model,
        "te_dates": te_dates,
    }
