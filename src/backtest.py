import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from .dataset import TimeSeriesDataset
from .models import build_model_from_cfg
from .utils import get_device
from .data import download_ohlcv, compute_log_return, build_features
from .signals import gen_signal, pnl_from_signals

def _dprint(enabled: bool, *args):
    if enabled:
        print(*args)

def build_panel(cfg, debug=False):
    feats = cfg["data"]["features"]
    add_tid_cfg = bool(cfg["data"].get("add_ticker_id", False))
    emb_dim = int(cfg["model"].get("ticker_embedding_dim", 0))
    # Ticker ID sütununu YALNIZCA embedding GERÇEKTEN aktifse ekle!
    add_tid_effective = bool(add_tid_cfg and emb_dim > 0)

    Xs, Ys = [], []
    for idx, t in enumerate(cfg["data"]["tickers"]):
        df = download_ohlcv(t, cfg["data"]["start"], cfg["data"]["end"], cfg["data"]["interval"])
        X = build_features(df, feats).astype(float)
        if add_tid_effective:
            tid = np.full((len(X), 1), float(idx), dtype=float)
            X = np.concatenate([X, tid], axis=1)
        y = compute_log_return(df["Close"])
        X = X[1:]  # align (log-return yüzünden)
        Xs.append(X); Ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    _dprint(debug, f"[build_panel] feats={len(feats)} add_tid_effective={add_tid_effective} X.shape={X.shape} y.shape={y.shape}")
    return X, y

def create_seq(X, y, L):
    xs, ys = [], []
    for i in range(len(X)-L):
        xs.append(X[i:i+L]); ys.append(y[i+L])
    return np.asarray(xs), np.asarray(ys)

def sharpe_ratio(returns, periods_per_year=252):
    mu = np.nanmean(returns)
    sigma = np.nanstd(returns) + 1e-12
    return float((mu * np.sqrt(periods_per_year)) / sigma)

def _plot_and_save_backtest(outdir, returns, preds, true, signals, equity):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(); plt.plot(equity); plt.title("Equity Curve"); plt.xlabel("T"); plt.ylabel("Cumulative PnL"); plt.tight_layout(); plt.savefig(os.path.join(outdir, "equity.png")); plt.close()
    plt.figure(); plt.hist(returns, bins=50); plt.title("Strategy Returns Histogram"); plt.xlabel("Return"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(os.path.join(outdir, "returns_hist.png")); plt.close()
    plt.figure(); plt.scatter(true, preds, s=4, alpha=0.5); plt.title("Predicted vs True Returns"); plt.xlabel("True"); plt.ylabel("Pred"); plt.tight_layout(); plt.savefig(os.path.join(outdir, "pred_vs_true.png")); plt.close()


def walk_forward(cfg):
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    from .models import build_model_from_cfg
    from .utils import get_device
    from .signals import gen_signal, pnl_from_signals
    from .backtest import _plot_and_save_backtest, build_panel, create_seq, sharpe_ratio, _dprint

    device = get_device(cfg.get("device","auto"))
    outdir = cfg["backtest"]["results_dir"]
    os.makedirs(outdir, exist_ok=True)
    debug = bool(cfg.get("logging", {}).get("debug", False)) or (os.getenv("DEBUG_SHAPES","0") == "1")

    # --- Veriyi hazırla
    X, y = build_panel(cfg, debug=debug)
    L = int(cfg["data"]["seq_len"])
    n = len(X)
    n_splits = int(cfg["backtest"]["n_splits"])
    step = max(1, (n - L) // n_splits)
    splits = []
    for k in range(n_splits):
        start = k * step
        end = (k + 1) * step if k < n_splits - 1 else (n - L)
        splits.append((start, end))

    _dprint(debug, f"[walk_forward] n={n} L={L} n_splits={n_splits} step={step} splits(head)={splits[:3]}")

    all_returns, all_preds, all_true, all_signals = [], [], [], []

    for si, (s, e) in enumerate(splits):
        # --- PATCH 1: Train penceresi en az L+1 (en az 1 sequence garantisi)
        train_end = max(s + L + 1, L + 1)
        X_train, y_train = X[:train_end], y[:train_end]
        X_test,  y_test  = X[s:e+L], y[s:e+L]

        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_te = scaler.transform(X_test)

        Xs_tr, ys_tr = create_seq(X_tr, y_train, L)
        Xs_te, ys_te = create_seq(X_te, y_test, L)

        _dprint(debug, f"[split {si}] X_tr={X_tr.shape} X_te={X_te.shape} Xs_tr={getattr(Xs_tr,'shape',None)} Xs_te={getattr(Xs_te,'shape',None)}")

        # --- PATCH 2: F_seq güvenli seçimi (train boşsa test sekansının F'sini kullan)
        if isinstance(Xs_tr, np.ndarray) and Xs_tr.ndim == 3 and Xs_tr.size > 0:
            F_seq = Xs_tr.shape[-1]
        else:
            F_seq = Xs_te.shape[-1]

        model = build_model_from_cfg(input_size=F_seq, cfg=cfg).to(device)
        _dprint(debug, f"[split {si}] model.eff_input={getattr(model,'eff_input','?')} F_seq={F_seq}")

        criterion = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(),
                                 lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])

        # Kısa eğitim (her split’te)
        epochs = int(cfg["backtest"]["epochs"])
        bs = int(cfg["train"]["batch_size"])
        for ep in range(epochs):
            model.train()
            for i in range(0, len(Xs_tr), bs):
                Xb = torch.tensor(Xs_tr[i:i+bs], dtype=torch.float32, device=device)
                yb = torch.tensor(ys_tr[i:i+bs], dtype=torch.float32, device=device).unsqueeze(-1)
                optim.zero_grad()
                pr = model(Xb)
                loss = criterion(pr, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip"])
                optim.step()

        # Tahmin
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xs_te), bs):
                Xb = torch.tensor(Xs_te[i:i+bs], dtype=torch.float32, device=device)
                _dprint(debug, f"[split {si}] infer Xb.shape={tuple(Xb.shape)} model.expect={getattr(model,'eff_input','?')}")
                pr = model(Xb).squeeze(-1).cpu().numpy()
                preds.append(pr)
        preds = np.concatenate(preds) if preds else np.array([])

        # Sinyal & PnL
        st = cfg["strategy"]
        sig = gen_signal(preds, st["threshold_long"], st["threshold_short"])
        pnl, equity = pnl_from_signals(ys_te, sig,
                                       transaction_cost_bps=st["transaction_cost_bps"],
                                       hold_flat=st["hold_flat"])

        all_returns.append(pnl)
        all_preds.append(preds)
        all_true.append(ys_te)
        all_signals.append(sig)

    # Birleştir & metrikler
    all_returns = np.concatenate(all_returns) if all_returns else np.array([])
    all_preds   = np.concatenate(all_preds) if all_preds else np.array([])
    all_true    = np.concatenate(all_true) if all_true else np.array([])
    all_signals = np.concatenate(all_signals) if all_signals else np.array([])

    equity = all_returns.cumsum()
    sr = sharpe_ratio(all_returns, periods_per_year=252)
    mae = float(np.mean(np.abs(all_preds - all_true))) if all_preds.size else float("nan")
    dir_acc = float(np.mean(np.sign(all_preds) == np.sign(all_true))) if all_preds.size else float("nan")

    # Kayıtlar
    np.savetxt(os.path.join(outdir, "returns.csv"), all_returns, delimiter=",")
    np.savetxt(os.path.join(outdir, "equity.csv"), equity, delimiter=",")
    np.savetxt(os.path.join(outdir, "preds.csv"), all_preds, delimiter=",")
    np.savetxt(os.path.join(outdir, "targets.csv"), all_true, delimiter=",")
    np.savetxt(os.path.join(outdir, "signals.csv"), all_signals, delimiter=",")

    # Grafikler
    _plot_and_save_backtest(outdir, all_returns, all_preds, all_true, all_signals, equity)

    metrics = {
        "backtest_sharpe": sr,
        "backtest_mae": mae,
        "backtest_dir_acc": dir_acc,
        "n": int(len(all_true))
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
