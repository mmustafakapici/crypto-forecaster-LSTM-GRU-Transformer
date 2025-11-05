import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.utils import get_device
from src.data import download_ohlcv, compute_log_return, build_features, time_split
from src.dataset import TimeSeriesDataset
from src.models import build_model_from_cfg

def build_panel(cfg):
    feats = cfg["data"]["features"]
    add_tid = bool(cfg["data"].get("add_ticker_id", False))
    import numpy as np
    X_all, y_all = [], []
    for idx, t in enumerate(cfg["data"]["tickers"]):
        df = download_ohlcv(t, cfg["data"]["start"], cfg["data"]["end"], cfg["data"]["interval"])
        X = build_features(df, feats).astype(float)
        if add_tid:
            tid = np.full((len(X), 1), float(idx), dtype=float)
            X = np.concatenate([X, tid], axis=1)
        y = compute_log_return(df["Close"])
        X = X[1:]
        X_all.append(X); y_all.append(y)
    import numpy as np
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y

def run_eval(cfg, device=None):
    device = device or get_device(cfg.get("device","auto"))
    X, y = build_panel(cfg)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = time_split(X, y, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])

    scaler = StandardScaler().fit(X_tr)
    X_te = scaler.transform(X_te)

    seq_len = cfg["data"]["seq_len"]
    test_ds = TimeSeriesDataset(X_te, y_te, seq_len)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = build_model_from_cfg(input_size=test_ds[0][0].shape[-1], cfg=cfg).to(device)
    state = torch.load(cfg["save"]["best_model_path"], map_location=device)
    model.load_state_dict(state)

    import numpy as np
    model.eval(); criterion = nn.MSELoss()
    total=0; n=0; preds=[]; tgts=[]
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device); yb = yb.to(device)
            pr = model(Xb).squeeze(-1)
            loss = criterion(pr, yb)
            total += loss.item() * Xb.size(0); n += Xb.size(0)
            preds.append(pr.cpu().numpy()); tgts.append(yb.cpu().numpy())

    import numpy as np
    preds = np.concatenate(preds); tgts = np.concatenate(tgts)
    metrics = {
        "test_loss": total/max(n,1),
        "test_mae": float(np.mean(np.abs(preds-tgts))),
        "test_direction_acc": float(np.mean(np.sign(preds)==np.sign(tgts))),
        "n_test": int(len(tgts)),
    }
    print(json.dumps(metrics, indent=2))
    return metrics
