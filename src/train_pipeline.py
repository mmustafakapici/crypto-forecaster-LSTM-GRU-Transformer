import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

from src.utils import set_seed, get_device
from src.data import download_ohlcv, compute_log_return, build_features, time_split
from src.dataset import TimeSeriesDataset
from src.train_eval import train_one_epoch, evaluate
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

def run_train(cfg, device=None, ddp=False, rank=0, train_sampler=None, val_sampler=None, test_sampler=None):
    device = device or get_device(cfg.get("device","auto"))

    # Data
    X, y = build_panel(cfg)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = time_split(X, y, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    scaler = StandardScaler().fit(X_tr)
    import joblib, os
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, cfg["save"]["scaler_path"])
    X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va); X_te = scaler.transform(X_te)

    seq_len = cfg["data"]["seq_len"]
    train_ds = TimeSeriesDataset(X_tr, y_tr, seq_len)
    val_ds   = TimeSeriesDataset(X_va, y_va, seq_len)
    test_ds  = TimeSeriesDataset(X_te, y_te, seq_len)

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, sampler=val_sampler)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, sampler=test_sampler)

    # Model
    input_size = train_ds[0][0].shape[-1]
    model = build_model_from_cfg(input_size=input_size, cfg=cfg).to(device)

    # Optional compile
    if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            if rank == 0:
                print(">> torch.compile enabled.")
        except Exception as e:
            if rank == 0:
                print("torch.compile failed:", e)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler_amp = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda")

    writer = None
    if cfg.get("logging", {}).get("tensorboard_dir") and rank == 0:
        writer = SummaryWriter(log_dir=cfg["logging"]["tensorboard_dir"])

    best_val = float("inf")
    patience = cfg["early_stopping"]["patience"]
    min_delta = cfg["early_stopping"].get("min_delta", 0.0)
    wait = 0

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler_amp,
            grad_clip=cfg["train"]["gradient_clip"],
            mixed_precision=bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda"
        )
        va_loss, va_mae, va_dir, *_ = evaluate(model, val_loader, criterion, device)

        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", va_loss, epoch)
            writer.add_scalar("metric/val_mae", va_mae, epoch)
            writer.add_scalar("metric/val_dir_acc", va_dir, epoch)

        if rank == 0:
            print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} val_loss={va_loss:.6f} val_MAE={va_mae:.6f} val_dirAcc={va_dir:.3f}")
            import torch as _torch
            _torch.save(model.state_dict(), cfg["save"]["last_model_path"])
            if va_loss < best_val - min_delta:
                best_val = va_loss
                wait = 0
                _torch.save(model.state_dict(), cfg["save"]["best_model_path"])
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break

    if writer is not None:
        writer.close()

    # Final test
    import torch as _torch, numpy as np
    model.load_state_dict(_torch.load(cfg["save"]["best_model_path"], map_location=device))
    te_loss, te_mae, te_dir, te_preds, te_targets = evaluate(model, test_loader, criterion, device)
    metrics = {
        "val_best_loss": best_val,
        "test_loss": te_loss,
        "test_mae": te_mae,
        "test_direction_acc": te_dir,
        "n_test": len(te_targets),
    }
    with open(cfg["save"]["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if rank == 0:
        print("Saved:", cfg["save"]["best_model_path"], cfg["save"]["last_model_path"], cfg["save"]["scaler_path"], cfg["save"]["metrics_path"])
    return metrics
