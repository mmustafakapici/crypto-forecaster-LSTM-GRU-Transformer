import argparse
import json
import os
import yaml
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

from .utils import set_seed, get_device
from .data import download_ohlcv, compute_log_return, build_features, time_split, scale_train_only
from .dataset import TimeSeriesDataset
from .train_eval import train_one_epoch, evaluate
import torch.nn as nn
from torch.cuda.amp import GradScaler

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--mode", type=str, choices=["train", "eval", "backtest"], default="train")
    ap.add_argument("--ddp", type=int, default=None, help="Override cfg.dist.ddp (0/1)")
    return ap.parse_args()

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_panel_data(cfg):
    feats = cfg["data"]["features"]
    add_tid = bool(cfg["data"].get("add_ticker_id", False))
    X_all, y_all = [], []
    for idx, t in enumerate(cfg["data"]["tickers"]):
        df = download_ohlcv(t, cfg["data"]["start"], cfg["data"]["end"], cfg["data"]["interval"])
        X = build_features(df, feats).astype(float)
        if add_tid:
            tid = np.full((len(X), 1), float(idx), dtype=float)  # simple ID feature
            X = np.concatenate([X, tid], axis=1)
        y = compute_log_return(df["Close"])
        X = X[1:]  # align with y
        X_all.append(X); y_all.append(y)
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y, feats + (["__ticker_id__"] if add_tid else [])

def maybe_compile(model, cfg):
    if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print(">> torch.compile enabled.")
        except Exception as e:
            print("torch.compile failed:", e)
    return model

def init_ddp_if_needed(cfg, args):
    ddp = cfg["dist"]["ddp"] if args.ddp is None else bool(args.ddp)
    if not ddp:
        return False, None, None
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE envs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = cfg["dist"].get("backend", "nccl")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    return True, local_rank, rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.ddp is not None:
        cfg["dist"]["ddp"] = bool(args.ddp)

    set_seed(cfg.get("seed", 42))

    ddp, local_rank, rank = init_ddp_if_needed(cfg, args)
    device = torch.device(f"cuda:{local_rank}") if (ddp and torch.cuda.is_available()) else get_device(cfg.get("device", "auto"))
    if ddp:
        if rank == 0:
            os.makedirs("artifacts", exist_ok=True)
    else:
        os.makedirs("artifacts", exist_ok=True)

    # 1) Data (multi-ticker panel)
    X, y, feats_out = build_panel_data(cfg)

    # 2) split & scale
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = time_split(X, y, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va); X_te = scaler.transform(X_te)
    if (not ddp) or (ddp and rank == 0):
        joblib.dump(scaler, cfg["save"]["scaler_path"])

    # 3) datasets & samplers
    seq_len = cfg["data"]["seq_len"]
    train_ds = TimeSeriesDataset(X_tr, y_tr, seq_len)
    val_ds   = TimeSeriesDataset(X_va, y_va, seq_len)
    test_ds  = TimeSeriesDataset(X_te, y_te, seq_len)

    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds, shuffle=False)
        test_sampler  = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, sampler=val_sampler)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, sampler=test_sampler)

    # 4) model
    input_size = train_ds[0][0].shape[-1]
    model = build_model_from_cfg(input_size=input_size, cfg=cfg).to(device)
    model = maybe_compile(model, cfg)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda")

    writer = None
    if ((not ddp) or (ddp and rank == 0)) and cfg.get("logging", {}).get("tensorboard_dir"):
        writer = SummaryWriter(log_dir=cfg["logging"]["tensorboard_dir"])
        print("TensorBoard dir:", cfg["logging"]["tensorboard_dir"])

    if args.mode == "train":
        best_val = float("inf")
        patience = cfg["early_stopping"]["patience"]
        min_delta = cfg["early_stopping"].get("min_delta", 0.0)
        wait = 0
        epochs = cfg["train"]["epochs"]

        for epoch in range(1, epochs + 1):
            if ddp:
                train_sampler.set_epoch(epoch)
            tr_loss = train_one_epoch(
                model.module if ddp else model,
                train_loader, optimizer, criterion, device,
                scaler=scaler,
                grad_clip=cfg["train"]["gradient_clip"],
                mixed_precision=bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda"
            )
            va_loss, va_mae, va_dir, *_ = evaluate(model.module if ddp else model, val_loader, criterion, device)

            if writer is not None:
                writer.add_scalar("loss/train", tr_loss, epoch)
                writer.add_scalar("loss/val", va_loss, epoch)
                writer.add_scalar("metric/val_mae", va_mae, epoch)
                writer.add_scalar("metric/val_dir_acc", va_dir, epoch)

            if (not ddp) or (ddp and rank == 0):
                print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} val_loss={va_loss:.6f} val_MAE={va_mae:.6f} val_dirAcc={va_dir:.3f}")
                # save last
                torch.save((model.module if ddp else model).state_dict(), cfg["save"]["last_model_path"])

                # early stopping
                if va_loss < best_val - min_delta:
                    best_val = va_loss
                    wait = 0
                    torch.save((model.module if ddp else model).state_dict(), cfg["save"]["best_model_path"])
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping.")
                        break

        if (not ddp) or (ddp and rank == 0):
            # load best and evaluate on test
            (model.module if ddp else model).load_state_dict(torch.load(cfg["save"]["best_model_path"], map_location=device))
            te_loss, te_mae, te_dir, te_preds, te_targets = evaluate(model.module if ddp else model, test_loader, criterion, device)
            metrics = {
                "val_best_loss": best_val,
                "test_loss": te_loss,
                "test_mae": te_mae,
                "test_direction_acc": te_dir,
                "n_test": len(te_targets),
            }
            with open(cfg["save"]["metrics_path"], "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print("Saved:", cfg["save"]["best_model_path"], cfg["save"]["last_model_path"], cfg["save"]["scaler_path"], cfg["save"]["metrics_path"])

        if writer is not None:
            writer.close()

    
    elif args.mode == "backtest":
        from .backtest import walk_forward
        if (not ddp) or (ddp and rank == 0):
            metrics = walk_forward(cfg)
            print(json.dumps(metrics, indent=2))
        if ddp:
            cleanup_ddp()
        return
    else:  # eval

        state = torch.load(cfg["save"]["best_model_path"], map_location=device)
        (model.module if ddp else model).load_state_dict(state)
        te_loss, te_mae, te_dir, te_preds, te_targets = evaluate(model.module if ddp else model, test_loader, criterion, device)
        metrics = {
            "test_loss": te_loss,
            "test_mae": te_mae,
            "test_direction_acc": te_dir,
            "n_test": len(te_targets),
        }
        print(json.dumps(metrics, indent=2))

    if ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main()
