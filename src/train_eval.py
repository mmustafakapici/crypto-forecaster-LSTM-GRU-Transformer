import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler  # modern API
    _HAS_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # fallback
    _HAS_TORCH_AMP = False
import numpy as np

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_clip=1.0, mixed_precision=True):
    model.train()
    total = 0.0
    n = 0
    use_amp = bool(mixed_precision and device.type == "cuda")
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device).unsqueeze(-1) if yb.ndim == 1 else yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            # torch.amp autocast signature
            with autocast(device_type="cuda", enabled=True):
                pred = model(Xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * Xb.size(0)
        n += Xb.size(0)
    return total / max(n, 1)

def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    preds, tgts = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            pr = model(Xb).squeeze(-1)
            loss = criterion(pr, yb)
            total += loss.item() * Xb.size(0)
            n += Xb.size(0)
            preds.append(pr.detach().cpu().numpy())
            tgts.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    tgts  = np.concatenate(tgts) if tgts else np.array([])
    mae = float(np.mean(np.abs(preds - tgts))) if preds.size else float("nan")
    dir_acc = float(np.mean(np.sign(preds) == np.sign(tgts))) if preds.size else float("nan")
    return total / max(n, 1), mae, dir_acc, preds, tgts
