import yaml

def load_cfg(cfg_path: str, rocm_safe: bool):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if rocm_safe:
        cfg.setdefault("model", {})
        cfg.setdefault("train", {})
        cfg["model"]["num_layers"] = 1
        cfg["model"]["dropout"] = 0.0
        cfg["train"]["mixed_precision"] = False
        cfg["train"]["compile"] = False

    cfg.setdefault("logging", {}).setdefault("debug", False)
    cfg.setdefault("data", {}).setdefault("features", ["Open", "High", "Low", "Close", "Volume"])
    cfg["data"].setdefault("seq_len", 60)
    cfg.setdefault("strategy", {
        "threshold_long": 0.0005,
        "threshold_short": 0.0005,
        "transaction_cost_bps": 5,
        "hold_flat": False
    })
    cfg.setdefault("artifacts", {
        "dir": "artifacts",
        "tb_dir": "artifacts/tb",
        "ckpt_best": "artifacts/model.best.pt",
        "ckpt_last": "artifacts/model.last.pt"
    })
    cfg.setdefault("device", "auto")
    cfg.setdefault("train", {}).setdefault("gradient_clip", 1.0)
    return cfg