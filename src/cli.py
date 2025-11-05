import argparse, yaml, os, torch
from src.utils import set_seed, get_device

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--mode", type=str, choices=["train","eval","backtest"], default="train")
    ap.add_argument("--ddp", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get("seed", 42))

    if args.mode == "train":
        from src.train_pipeline import run_train
        run_train(cfg)
    elif args.mode == "eval":
        from src.eval_pipeline import run_eval
        run_eval(cfg)
    elif args.mode == "backtest":
        from src.backtest_pipeline import run_backtest
        run_backtest(cfg)

if __name__ == "__main__":
    main()
