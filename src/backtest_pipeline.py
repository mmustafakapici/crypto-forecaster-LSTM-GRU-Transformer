import json
from src.backtest import walk_forward
from src.utils import get_device

def run_backtest(cfg):
    metrics = walk_forward(cfg)
    print(json.dumps(metrics, indent=2))
    return metrics
