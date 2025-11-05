import numpy as np

def gen_signal(pred_ret, thr_long=0.0, thr_short=0.0):
    """Generate long/flat/short signals from predicted returns."""
    s = np.zeros_like(pred_ret)
    s[pred_ret > thr_long] = 1.0
    s[pred_ret < -thr_short] = -1.0
    return s

def pnl_from_signals(true_ret, signals, transaction_cost_bps=0.0, hold_flat=False):
    """Compute PnL and equity curve from signals and true returns.
    Args:
        true_ret (np.ndarray): actual returns
        signals (np.ndarray): positions in {-1,0,1}
        transaction_cost_bps (float): per-change cost in basis points
        hold_flat (bool): if True, keep previous position during flat zone
    Returns:
        pnl_after_costs (np.ndarray), equity (np.ndarray)
    """
    sig = signals.copy()
    if hold_flat and len(sig) > 0:
        last = 0.0
        for i in range(len(sig)):
            if sig[i] == 0.0:
                sig[i] = last
            else:
                last = sig[i]

    # position applies to next period return
    pos_prev = np.roll(sig, 1)
    pos_prev[0] = 0.0
    pnl = pos_prev * true_ret

    change = np.abs(sig - pos_prev)
    cost = (transaction_cost_bps/10000.0) * change
    pnl_after_costs = pnl - cost
    equity = pnl_after_costs.cumsum()
    return pnl_after_costs, equity
