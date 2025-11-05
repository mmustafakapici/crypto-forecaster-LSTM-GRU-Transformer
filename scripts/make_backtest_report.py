#!/usr/bin/env python
import os, json, datetime

BT_DIR = "artifacts/backtest"
OUT_MD = os.path.join(BT_DIR, "report.md")

def main():
    os.makedirs(BT_DIR, exist_ok=True)
    summary_path = os.path.join(BT_DIR, "summary.json")
    if not os.path.exists(summary_path):
        raise SystemExit("Run `make backtest` first. summary.json not found.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    lines = []
    lines.append(f"# Backtest Report ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- Sharpe: **{summary.get('backtest_sharpe'):.4f}**")
    lines.append(f"- MAE: **{summary.get('backtest_mae'):.6f}**")
    lines.append(f"- Direction Acc.: **{summary.get('backtest_dir_acc'):.3f}**")
    lines.append(f"- N: {summary.get('n')}")
    lines.append("")
    for img in ["equity.png", "returns_hist.png", "pred_vs_true.png"]:
        p = os.path.join(BT_DIR, img)
        if os.path.exists(p):
            lines.append(f"## {img.replace('_', ' ').replace('.png','').title()}")
            lines.append(f"![{img}]({img})")
            lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to: {OUT_MD}")

if __name__ == "__main__":
    main()
