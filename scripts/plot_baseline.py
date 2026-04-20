"""
Tier-1 ablation baseline plots.

Reads all JSONL files in logs/baseline/ and produces a 2×2 figure:
  (a) Smoothed total reward over episodes (mean ± std across seeds)
  (b) Success rate (rolling 50-episode window)
  (c) Collision rate (rolling 50-episode window)
  (d) Order entropy over episodes

Usage:
    python3 scripts/plot_baseline.py
    python3 scripts/plot_baseline.py --log-dir logs/baseline --out figures/baseline.png
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Colours / labels ──────────────────────────────────────────────────────────

MODE_STYLE = {
    "seqcomm":           {"color": "#2196F3", "label": "SeqComm (full)",       "lw": 2.5, "zorder": 5},
    "mappo":             {"color": "#F44336", "label": "MAPPO (no comm)",       "lw": 2.0, "zorder": 4},
    "seqcomm_random":    {"color": "#FF9800", "label": "SeqComm random order",  "lw": 1.8, "zorder": 3},
    "seqcomm_no_action": {"color": "#9C27B0", "label": "SeqComm no action share","lw": 1.8, "zorder": 3},
    "seqcomm_fixed":     {"color": "#4CAF50", "label": "SeqComm fixed order",   "lw": 1.8, "zorder": 3},
}

SMOOTH = 50   # rolling-average window


def rolling(arr, w):
    """1-D rolling mean; clamps window to data length."""
    w = min(w, len(arr))
    out = np.empty_like(arr, dtype=float)
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0)
    out[w - 1:] = (cumsum[w:] - cumsum[:-w]) / w
    for i in range(w - 1):
        out[i] = arr[: i + 1].mean()
    return out


# ── Load logs ─────────────────────────────────────────────────────────────────

def load_logs(log_dir: str) -> dict[str, list[list[dict]]]:
    """Returns {mode: [[ep_records seed0], [ep_records seed1], ...]}."""
    runs: dict[str, list] = defaultdict(list)
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(log_dir, fname)
        records = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                if "_meta" in obj:
                    meta = obj["_meta"]
                else:
                    records.append(obj)
        if not records:
            continue
        mode = meta.get("mode", fname)
        runs[mode].append(records)
    return dict(runs)


def extract(runs, key):
    """
    For each mode, return (episodes, mean, std) across seeds.
    Truncates to the shortest run so all arrays align.
    """
    out = {}
    for mode, seeds in runs.items():
        arrs = [np.array([r[key] for r in seed]) for seed in seeds]
        min_len = min(len(a) for a in arrs)
        arrs = np.stack([a[:min_len] for a in arrs])  # (n_seeds, T)
        mean = arrs.mean(axis=0)
        std  = arrs.std(axis=0)
        eps  = np.arange(min_len)
        out[mode] = (eps, mean, std)
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_metric(ax, data, key, ylabel, title, smooth=True):
    for mode, (eps, mean, std) in data.items():
        style = MODE_STYLE.get(mode, {"color": "gray", "label": mode, "lw": 1.5, "zorder": 1})
        m = rolling(mean, SMOOTH) if smooth else mean
        s = rolling(std,  SMOOTH) if smooth else std
        ax.plot(eps, m,
                color=style["color"], lw=style["lw"],
                label=style["label"], zorder=style["zorder"])
        ax.fill_between(eps, m - s, m + s,
                        color=style["color"], alpha=0.15, zorder=style["zorder"] - 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/baseline")
    parser.add_argument("--out",     default="figures/baseline.png")
    args = parser.parse_args()

    runs = load_logs(args.log_dir)
    if not runs:
        print(f"No JSONL files found in {args.log_dir}")
        return

    print(f"Loaded modes: {list(runs.keys())}")
    for mode, seeds in runs.items():
        print(f"  {mode}: {len(seeds)} seed(s), {len(seeds[0])} episodes each")

    reward_data    = extract(runs, "total_reward")
    success_data   = extract(runs, "success")
    collision_data = extract(runs, "n_collisions")
    entropy_data   = extract(runs, "order_entropy")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SeqComm Ablation Baseline — Intersection Crossing",
                 fontsize=13, fontweight="bold", y=1.01)

    plot_metric(axes[0, 0], reward_data,    "total_reward",  "Total Reward",    "(a) Reward")
    plot_metric(axes[0, 1], success_data,   "success",       "Success Rate",    "(b) Success Rate")
    plot_metric(axes[1, 0], collision_data, "n_collisions",  "Collisions / ep", "(c) Collisions")
    plot_metric(axes[1, 1], entropy_data,   "order_entropy", "Order Entropy",   "(d) Order Entropy")

    # Single shared legend below the plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
