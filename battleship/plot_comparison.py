#!/usr/bin/env python3
"""
Side-by-side comparison plot: SeqComm vs MAPPO (or any set of modes).

Usage:
    python battleship/plot_comparison.py runs/comparison/20260426_120000
    python battleship/plot_comparison.py runs/comparison/20260426_120000 --out figures/comparison.png
    python battleship/plot_comparison.py runs/comparison/20260426_120000 --window 200
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "_meta" not in obj:
                rows.append(obj)
    return rows


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = max(1, min(window, len(values)))
    if window == 1:
        return values
    kernel = np.ones(window) / window
    left = window // 2
    padded = np.pad(values, (left, window - 1 - left), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_mode_runs(run_root: Path, mode: str) -> list[list[dict]]:
    """Return list of episode-row lists, one per seed."""
    runs = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.name.startswith(f"{mode}_seed"):
            continue
        log_dir = run_dir / "logs"
        jsonls = sorted(log_dir.glob("*.jsonl"))
        if not jsonls:
            continue
        rows = read_jsonl(jsonls[-1])
        if rows:
            runs.append(rows)
    return runs


def mean_std_over_seeds(
    seed_runs: list[list[dict]],
    key: str,
    n_ep: int,
    transform=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (episodes, mean, std) arrays aligned to n_ep."""
    arrays = []
    for rows in seed_runs:
        vals = np.array([float(r.get(key, 0.0)) for r in rows])
        if transform:
            vals = transform(vals)
        if len(vals) < n_ep:
            vals = np.pad(vals, (0, n_ep - len(vals)), mode="edge")
        arrays.append(vals[:n_ep])
    stacked = np.stack(arrays, axis=0)
    return np.arange(n_ep), stacked.mean(0), stacked.std(0)


def plot_metric(
    ax,
    mode_data: dict[str, tuple],
    window: int,
    colors: dict[str, str],
    title: str,
    ylabel: str,
    ylim: tuple | None = None,
):
    for mode, (ep, mean, std) in mode_data.items():
        m = rolling(mean, window)
        s = rolling(std, window)
        ax.plot(ep, m, color=colors[mode], lw=2.2, label=mode)
        ax.fill_between(ep, m - s, m + s, color=colors[mode], alpha=0.15)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path, help="Directory containing mode_seed* subdirs")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=["seqcomm", "mappo"])
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    run_root: Path = args.run_root
    out_path: Path = args.out or (run_root / "comparison.png")

    colors = {
        "seqcomm": "#1f77b4",
        "mappo":   "#d62728",
        "fixed_order": "#2ca02c",
    }

    # Load all seeds for each mode
    mode_runs: dict[str, list[list[dict]]] = {}
    for mode in args.modes:
        runs = load_mode_runs(run_root, mode)
        if not runs:
            print(f"Warning: no completed runs found for mode '{mode}' in {run_root}")
        else:
            mode_runs[mode] = runs
            print(f"{mode}: {len(runs)} seed(s), {len(runs[0])} episodes each")

    if not mode_runs:
        raise SystemExit(f"No runs found in {run_root}. Check that runs completed.")

    n_ep = min(len(r) for runs in mode_runs.values() for r in runs)
    window = args.window if args.window > 0 else min(200, max(10, n_ep // 20))

    # Build per-metric dicts
    def build(key, transform=None):
        return {
            mode: mean_std_over_seeds(runs, key, n_ep, transform)
            for mode, runs in mode_runs.items()
        }

    def win_transform(arr):
        return arr.astype(float)  # boss_wins is bool already cast

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Battleship: SeqComm vs MAPPO Comparison\n"
        f"(rolling window={window}, shading=±1 std over seeds)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    metrics = [
        (axes[0, 0], "reward",          None,  "Reward",             "episode reward",     None),
        (axes[0, 1], "boss_hits",        None,  "Boss Hits / Episode","hits",               (-0.1, 6.5)),
        (axes[0, 2], "agents_won",       lambda a: a.astype(float),
                                               "Win Rate",           "fraction",           (-0.03, 1.03)),
        (axes[1, 0], "agent_hits",       None,  "Agent Hits / Episode","hits",              None),
        (axes[1, 1], "mean_fire_dist",   None,  "Mean Fire Distance", "grid cells",         None),
        (axes[1, 2], "intention_spread", None,  "Intention Spread",   "spread",             None),
    ]

    for ax, key, transform, title, ylabel, ylim in metrics:
        data = {}
        for mode, runs in mode_runs.items():
            ep, mean, std = mean_std_over_seeds(runs, key, n_ep, transform)
            data[mode] = (ep, mean, std)
        plot_metric(ax, data, window, colors, title, ylabel, ylim)

    for ax in axes[1]:
        ax.set_xlabel("episode", fontsize=9)

    # Summary table
    summary_lines = []
    for mode, runs in mode_runs.items():
        all_rows = [r for run in runs for r in run[-500:]]
        n = max(1, len(all_rows))
        win  = sum(1 for r in all_rows if r.get("agents_won")) / n
        hits = sum(float(r.get("boss_hits", 0)) for r in all_rows) / n
        zero = sum(1 for r in all_rows if int(r.get("boss_hits", 0)) == 0) / n
        summary_lines.append(
            f"{mode}: last-500 win={win:.1%} boss_hits={hits:.2f} zero_hit={zero:.1%}"
        )
    fig.text(0.5, 0.01, "   |   ".join(summary_lines), ha="center", fontsize=10, color="#333")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
