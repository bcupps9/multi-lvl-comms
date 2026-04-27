#!/usr/bin/env python3
"""
Plot battleship JSONL training logs.

Examples:
    python battleship/plot_battleship.py logs/battleship/seqcomm_20260426_154415.jsonl
    python battleship/plot_battleship.py logs/battleship/*.jsonl --out-dir figures/battleship
    python battleship/plot_battleship.py --window 500
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                meta = obj["_meta"]
            else:
                rows.append(obj)

    if not rows:
        raise ValueError(f"no episode rows found in {path}")
    return meta, rows


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    window = max(1, min(window, values.size))
    if window == 1:
        return values

    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def stats(rows: list[dict[str, Any]], start: int = 0) -> dict[str, float]:
    subset = rows[start:]
    n = max(1, len(subset))
    return {
        "reward": sum(float(r.get("reward", 0.0)) for r in subset) / n,
        "boss_hits": sum(float(r.get("boss_hits", 0.0)) for r in subset) / n,
        "zero_hit": sum(1 for r in subset if int(r.get("boss_hits", 0)) == 0) / n,
        "win": sum(1 for r in subset if bool(r.get("agents_won", False))) / n,
        "timeout": sum(
            1
            for r in subset
            if not bool(r.get("agents_won", False)) and not bool(r.get("boss_won", False))
        ) / n,
        "fire_dist": sum(float(r.get("mean_fire_dist", 0.0)) for r in subset) / n,
        "oob_rate": (
            sum(float(r.get("fire_oob", 0.0)) for r in subset) /
            max(1.0, sum(float(r.get("agent_shots", 0.0)) for r in subset))
        ),
    }


def plot_log(path: Path, out_path: Path, window_arg: int, dpi: int) -> dict[str, float]:
    meta, rows = read_log(path)
    n_rows = len(rows)
    window = window_arg if window_arg > 0 else min(200, max(10, n_rows // 20))

    ep = np.array([float(r.get("ep", i)) for i, r in enumerate(rows)])
    reward = np.array([float(r.get("reward", 0.0)) for r in rows])
    boss_hits = np.array([float(r.get("boss_hits", 0.0)) for r in rows])
    agent_hits = np.array([float(r.get("agent_hits", 0.0)) for r in rows])
    steps = np.array([float(r.get("steps", 0.0)) for r in rows])
    win = np.array([1.0 if r.get("agents_won", False) else 0.0 for r in rows])
    loss = np.array([1.0 if r.get("boss_won", False) else 0.0 for r in rows])
    timeout = 1.0 - win - loss
    spread = np.array([float(r.get("intention_spread", 0.0)) for r in rows])
    zero_hit = (boss_hits == 0).astype(float)
    three_hit = (boss_hits >= 3).astype(float)
    mean_fire_dist = np.array([float(r.get("mean_fire_dist", np.nan)) for r in rows])

    first0 = []
    for r in rows:
        fm = r.get("first_mover", [])
        total = sum(fm) if fm else 0
        first0.append(float(fm[0]) / total if total else 0.5)
    first0_arr = np.array(first0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("white")

    subtitle = (
        f"M={meta.get('M', '?')} agents={meta.get('n_agents', '?')} "
        f"boss={meta.get('n_boss', '?')} fire={meta.get('fire_range', '?')} "
        f"survive={meta.get('reward_survive', '?')} "
        f"near={meta.get('reward_near_boss', '?')} "
        f"win={meta.get('reward_agents_win', '?')}"
    )
    fig.suptitle(
        f"Battleship SeqComm Training Run - {path.stem}\n{subtitle}",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    ax = axes[0, 0]
    ax.plot(ep, reward, color="#b8c0cc", lw=0.5, alpha=0.45, label="episode")
    ax.plot(ep, rolling(reward, window), color="#1f77b4", lw=2.2,
            label=f"rolling {window}")
    ax.axhline(0, color="#555", lw=0.8, alpha=0.6)
    ax.set_title("Reward")
    ax.set_ylabel("episode reward")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    ax.plot(ep, rolling(boss_hits, window), color="#2ca02c", lw=2.2,
            label="boss hits")
    ax.plot(ep, rolling(agent_hits, window), color="#d62728", lw=2.0,
            label="agent hits")
    ax.axhline(3, color="#2ca02c", lw=1.0, ls="--", alpha=0.7,
               label="boss sunk threshold")
    ax.set_title("Hits Per Episode")
    ax.set_ylabel("hits")
    ax.set_ylim(-0.1, max(6.3, float(np.nanmax(agent_hits)) + 0.2))
    ax.legend(frameon=False, loc="center right")

    ax = axes[1, 0]
    ax.plot(ep, rolling(win, window), color="#2ca02c", lw=2.2, label="agent win")
    ax.plot(ep, rolling(loss, window), color="#d62728", lw=2.0, label="boss win")
    ax.plot(ep, rolling(timeout, window), color="#7f7f7f", lw=1.8, label="timeout")
    ax.set_title("Outcome Rate")
    ax.set_ylabel("rolling fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="center right")

    ax = axes[1, 1]
    ax.plot(ep, rolling(zero_hit, window), color="#9467bd", lw=2.1,
            label="zero boss hits")
    ax.plot(ep, rolling(three_hit, window), color="#17becf", lw=2.1,
            label="3+ boss hits")
    ax.set_title("Aiming Progress")
    ax.set_ylabel("rolling fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="center right")
    if not np.isnan(mean_fire_dist).all():
        ax2 = ax.twinx()
        ax2.plot(ep, rolling(mean_fire_dist, window), color="#555", lw=1.6,
                 alpha=0.75, label="mean fire dist")
        ax2.set_ylabel("mean fire distance")
        ax2.legend(frameon=False, loc="upper right")

    ax = axes[2, 0]
    ax.plot(ep, rolling(steps, window), color="#ff7f0e", lw=2.0, label="steps")
    ax.set_title("Episode Length")
    ax.set_ylabel("steps")
    ax.set_xlabel("episode")
    ax.set_ylim(0, max(62, float(np.nanmax(steps)) + 2))
    ax.legend(frameon=False, loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(ep, rolling(spread, window), color="#4c78a8", lw=1.8, alpha=0.8,
             label="intention spread")
    ax2.set_ylabel("intention spread")
    ax2.legend(frameon=False, loc="upper right")

    ax = axes[2, 1]
    ax.plot(ep, rolling(first0_arr, window), color="#8c564b", lw=2.0,
            label="agent 0 first-mover share")
    ax.axhline(0.5, color="#555", lw=0.9, ls="--", alpha=0.6)
    ax.set_title("Ordering Balance")
    ax.set_ylabel("rolling fraction")
    ax.set_xlabel("episode")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="center right")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    all_stats = stats(rows)
    last_start = max(0, n_rows - 500)
    last_stats = stats(rows, last_start)
    summary = (
        f"episodes logged: {n_rows} / requested {meta.get('episodes', '?')}   "
        f"overall win {all_stats['win']:.1%}, boss hits {all_stats['boss_hits']:.2f}, "
        f"zero-hit {all_stats['zero_hit']:.1%}   |   "
        f"last {n_rows - last_start} win {last_stats['win']:.1%}, "
        f"boss hits {last_stats['boss_hits']:.2f}, "
        f"zero-hit {last_stats['zero_hit']:.1%}, "
        f"timeout {last_stats['timeout']:.1%}, "
        f"fire dist {last_stats['fire_dist']:.2f}"
    )
    fig.text(0.5, 0.015, summary, ha="center", fontsize=10, color="#333")

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return last_stats


def default_logs() -> list[Path]:
    return sorted(Path("logs/battleship").glob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot battleship JSONL logs.")
    parser.add_argument("logs", nargs="*", type=Path,
                        help="JSONL log file(s). Defaults to logs/battleship/*.jsonl")
    parser.add_argument("--out", type=Path,
                        help="Output PNG path. Only valid with one input log.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures/battleship"),
                        help="Directory for generated PNGs when plotting multiple logs.")
    parser.add_argument("--window", type=int, default=0,
                        help="Rolling window. Default chooses an automatic window.")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    logs = args.logs or default_logs()
    if not logs:
        raise SystemExit("No logs found. Pass log paths or create logs/battleship/*.jsonl")
    if args.out and len(logs) != 1:
        raise SystemExit("--out can only be used with exactly one log file")

    for log_path in logs:
        out_path = args.out or (args.out_dir / f"{log_path.stem}.png")
        last = plot_log(log_path, out_path, args.window, args.dpi)
        print(
            f"{log_path} -> {out_path} | "
            f"last win={last['win']:.1%} "
            f"boss_hits={last['boss_hits']:.2f} "
            f"zero_hit={last['zero_hit']:.1%}"
        )


if __name__ == "__main__":
    main()
