#!/usr/bin/env bash
# Island coverage ablation: 5 modes × 3 seeds, runs in parallel.
# All 15 jobs launch simultaneously so total wall time ≈ slowest single run.
#
# Usage:  bash scripts/run_island.sh
# Override episodes: EPISODES=500 bash scripts/run_island.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/island}"
EPISODES="${EPISODES:-150}"
# Reduce world-model rollout depth for seqcomm variants to keep wall time ~1 hour.
# Full H=5/F=4 takes ~1.6 min/episode; H=2/F=2 is ~5x faster with minor quality loss.
WM_H="${WM_H:-2}"
WM_F="${WM_F:-2}"
MODES=(seqcomm mappo seqcomm_random seqcomm_no_action seqcomm_fixed)
SEEDS=(0 1 2)

mkdir -p "$LOG_DIR"

pids=()

for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Launching mode=$mode  seed=$seed"
        docker run --rm \
            -v "$REPO_DIR":/workspace \
            -w /workspace \
            cs286:latest \
            python3 -m training.train \
                --env coverage \
                --mode "$mode" \
                --seed "$seed" \
                --episodes "$EPISODES" \
                --wm-H "$WM_H" \
                --wm-F "$WM_F" \
                --log-dir logs/island \
        >> "$LOG_DIR/${mode}_seed${seed}.stdout" 2>&1 &
        pids+=($!)
    done
done

echo ""
echo "All ${#pids[@]} runs launched. Waiting for completion..."
echo "Tail progress with: tail -f $LOG_DIR/seqcomm_seed0.stdout"

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "A run failed (pid $pid)"
        failed=$(( failed + 1 ))
    fi
done

echo ""
if [[ $failed -eq 0 ]]; then
    echo "All runs complete. Logs in $LOG_DIR"
else
    echo "$failed run(s) failed. Check stdout logs in $LOG_DIR"
fi
ls "$LOG_DIR"/*.jsonl 2>/dev/null | wc -l | xargs echo "JSONL files produced:"
