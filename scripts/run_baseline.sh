#!/usr/bin/env bash
# Tier-1 ablation baseline: 5 modes × 3 seeds on intersection env.
# Runs inside the cs286:latest Docker image (has torch + numpy).
# Logs land in logs/baseline/ on the host via volume mount.
#
# Usage:  bash scripts/run_baseline.sh
# To override episode count: EPISODES=200 bash scripts/run_baseline.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${2:-$REPO_DIR/logs/baseline}"
EPISODES="${EPISODES:-2000}"
MODES=(seqcomm mappo seqcomm_random seqcomm_no_action seqcomm_fixed)
SEEDS=(0 1 2)

mkdir -p "$LOG_DIR"

total=$(( ${#MODES[@]} * ${#SEEDS[@]} ))
i=0

for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        i=$(( i + 1 ))
        echo "── run $i/$total  mode=$mode  seed=$seed ──"
        docker run --rm \
            -v "$REPO_DIR":/workspace \
            -w /workspace \
            cs286:latest \
            python3 -m training.train \
                --env intersection \
                --mode "$mode" \
                --seed "$seed" \
                --episodes "$EPISODES" \
                --log-dir logs/baseline
    done
done

echo ""
echo "Done. Logs in $LOG_DIR"
ls "$LOG_DIR"
