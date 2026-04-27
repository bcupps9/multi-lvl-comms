#!/usr/bin/env bash
# Run SeqComm vs MAPPO comparison in the battleship environment.
#
# Uses a single representative config ("stable") to keep runtime manageable.
# After both modes finish, generates a side-by-side comparison plot.
#
# Usage:
#   bash battleship/run_battleship_comparison.sh
#
# Overrides:
#   EPISODES=3000 SEEDS="0 1" bash battleship/run_battleship_comparison.sh
#   RUN_ROOT=runs/comparison bash battleship/run_battleship_comparison.sh

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

SIM_BIN="${SIM_BIN:-./build-local/battleship/battleship-sim}"
BUILD_DIR="${BUILD_DIR:-build-local}"

EPISODES="${EPISODES:-3000}"
SEEDS="${SEEDS:-0 1}"
MODES="${MODES:-seqcomm mappo}"

AGENTS="${AGENTS:-2}"
BOSS="${BOSS:-1}"
SIGHT="${SIGHT:-4}"
FIRE="${FIRE:-2}"
STEPS="${STEPS:-60}"
SURVIVE="${SURVIVE:-0.005}"
WIN_REWARD="${WIN_REWARD:-10}"
NEAR_BOSS="${NEAR_BOSS:-0.10}"

LR_ENC="${LR_ENC:-0.0001}"
LR_WORLD="${LR_WORLD:-0.0003}"
LR_POL="${LR_POL:-0.0003}"
ENTROPY_COEF="${ENTROPY_COEF:-0.003}"
UPDATE_EVERY="${UPDATE_EVERY:-8}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

TRAINER_GRACE_SEC="${TRAINER_GRACE_SEC:-60}"
CPP_AFTER_PY_GRACE_SEC="${CPP_AFTER_PY_GRACE_SEC:-300}"

OBS_DIM=$(( (2 * SIGHT + 1) * (2 * SIGHT + 1) * 3 + 2 ))
RUN_ROOT="${RUN_ROOT:-runs/comparison/$(date +%Y%m%d_%H%M%S)}"

current_py_pid=""
current_cpp_pid=""
current_run_id=""

cleanup_pair() {
    local py_pid="${1:-}"
    local cpp_pid="${2:-}"
    if [[ -n "$py_pid" ]] && kill -0 "$py_pid" 2>/dev/null; then
        kill "$py_pid" 2>/dev/null || true
        wait "$py_pid" 2>/dev/null || true
    fi
    if [[ -n "$cpp_pid" ]] && kill -0 "$cpp_pid" 2>/dev/null; then
        kill "$cpp_pid" 2>/dev/null || true
        wait "$cpp_pid" 2>/dev/null || true
    fi
}

cleanup_current_pair() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -n "${current_py_pid:-}" || -n "${current_cpp_pid:-}" ]]; then
        echo "" >&2
        echo "Cleaning up interrupted run ${current_run_id:-unknown} ..." >&2
        cleanup_pair "$current_py_pid" "$current_cpp_pid"
    fi
    exit "$status"
}

trap cleanup_current_pair EXIT INT TERM HUP

tail_logs() {
    echo ""
    echo "---- C++ tail: $1 ----"
    tail -30 "$1" 2>/dev/null || true
    echo ""
    echo "---- Python tail: $2 ----"
    tail -30 "$2" 2>/dev/null || true
}

run_one() {
    local mode="$1"
    local seed="$2"

    local run_id="${mode}_seed${seed}"
    local run_dir="$RUN_ROOT/$run_id"
    local weights_dir="$run_dir/weights"
    local log_dir="$run_dir/logs"
    local fig_dir="$run_dir/figures"
    local stdout_dir="$run_dir/stdout"
    local trainer_log="$run_dir/trainer.jsonl"
    local cpp_out="$stdout_dir/cpp.out"
    local py_out="$stdout_dir/python.out"
    local cpp_status_file="$stdout_dir/cpp.status"
    local py_status_file="$stdout_dir/python.status"

    if [[ -e "$run_dir" ]]; then
        echo "Skipping existing run directory: $run_dir" >&2
        return 0
    fi

    mkdir -p "$weights_dir" "$log_dir" "$fig_dir" "$stdout_dir"

    echo ""
    echo "== $run_id (mode=$mode, seed=$seed, episodes=$EPISODES) =="

    "$PYTHON" battleship/train_battleship.py "$weights_dir" \
        --init \
        --obs-dim "$OBS_DIM" \
        --n-agents "$AGENTS" \
        > "$stdout_dir/init.out" 2>&1

    rm -f "$weights_dir/traj.ready" "$weights_dir/weights.ready" "$weights_dir/traj.done"
    rm -f "$py_status_file" "$cpp_status_file"

    (
        set +e
        "$PYTHON" battleship/train_battleship.py "$weights_dir" \
            --obs-dim "$OBS_DIM" \
            --n-agents "$AGENTS" \
            --update-every "$UPDATE_EVERY" \
            --lr-enc "$LR_ENC" \
            --lr-world "$LR_WORLD" \
            --lr-policy "$LR_POL" \
            --entropy-coef "$ENTROPY_COEF" \
            --grad-clip "$GRAD_CLIP" \
            --trainer-log "$trainer_log" &
        child_pid=$!
        trap 'kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; printf "%s\n" 143 > "$py_status_file"; exit 143' INT TERM HUP
        wait "$child_pid"
        status=$?
        trap - INT TERM HUP
        printf '%s\n' "$status" > "$py_status_file"
        exit "$status"
    ) > "$py_out" 2>&1 &
    local py_pid=$!
    current_py_pid="$py_pid"
    current_run_id="$run_id"

    (
        set +e
        "$SIM_BIN" "$weights_dir" \
            --mode "$mode" \
            --episodes "$EPISODES" \
            --seed "$seed" \
            --agents "$AGENTS" \
            --boss "$BOSS" \
            --sight "$SIGHT" \
            --fire "$FIRE" \
            --steps "$STEPS" \
            --survive "$SURVIVE" \
            --near-boss "$NEAR_BOSS" \
            --win-reward "$WIN_REWARD" \
            --log-dir "$log_dir" &
        child_pid=$!
        trap 'kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; printf "%s\n" 143 > "$cpp_status_file"; exit 143' INT TERM HUP
        wait "$child_pid"
        status=$?
        trap - INT TERM HUP
        printf '%s\n' "$status" > "$cpp_status_file"
        exit "$status"
    ) > "$cpp_out" 2>&1 &
    local cpp_pid=$!
    current_cpp_pid="$cpp_pid"

    local cpp_status=0
    local py_status=0
    local py_collected=0

    while true; do
        if [[ -f "$cpp_status_file" ]]; then
            cpp_status="$(cat "$cpp_status_file")"
            wait "$cpp_pid" || true
            break
        fi
        if [[ -f "$py_status_file" ]]; then
            py_status="$(cat "$py_status_file")"
            wait "$py_pid" || true
            py_collected=1
            if [[ "$py_status" -ne 0 ]]; then
                echo "Python trainer exited early (status=$py_status) for $run_id." >&2
                cleanup_pair "" "$cpp_pid"
                tail_logs "$cpp_out" "$py_out"
                exit 1
            fi
            local cpp_grace=0
            while [[ ! -f "$cpp_status_file" ]]; do
                if [[ "$cpp_grace" -ge "$CPP_AFTER_PY_GRACE_SEC" ]]; then
                    echo "C++ sim still running ${CPP_AFTER_PY_GRACE_SEC}s after Python exited." >&2
                    cleanup_pair "" "$cpp_pid"
                    tail_logs "$cpp_out" "$py_out"
                    exit 1
                fi
                sleep 1
                cpp_grace=$((cpp_grace + 1))
            done
            cpp_status="$(cat "$cpp_status_file")"
            wait "$cpp_pid" || true
            break
        fi
        sleep 2
    done

    if [[ "$cpp_status" -ne 0 ]]; then
        echo "C++ sim failed for $run_id (status=$cpp_status)." >&2
        cleanup_pair "$py_pid" ""
        tail_logs "$cpp_out" "$py_out"
        exit 1
    fi

    if [[ "$py_collected" -eq 0 ]]; then
        local waited=0
        while [[ ! -f "$py_status_file" ]]; do
            if [[ "$waited" -ge "$TRAINER_GRACE_SEC" ]]; then
                echo "Python trainer did not exit after C++ finished; killing." >&2
                cleanup_pair "$py_pid" ""
                tail_logs "$cpp_out" "$py_out"
                exit 1
            fi
            sleep 1
            waited=$((waited + 1))
        done
        py_status="$(cat "$py_status_file")"
        wait "$py_pid" || true
    fi

    if [[ "$py_status" -ne 0 ]]; then
        echo "Python trainer failed for $run_id (status=$py_status)." >&2
        tail_logs "$cpp_out" "$py_out"
        exit 1
    fi

    current_py_pid=""
    current_cpp_pid=""
    current_run_id=""

    local log_file
    log_file="$(ls -t "$log_dir"/*.jsonl | head -1)"
    "$PYTHON" battleship/plot_battleship.py "$log_file" \
        --out "$fig_dir/${run_id}.png" \
        > "$stdout_dir/plot.out" 2>&1
    echo "done: $run_id -> $log_dir"
}

if [[ ! -x "$SIM_BIN" ]]; then
    echo "Building battleship-sim in $BUILD_DIR ..."
    make -C "$BUILD_DIR" battleship-sim
fi

mkdir -p "$RUN_ROOT"

for mode in $MODES; do
    for seed in $SEEDS; do
        run_one "$mode" "$seed"
    done
done

echo ""
echo "All runs done. Generating comparison plot..."
"$PYTHON" battleship/plot_comparison.py "$RUN_ROOT" \
    --out "$RUN_ROOT/comparison.png"
echo "Comparison plot: $RUN_ROOT/comparison.png"
echo "Run root: $RUN_ROOT"
