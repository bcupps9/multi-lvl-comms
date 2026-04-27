#!/usr/bin/env bash
set -euo pipefail

# Sequential battleship experiment grid.
#
# Each run launches one Python trainer and one C++ simulator in the background.
# The script waits for both to finish cleanly before starting the next run.
#
# Typical use:
#   bash battleship/run_battleship_grid.sh
#
# Useful overrides:
#   EPISODES=3000 SEEDS="0 1" bash battleship/run_battleship_grid.sh
#   RUN_ROOT=runs/my_grid bash battleship/run_battleship_grid.sh

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

SIM_BIN="${SIM_BIN:-./build-local/battleship/battleship-sim}"
BUILD_DIR="${BUILD_DIR:-build-local}"

MODE="${MODE:-seqcomm}"
EPISODES="${EPISODES:-5000}"
SEEDS="${SEEDS:-0 1 2}"

AGENTS="${AGENTS:-2}"
BOSS="${BOSS:-1}"
SIGHT="${SIGHT:-4}"
FIRE="${FIRE:-2}"
STEPS="${STEPS:-60}"
SURVIVE="${SURVIVE:-0.005}"
WIN_REWARD="${WIN_REWARD:-10}"

LR_ENC="${LR_ENC:-0.0001}"
LR_WORLD="${LR_WORLD:-0.0003}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TRAINER_GRACE_SEC="${TRAINER_GRACE_SEC:-60}"

OBS_DIM=$(( (2 * SIGHT + 1) * (2 * SIGHT + 1) * 3 + 2 ))
RUN_ROOT="${RUN_ROOT:-runs/battleship_grid/$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
current_py_pid=""
current_cpp_pid=""
current_run_id=""

# name | update_every | lr_policy | entropy_coef | near_boss
CONFIGS=(
    "stable|8|0.0003|0.003|0.20"
    "less_shaping|8|0.0003|0.003|0.05"
    "more_stable|16|0.0003|0.003|0.10"
    "lower_explore|8|0.0003|0.001|0.10"
)

find_existing_battleship_processes() {
    local matches=""
    if command -v pgrep >/dev/null 2>&1; then
        matches="$(pgrep -fl 'battleship/train_battleship.py|battleship-sim' 2>/dev/null || true)"
    fi
    if [[ -n "$matches" ]]; then
        echo "$matches"
        return
    fi
    ps -axo pid=,command= | awk '
        /battleship\/train_battleship\.py|battleship-sim/ && $0 !~ /awk/ {
            print
        }
    ' || true
}

if [[ "${ALLOW_EXISTING_BATTLESHIP:-0}" != "1" ]]; then
    existing_processes="$(find_existing_battleship_processes)"
    if [[ -n "$existing_processes" ]]; then
        echo "Existing battleship train/sim processes are already running:" >&2
        echo "$existing_processes" >&2
        echo "" >&2
        echo "Stop those first, or rerun with ALLOW_EXISTING_BATTLESHIP=1 if this is intentional." >&2
        exit 1
    fi
fi

mkdir -p "$RUN_ROOT"

if [[ ! -x "$SIM_BIN" ]]; then
    echo "Building battleship-sim in $BUILD_DIR ..."
    make -C "$BUILD_DIR" battleship-sim
fi

printf 'run_id,config,seed,episodes_logged,last500_win,last500_boss_hits,last500_zero_hit,last500_agent_hits,last500_timeout,last500_fire_dist,last500_oob_rate,last_policy_std,last_entropy,log_file,trainer_log\n' > "$SUMMARY_CSV"

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
    local cpp_out="$1"
    local py_out="$2"
    echo ""
    echo "---- C++ tail: $cpp_out ----"
    tail -40 "$cpp_out" 2>/dev/null || true
    echo ""
    echo "---- Python tail: $py_out ----"
    tail -40 "$py_out" 2>/dev/null || true
}

summarize_run() {
    local log_file="$1"
    local trainer_log="$2"
    local run_id="$3"
    local config_name="$4"
    local seed="$5"

    "$PYTHON" - "$log_file" "$trainer_log" "$run_id" "$config_name" "$seed" <<'PY' >> "$SUMMARY_CSV"
import csv
import json
import sys

log_file, trainer_log, run_id, config_name, seed = sys.argv[1:]
rows = []
with open(log_file) as f:
    for line in f:
        obj = json.loads(line)
        if "_meta" not in obj:
            rows.append(obj)

tail = rows[-500:] if len(rows) >= 500 else rows
n = max(1, len(tail))
win = sum(1 for r in tail if r.get("agents_won")) / n
boss_hits = sum(float(r.get("boss_hits", 0.0)) for r in tail) / n
zero = sum(1 for r in tail if int(r.get("boss_hits", 0)) == 0) / n
agent_hits = sum(float(r.get("agent_hits", 0.0)) for r in tail) / n
timeout = sum(1 for r in tail if not r.get("agents_won") and not r.get("boss_won")) / n
fire_dist = sum(float(r.get("mean_fire_dist", 0.0)) for r in tail) / n
shots = sum(float(r.get("agent_shots", 0.0)) for r in tail)
oob = sum(float(r.get("fire_oob", 0.0)) for r in tail) / max(1.0, shots)

last_policy_std = ""
last_entropy = ""
try:
    trainer_rows = []
    with open(trainer_log) as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" not in obj:
                trainer_rows.append(obj)
    if trainer_rows:
        last_policy_std = f"{float(trainer_rows[-1].get('policy_std_mean', 0.0)):.6f}"
        last_entropy = f"{float(trainer_rows[-1].get('entropy', 0.0)):.6f}"
except FileNotFoundError:
    pass

writer = csv.writer(sys.stdout)
writer.writerow([
    run_id,
    config_name,
    seed,
    len(rows),
    f"{win:.6f}",
    f"{boss_hits:.6f}",
    f"{zero:.6f}",
    f"{agent_hits:.6f}",
    f"{timeout:.6f}",
    f"{fire_dist:.6f}",
    f"{oob:.6f}",
    last_policy_std,
    last_entropy,
    log_file,
    trainer_log,
])
PY
}

run_one() {
    local config_name="$1"
    local update_every="$2"
    local lr_policy="$3"
    local entropy_coef="$4"
    local near_boss="$5"
    local seed="$6"

    local run_id="${config_name}_seed${seed}"
    local run_dir="$RUN_ROOT/$run_id"
    local weights_dir="$run_dir/weights"
    local log_dir="$run_dir/logs"
    local fig_dir="$run_dir/figures"
    local stdout_dir="$run_dir/stdout"
    local trainer_log="$run_dir/trainer.jsonl"
    local cpp_out="$stdout_dir/cpp.out"
    local py_out="$stdout_dir/python.out"

    if [[ -e "$run_dir" ]]; then
        echo "Refusing to overwrite existing run directory: $run_dir" >&2
        exit 1
    fi

    mkdir -p "$weights_dir" "$log_dir" "$fig_dir" "$stdout_dir"

    echo ""
    echo "== $run_id =="
    echo "config: update_every=$update_every lr_policy=$lr_policy entropy=$entropy_coef near_boss=$near_boss seed=$seed"

    "$PYTHON" battleship/train_battleship.py "$weights_dir" \
        --init \
        --obs-dim "$OBS_DIM" \
        --n-agents "$AGENTS" \
        > "$stdout_dir/init.out" 2>&1

    rm -f "$weights_dir/traj.ready" "$weights_dir/weights.ready" "$weights_dir/traj.done"

    "$PYTHON" battleship/train_battleship.py "$weights_dir" \
        --obs-dim "$OBS_DIM" \
        --n-agents "$AGENTS" \
        --update-every "$update_every" \
        --lr-enc "$LR_ENC" \
        --lr-world "$LR_WORLD" \
        --lr-policy "$lr_policy" \
        --entropy-coef "$entropy_coef" \
        --grad-clip "$GRAD_CLIP" \
        --trainer-log "$trainer_log" \
	    > "$py_out" 2>&1 &
    local py_pid=$!
    current_py_pid="$py_pid"
    current_run_id="$run_id"

    "$SIM_BIN" "$weights_dir" \
        --mode "$MODE" \
        --episodes "$EPISODES" \
        --seed "$seed" \
        --agents "$AGENTS" \
        --boss "$BOSS" \
        --sight "$SIGHT" \
        --fire "$FIRE" \
        --steps "$STEPS" \
        --survive "$SURVIVE" \
        --near-boss "$near_boss" \
        --win-reward "$WIN_REWARD" \
        --log-dir "$log_dir" \
	    > "$cpp_out" 2>&1 &
    local cpp_pid=$!
    current_cpp_pid="$cpp_pid"
    printf 'python_pid=%s\ncpp_pid=%s\n' "$py_pid" "$cpp_pid" > "$stdout_dir/pids.env"

    local cpp_status=0
    local py_status=0
    local py_collected=0

    while true; do
        local cpp_alive=0
        local py_alive=0
        kill -0 "$cpp_pid" 2>/dev/null && cpp_alive=1
        kill -0 "$py_pid" 2>/dev/null && py_alive=1

        if [[ "$cpp_alive" -eq 0 ]]; then
            wait "$cpp_pid" || cpp_status=$?
            break
        fi

        if [[ "$py_alive" -eq 0 ]]; then
            wait "$py_pid" || py_status=$?
            py_collected=1
            if [[ "$py_status" -ne 0 ]]; then
                echo "Python trainer exited before C++ simulator finished (status=$py_status)." >&2
                cleanup_pair "" "$cpp_pid"
                tail_logs "$cpp_out" "$py_out"
                exit 1
            fi

            local cpp_grace=0
            while kill -0 "$cpp_pid" 2>/dev/null; do
                if [[ "$cpp_grace" -ge 10 ]]; then
                    echo "Python trainer exited cleanly, but C++ simulator kept running." >&2
                    cleanup_pair "" "$cpp_pid"
                    tail_logs "$cpp_out" "$py_out"
                    exit 1
                fi
                sleep 1
                cpp_grace=$((cpp_grace + 1))
            done
            wait "$cpp_pid" || cpp_status=$?
            break
        fi

        sleep 2
    done

    if [[ "$cpp_status" -ne 0 ]]; then
        echo "C++ simulator failed for $run_id (status=$cpp_status)." >&2
        cleanup_pair "$py_pid" ""
        tail_logs "$cpp_out" "$py_out"
        exit 1
    fi

    if [[ "$py_collected" -eq 0 ]]; then
        local waited=0
        while kill -0 "$py_pid" 2>/dev/null; do
            if [[ "$waited" -ge "$TRAINER_GRACE_SEC" ]]; then
                echo "Python trainer did not exit after C++ completion; killing it." >&2
                cleanup_pair "$py_pid" ""
                tail_logs "$cpp_out" "$py_out"
                exit 1
            fi
            sleep 1
            waited=$((waited + 1))
        done

        wait "$py_pid" || py_status=$?
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

    summarize_run "$log_file" "$trainer_log" "$run_id" "$config_name" "$seed"
    echo "done: $run_id"
    echo "  log:     $log_file"
    echo "  trainer: $trainer_log"
    echo "  plot:    $fig_dir/${run_id}.png"
}

for config in "${CONFIGS[@]}"; do
    old_ifs="$IFS"
    IFS='|'
    set -- $config
    IFS="$old_ifs"

    config_name="$1"
    update_every="$2"
    lr_policy="$3"
    entropy_coef="$4"
    near_boss="$5"

    for seed in $SEEDS; do
        run_one "$config_name" "$update_every" "$lr_policy" "$entropy_coef" "$near_boss" "$seed"
    done
done

echo ""
echo "All runs complete."
echo "Summary: $SUMMARY_CSV"
