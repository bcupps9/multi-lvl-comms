#!/usr/bin/env bash
# Run a long SeqComm vs MAPPO comparison on the battleship environment.
#
# Environment and training hyperparameters exactly match run_battleship_grid.sh
# (SIGHT=2, 4 scalar obs channels, boss-fire-period/aim-lag, proximity/hit rewards,
# UPDATE_EVERY=16, curriculum enabled).  SeqComm uses the H=2 world-model rollout
# for intention ordering (--wm-intention, matching wm_loss00 in exp1).  MAPPO runs
# synchronously with no ordering.  After all runs finish a side-by-side comparison
# plot is generated via plot_comparison.py.
#
# Usage:
#   bash battleship/run_comparison_long.sh
#
# Overrides (all optional):
#   EPISODES=20000 SEEDS="0 1 2" bash battleship/run_comparison_long.sh
#   MODES="seqcomm mappo fixed_order" bash battleship/run_comparison_long.sh
#   WM_H=4 bash battleship/run_comparison_long.sh
#   RUN_ROOT=runs/comparison/my_run bash battleship/run_comparison_long.sh
#   SKIP_BUILD=1 bash battleship/run_comparison_long.sh

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

SIM_BIN="${SIM_BIN:-./build-local/battleship/battleship-sim}"
BUILD_DIR="${BUILD_DIR:-build-local}"

EPISODES="${EPISODES:-15000}"
SEEDS="${SEEDS:-0}"
MODES="${MODES:-seqcomm mappo}"

# ── Environment — must match run_battleship_grid.sh exactly ───────────────────
AGENTS="${AGENTS:-2}"
BOSS="${BOSS:-1}"
SIGHT="${SIGHT:-2}"
FIRE="${FIRE:-2}"
BOSS_FIRE_PERIOD="${BOSS_FIRE_PERIOD:-8}"
BOSS_AIM_LAG="${BOSS_AIM_LAG:-1}"
STEPS="${STEPS:-60}"
SURVIVE="${SURVIVE:-0.005}"
NEAR_BOSS="${NEAR_BOSS:-0.15}"
PROXIMITY="${PROXIMITY:-0.02}"
HIT_BOSS="${HIT_BOSS:-2.0}"
HIT_SELF="${HIT_SELF:--0.3}"
WIN_REWARD="${WIN_REWARD:-10}"

# 4 scalars: hp, step, row, col  (must match grid script)
OBS_DIM=$(( (2 * SIGHT + 1) * (2 * SIGHT + 1) * 3 + 4 ))

# ── World-model intention ordering (seqcomm only) ──────────────────────────────
# Matches wm_loss00 in run_battleship_grid.sh EXP=exp1.
# MAPPO never uses ordering, so this flag is passed only to seqcomm/fixed_order.
WM_H="${WM_H:-2}"

# ── Training hyperparameters — must match grid "curriculum" config ─────────────
UPDATE_EVERY="${UPDATE_EVERY:-64}"
LR_ENC="${LR_ENC:-0.0001}"
LR_WORLD="${LR_WORLD:-0.0003}"
LR_POL="${LR_POL:-0.0003}"
ENTROPY_COEF="${ENTROPY_COEF:-0.003}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# ── Process timeouts ───────────────────────────────────────────────────────────
TRAINER_GRACE_SEC="${TRAINER_GRACE_SEC:-60}"
CPP_AFTER_PY_GRACE_SEC="${CPP_AFTER_PY_GRACE_SEC:-300}"

RUN_ROOT="${RUN_ROOT:-runs/comparison/$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="$RUN_ROOT/summary.csv"

current_py_pid=""
current_cpp_pid=""
current_run_id=""

# ── Guard against accidental double-launch ─────────────────────────────────────
find_existing_battleship_processes() {
    local matches=""
    if command -v pgrep >/dev/null 2>&1; then
        matches="$(pgrep -fl 'battleship/train_battleship.py|battleship-sim' 2>/dev/null || true)"
    fi
    if [[ -n "$matches" ]]; then echo "$matches"; return; fi
    ps -axo pid=,command= 2>/dev/null | awk '
        /battleship\/train_battleship\.py|battleship-sim/ && $0 !~ /awk/ { print }
    ' || true
}

if [[ "${ALLOW_EXISTING_BATTLESHIP:-0}" != "1" ]]; then
    existing="$(find_existing_battleship_processes)"
    if [[ -n "$existing" ]]; then
        echo "Existing battleship train/sim processes are already running:" >&2
        echo "$existing" >&2
        echo "" >&2
        echo "Stop those first, or rerun with ALLOW_EXISTING_BATTLESHIP=1." >&2
        exit 1
    fi
fi

# ── Build ──────────────────────────────────────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "Ensuring battleship-sim is built in $BUILD_DIR ..."
    make -C "$BUILD_DIR" battleship-sim
elif [[ ! -x "$SIM_BIN" ]]; then
    echo "error: simulator not found or not executable: $SIM_BIN" >&2
    echo "rerun without SKIP_BUILD=1 to build it automatically" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"
printf 'run_id,mode,seed,episodes_logged,last500_win,last500_boss_hits,last500_zero_hit,last500_agent_hits,last500_timeout,last500_fire_dist,last500_oob_rate,last_policy_std,last_entropy,final_curriculum_stage,last500_comm_rate,log_file,trainer_log\n' > "$SUMMARY_CSV"

# ── Cleanup helpers ────────────────────────────────────────────────────────────
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

# ── Per-run summary row ────────────────────────────────────────────────────────
summarize_run() {
    local log_file="$1"
    local trainer_log="$2"
    local run_id="$3"
    local mode="$4"
    local seed="$5"

    "$PYTHON" - "$log_file" "$trainer_log" "$run_id" "$mode" "$seed" <<'PY' >> "$SUMMARY_CSV"
import csv, json, sys

log_file, trainer_log, run_id, mode, seed = sys.argv[1:]
rows = []
with open(log_file) as f:
    for line in f:
        obj = json.loads(line)
        if "_meta" not in obj:
            rows.append(obj)

tail = rows[-500:] if len(rows) >= 500 else rows
n = max(1, len(tail))
win        = sum(1 for r in tail if r.get("agents_won")) / n
boss_hits  = sum(float(r.get("boss_hits",  0.0)) for r in tail) / n
zero       = sum(1 for r in tail if int(r.get("boss_hits", 0)) == 0) / n
agent_hits = sum(float(r.get("agent_hits", 0.0)) for r in tail) / n
timeout    = sum(1 for r in tail if not r.get("agents_won") and not r.get("boss_won")) / n
fire_dist  = sum(float(r.get("mean_fire_dist", 0.0)) for r in tail) / n
shots      = sum(float(r.get("agent_shots",  0.0)) for r in tail)
oob        = sum(float(r.get("fire_oob",     0.0)) for r in tail) / max(1.0, shots)
final_stage = rows[-1].get("curriculum_stage", -1) if rows else -1
comm_rate   = sum(float(r.get("comm_rate", 1.0)) for r in tail) / n

last_policy_std = last_entropy = ""
try:
    trainer_rows = []
    with open(trainer_log) as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" not in obj:
                trainer_rows.append(obj)
    if trainer_rows:
        last_policy_std = f"{float(trainer_rows[-1].get('policy_std_mean', 0.0)):.6f}"
        last_entropy    = f"{float(trainer_rows[-1].get('entropy',         0.0)):.6f}"
except FileNotFoundError:
    pass

writer = csv.writer(sys.stdout)
writer.writerow([
    run_id, mode, seed, len(rows),
    f"{win:.6f}", f"{boss_hits:.6f}", f"{zero:.6f}", f"{agent_hits:.6f}",
    f"{timeout:.6f}", f"{fire_dist:.6f}", f"{oob:.6f}",
    last_policy_std, last_entropy, final_stage,
    f"{comm_rate:.6f}", log_file, trainer_log,
])
PY
}

# ── Single run ─────────────────────────────────────────────────────────────────
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
    local wm_flags=""
    if [[ "$mode" != "mappo" ]]; then
        wm_flags="--wm-intention --wm-H $WM_H"
    fi

    echo "   env:   M=8 sight=$SIGHT fire=$FIRE boss_period=$BOSS_FIRE_PERIOD boss_lag=$BOSS_AIM_LAG"
    echo "   obs:   obs_dim=$OBS_DIM (patch=${SIGHT}x${SIGHT}*3 + 4 scalars)"
    echo "   train: update_every=$UPDATE_EVERY lr_pol=$LR_POL entropy=$ENTROPY_COEF"
    echo "   rewards: survive=$SURVIVE near=$NEAR_BOSS proximity=$PROXIMITY hit_boss=$HIT_BOSS hit_self=$HIT_SELF win=$WIN_REWARD"
    if [[ -n "$wm_flags" ]]; then
        echo "   ordering: world-model H=$WM_H rollout (wm_flags: $wm_flags)"
    else
        echo "   ordering: none (mappo — independent actions)"
    fi

    "$PYTHON" battleship/train_battleship.py "$weights_dir" \
        --init \
        --obs-dim "$OBS_DIM" \
        --n-agents "$AGENTS" \
        > "$stdout_dir/init.out" 2>&1

    rm -f "$weights_dir/traj.ready" "$weights_dir/weights.ready" "$weights_dir/traj.done"
    rm -f "$py_status_file" "$cpp_status_file"

    # Python trainer
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

    # C++ simulator
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
            --boss-fire-period "$BOSS_FIRE_PERIOD" \
            --boss-aim-lag "$BOSS_AIM_LAG" \
            --steps "$STEPS" \
            --survive "$SURVIVE" \
            --near-boss "$NEAR_BOSS" \
            --proximity "$PROXIMITY" \
            --hit-boss "$HIT_BOSS" \
            --hit-self "$HIT_SELF" \
            --win-reward "$WIN_REWARD" \
            --log-dir "$log_dir" \
            --curriculum \
            --update-every "$UPDATE_EVERY" \
            $wm_flags &
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
    printf 'python_pid=%s\ncpp_pid=%s\npython_status=%s\ncpp_status=%s\n' \
        "$py_pid" "$cpp_pid" "$py_status_file" "$cpp_status_file" > "$stdout_dir/pids.env"

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
                echo "Python trainer exited before C++ simulator finished (status=$py_status)." >&2
                cleanup_pair "" "$cpp_pid"
                tail_logs "$cpp_out" "$py_out"
                exit 1
            fi

            local cpp_grace=0
            while [[ ! -f "$cpp_status_file" ]]; do
                if [[ "$cpp_grace" -ge "$CPP_AFTER_PY_GRACE_SEC" ]]; then
                    echo "Python trainer exited cleanly, but C++ simulator kept running after ${CPP_AFTER_PY_GRACE_SEC}s." >&2
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
        echo "C++ simulator failed for $run_id (status=$cpp_status)." >&2
        cleanup_pair "$py_pid" ""
        tail_logs "$cpp_out" "$py_out"
        exit 1
    fi

    if [[ "$py_collected" -eq 0 ]]; then
        local waited=0
        while [[ ! -f "$py_status_file" ]]; do
            if [[ "$waited" -ge "$TRAINER_GRACE_SEC" ]]; then
                echo "Python trainer did not exit after C++ completion; killing it." >&2
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

    summarize_run "$log_file" "$trainer_log" "$run_id" "$mode" "$seed"
    echo "done: $run_id"
    echo "  log:     $log_file"
    echo "  trainer: $trainer_log"
    echo "  plot:    $fig_dir/${run_id}.png"
}

# ── Main loop: mode × seed ─────────────────────────────────────────────────────
for mode in $MODES; do
    for seed in $SEEDS; do
        run_one "$mode" "$seed"
    done
done

echo ""
echo "All runs complete."
echo "Summary: $SUMMARY_CSV"

echo ""
echo "Generating comparison plot..."
"$PYTHON" battleship/plot_comparison.py "$RUN_ROOT" \
    --out "$RUN_ROOT/comparison.png" \
    --modes $MODES
echo "Comparison plot: $RUN_ROOT/comparison.png"
echo "Run root: $RUN_ROOT"
