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
EPISODES="${EPISODES:-50000}"
SEEDS="${SEEDS:-0}"

# ── Experiment selection ───────────────────────────────────────────────────────
# EXP=baseline  → standard curriculum runs (default)
# EXP=exp1      → Exp1: world-model intention + comms-loss variants
# EXP=exp2      → Exp2: optional comm gate with 3 penalty levels
EXP="${EXP:-baseline}"

# Per-experiment variable defaults — applied early so SINGLE_CONFIG parallel
# launches also pick them up (the per-EXP CONFIGS block is skipped by SINGLE_CONFIG).
if [[ "$EXP" == "scale3" ]]; then
    AGENTS="${AGENTS:-5}"
    BOSS="${BOSS:-5}"
    M="${M:-12}"
    WM_H="${WM_H:-3}"
    EPISODES="${EPISODES:-60000}"
fi

AGENTS="${AGENTS:-2}"
BOSS="${BOSS:-1}"
M="${M:-8}"           # grid side length (8 for exp1/exp2, 10 for scale3)
SIGHT="${SIGHT:-2}"
FIRE="${FIRE:-2}"
BOSS_FIRE_PERIOD="${BOSS_FIRE_PERIOD:-8}"
BOSS_AIM_LAG="${BOSS_AIM_LAG:-1}"
STEPS="${STEPS:-60}"
SURVIVE="${SURVIVE:-0.005}"
PROXIMITY="${PROXIMITY:-0.02}"
HIT_BOSS="${HIT_BOSS:-2.0}"
HIT_SELF="${HIT_SELF:--0.3}"
WIN_REWARD="${WIN_REWARD:-10}"
CURRICULUM="${CURRICULUM:-0}"  # set to 1 to enable boss curriculum

LR_ENC="${LR_ENC:-0.0001}"
LR_WORLD="${LR_WORLD:-0.0003}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TRAINER_GRACE_SEC="${TRAINER_GRACE_SEC:-60}"
CPP_AFTER_PY_GRACE_SEC="${CPP_AFTER_PY_GRACE_SEC:-300}"

# Experiment 1 flags (passed through to sim)
WM_INTENTION="${WM_INTENTION:-0}"    # 1 = use world-model intention ordering
WM_H="${WM_H:-2}"                    # rollout horizon for WM
COMMS_LOSS="${COMMS_LOSS:-0.0}"      # fraction of steps with random ordering

# Experiment 2 flags
COMM_GATE="${COMM_GATE:-0}"          # 1 = enable comm gate
COMM_PENALTY="${COMM_PENALTY:-0.0}"  # reward cost per comm step

OBS_DIM=$(( (2 * SIGHT + 1) * (2 * SIGHT + 1) * 3 + 4 ))
RUN_ROOT="${RUN_ROOT:-runs/battleship_grid/$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="$RUN_ROOT/summary.csv"
current_py_pid=""
current_cpp_pid=""
current_run_id=""

# name | update_every | lr_policy | entropy_coef | near_boss
# Global reward overrides: SURVIVE, PROXIMITY, WIN_REWARD env vars.
# To run a single config: SINGLE_CONFIG="curriculum|8|0.0003|0.003|0.10" bash run_battleship_grid.sh
if [[ -n "${SINGLE_CONFIG:-}" ]]; then
    CONFIGS=("$SINGLE_CONFIG")
elif [[ "$EXP" == "exp1" ]]; then
    # Experiment 1: compare critic proxy vs world-model intention ordering,
    # with four comms-loss levels (0%, 10%, 20%, 30%).
    # The first config uses critic (baseline_critic); the rest use WM intention.
    #   name           | update_every | lr_policy | entropy | near_boss
    CONFIGS=(
        "baseline_critic|64|0.0003|0.003|0.15"
        "wm_loss00|64|0.0003|0.003|0.15"
        "wm_loss10|64|0.0003|0.003|0.15"
        "wm_loss20|64|0.0003|0.003|0.15"
        "wm_loss30|64|0.0003|0.003|0.15"
    )
elif [[ "$EXP" == "exp2" ]]; then
    # Experiment 2: comm gate with three penalty sizes.
    # All use WM intention for ordering when comm succeeds.
    #   name              | update_every | lr_policy | entropy | near_boss
    CONFIGS=(
        "commgate_tiny|64|0.0003|0.003|0.15"
        "commgate_medium|64|0.0003|0.003|0.15"
        "commgate_large|64|0.0003|0.003|0.15"
    )
elif [[ "$EXP" == "scale3" ]]; then
    # Scale-3: 3 agents vs 2 bosses on a 10×10 grid, H=3 WM rollout.
    # Tests Option A: does ordering signal value compound with more agents in the queue?
    # The fourth config (commgate_large) is the 0%-comm contrast from exp2.
    # Variable defaults (AGENTS=3 BOSS=2 M=10 WM_H=3 EPISODES=60000) are set above
    # so they also apply when using SINGLE_CONFIG for parallel launches.
    #   name              | update_every | lr_policy | entropy | near_boss
    CONFIGS=(
        "baseline_critic|64|0.0003|0.003|0.15"
        "wm_loss00|64|0.0003|0.003|0.15"
        "wm_loss30|64|0.0003|0.003|0.15"
        "commgate_large|64|0.0003|0.003|0.15"
    )
else
    CONFIGS=(
        "curriculum|64|0.0003|0.003|0.15"
        "curriculum_hi_ent|64|0.0003|0.010|0.15"
        "curriculum_lo_lr|64|0.0001|0.003|0.15"
    )
fi

find_existing_battleship_processes() {
    local matches=""
    if command -v pgrep >/dev/null 2>&1; then
        matches="$(pgrep -fl 'battleship/train_battleship.py|battleship-sim' 2>/dev/null || true)"
    fi
    if [[ -n "$matches" ]]; then
        echo "$matches"
        return
    fi
    ps -axo pid=,command= 2>/dev/null | awk '
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

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "Ensuring battleship-sim is built in $BUILD_DIR ..."
    make -C "$BUILD_DIR" battleship-sim
elif [[ ! -x "$SIM_BIN" ]]; then
    echo "error: simulator not found or not executable: $SIM_BIN" >&2
    echo "rerun without SKIP_BUILD=1 to build it automatically" >&2
    exit 1
fi

printf 'run_id,config,seed,episodes_logged,last500_win,last500_boss_hits,last500_zero_hit,last500_agent_hits,last500_timeout,last500_fire_dist,last500_oob_rate,last_policy_std,last_entropy,final_curriculum_stage,last500_comm_rate,log_file,trainer_log\n' > "$SUMMARY_CSV"

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
final_stage = rows[-1].get("curriculum_stage", -1) if rows else -1
comm_rate = sum(float(r.get("comm_rate", 1.0)) for r in tail) / n

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
    final_stage,
    f"{comm_rate:.6f}",
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
    local cpp_status_file="$stdout_dir/cpp.status"
    local py_status_file="$stdout_dir/python.status"

    if [[ -e "$run_dir" ]]; then
        echo "Refusing to overwrite existing run directory: $run_dir" >&2
        exit 1
    fi

    mkdir -p "$weights_dir" "$log_dir" "$fig_dir" "$stdout_dir"

    echo ""
    echo "== $run_id =="
    echo "config: update_every=$update_every lr_policy=$lr_policy entropy=$entropy_coef near_boss=$near_boss boss_period=$BOSS_FIRE_PERIOD boss_lag=$BOSS_AIM_LAG seed=$seed"
    echo "note: curriculum configs override boss_period/boss_lag by stage"

    # Comm gate init flag (Experiment 2, and scale3 commgate_large)
    comm_gate_init_flag=""
    if [[ "$EXP" == "exp2" ]] || [[ "${COMM_GATE:-0}" == "1" ]] || { [[ "$EXP" == "scale3" ]] && [[ "$config_name" == commgate* ]]; }; then
        comm_gate_init_flag="--comm-gate"
    fi

    "$PYTHON" battleship/train_battleship.py "$weights_dir" \
        --init \
        --obs-dim "$OBS_DIM" \
        --n-agents "$AGENTS" \
        $comm_gate_init_flag \
        > "$stdout_dir/init.out" 2>&1

    rm -f "$weights_dir/traj.ready" "$weights_dir/weights.ready" "$weights_dir/traj.done"

    rm -f "$py_status_file" "$cpp_status_file"

    # Comm gate trainer flags (Experiment 2, and scale3 commgate_large)
    comm_gate_train_flags=""
    if [[ "$EXP" == "exp2" ]] || [[ "${COMM_GATE:-0}" == "1" ]] || { [[ "$EXP" == "scale3" ]] && [[ "$config_name" == commgate* ]]; }; then
        comm_gate_train_flags="--comm-gate"
        case "$config_name" in
            commgate_tiny)   comm_gate_train_flags="$comm_gate_train_flags --comm-penalty 0.005" ;;
            commgate_medium) comm_gate_train_flags="$comm_gate_train_flags --comm-penalty 0.050" ;;
            commgate_large)  comm_gate_train_flags="$comm_gate_train_flags --comm-penalty 0.500" ;;
            *)               comm_gate_train_flags="$comm_gate_train_flags --comm-penalty ${COMM_PENALTY:-0.0}" ;;
        esac
    fi

    (
        set +e
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
            $comm_gate_train_flags &
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
        curriculum_flag=""
        if [[ "$config_name" == curriculum* ]] || [[ "${CURRICULUM:-0}" == "1" ]] || [[ "$EXP" == "exp1" ]] || [[ "$EXP" == "exp2" ]] || [[ "$EXP" == "scale3" ]]; then
            curriculum_flag="--curriculum"
        fi

        # ── Experiment 1: world-model intention flags ──────────────────────────
        exp1_flags=""
        if [[ "$EXP" == "exp1" && "$config_name" != "baseline_critic" ]]; then
            exp1_flags="--wm-intention --wm-H $WM_H"
            # Per-config comms-loss values
            case "$config_name" in
                wm_loss00) exp1_flags="$exp1_flags --comms-loss 0.00" ;;
                wm_loss10) exp1_flags="$exp1_flags --comms-loss 0.10" ;;
                wm_loss20) exp1_flags="$exp1_flags --comms-loss 0.20" ;;
                wm_loss30) exp1_flags="$exp1_flags --comms-loss 0.30" ;;
            esac
        elif [[ "${WM_INTENTION:-0}" == "1" ]]; then
            exp1_flags="--wm-intention --wm-H $WM_H --comms-loss $COMMS_LOSS"
        fi

        # ── Experiment 2: comm gate flags ──────────────────────────────────────
        exp2_flags=""
        if [[ "$EXP" == "exp2" ]]; then
            exp2_flags="--comm-gate --wm-intention --wm-H $WM_H"
            case "$config_name" in
                commgate_tiny)   exp2_flags="$exp2_flags --comm-penalty 0.005" ;;
                commgate_medium) exp2_flags="$exp2_flags --comm-penalty 0.050" ;;
                commgate_large)  exp2_flags="$exp2_flags --comm-penalty 0.500" ;;
            esac
        elif [[ "${COMM_GATE:-0}" == "1" ]]; then
            exp2_flags="--comm-gate --comm-penalty $COMM_PENALTY"
        fi

        # ── Scale-3: 3-agent/2-boss/10×10, H=3 WM, per-config flags ──────────
        scale3_flags=""
        if [[ "$EXP" == "scale3" ]]; then
            case "$config_name" in
                wm_loss00)      scale3_flags="--wm-intention --wm-H $WM_H" ;;
                wm_loss30)      scale3_flags="--wm-intention --wm-H $WM_H --comms-loss 0.30" ;;
                commgate_large) scale3_flags="--comm-gate --wm-intention --wm-H $WM_H --comm-penalty 0.500" ;;
                baseline_critic) scale3_flags="" ;;
            esac
        fi

        "$SIM_BIN" "$weights_dir" \
            --mode "$MODE" \
            --episodes "$EPISODES" \
            --seed "$seed" \
            --M "$M" \
            --agents "$AGENTS" \
            --boss "$BOSS" \
            --sight "$SIGHT" \
            --fire "$FIRE" \
            --boss-fire-period "$BOSS_FIRE_PERIOD" \
            --boss-aim-lag "$BOSS_AIM_LAG" \
            --steps "$STEPS" \
            --survive "$SURVIVE" \
            --near-boss "$near_boss" \
            --proximity "$PROXIMITY" \
            --hit-boss "$HIT_BOSS" \
            --hit-self "$HIT_SELF" \
            --win-reward "$WIN_REWARD" \
            --log-dir "$log_dir" \
            $curriculum_flag $exp1_flags $exp2_flags $scale3_flags &
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
