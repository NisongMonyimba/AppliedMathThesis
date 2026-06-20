#!/usr/bin/env bash
#
# runall.sh — run the full thesis pipeline end-to-end, in dependency order.
#
# Order: SDE convergence sims -> particle filter -> RL training ->
#        crowding measure / Sobol sensitivity -> empirical backtest ->
#        figure generation.
#
# Usage:
#   chmod +x runall.sh
#   ./runall.sh                 # run everything
#   ./runall.sh --only backtest # run a single stage (see STAGE names below)
#   ./runall.sh --list          # list stage names
#
# Assumes you're in the repo root and have activated the venv
# (source .venv/bin/activate) with requirements.txt installed.

set -euo pipefail

REPO_ROOT="$PWD"
cd "$REPO_ROOT"

if [[ ! -d "manuscript" || ! -d "code" || ! -d "simulations" ]]; then
    echo "ERROR: not in the thesis repo root (expected manuscript/, code/, simulations/ here)."
    echo "       cd into msc-thesis-mfg/ and run: ./runall.sh"
    exit 1
fi

LOG_DIR="simulations/outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

declare -A STAGES
STAGES[euler_milstein]="python simulations/euler_milstein/run_convergence_simulation.py"
STAGES[particle_filter]="python simulations/particle_filter/run_particle_filter.py"
STAGES[rl_training]="python simulations/rl_training/train_multitimescale_agent.py"
STAGES[sobol_sensitivity]="python simulations/sobol_sensitivity/run_sobol_analysis.py"
STAGES[backtest_sp500]="python simulations/backtest_sp500/run_backtest.py"
STAGES[walk_forward]="python simulations/backtest_sp500/walk_forward_validation.py"
STAGES[figures]="python code/scripts/generate_figures.py"

STAGE_ORDER=(
    euler_milstein
    particle_filter
    rl_training
    sobol_sensitivity
    backtest_sp500
    walk_forward
    figures
)

list_stages() {
    echo "Available stages (in run order):"
    for s in "${STAGE_ORDER[@]}"; do
        echo "  - $s   (${STAGES[$s]})"
    done
}

run_stage() {
    local name="$1"
    local cmd="${STAGES[$name]}"
    local logfile="${LOG_DIR}/${name}_${TIMESTAMP}.log"

    echo "----------------------------------------------------------------"
    echo "==> Running stage: $name"
    echo "    Command: $cmd"
    echo "    Log:     $logfile"
    echo "----------------------------------------------------------------"

    if [[ ! -s "${cmd#* }" ]]; then
        echo "    [SKIP] script is empty (not yet implemented): ${cmd#* }"
        echo "skipped: empty script" > "$logfile"
        return 0
    fi

    if $cmd 2>&1 | tee "$logfile"; then
        echo "==> Stage '$name' completed OK"
    else
        echo "==> Stage '$name' FAILED — see $logfile"
        exit 1
    fi
}

if [[ "${1:-}" == "--list" ]]; then
    list_stages
    exit 0
fi

if [[ "${1:-}" == "--only" ]]; then
    target="${2:-}"
    if [[ -z "$target" || -z "${STAGES[$target]+x}" ]]; then
        echo "ERROR: unknown or missing stage name after --only"
        list_stages
        exit 1
    fi
    run_stage "$target"
    exit 0
fi

echo "==> Running full pipeline (${#STAGE_ORDER[@]} stages)"
for stage in "${STAGE_ORDER[@]}"; do
    run_stage "$stage"
done

echo ""
echo "=============================================="
echo " All stages complete. Logs in: $LOG_DIR"
echo "=============================================="
