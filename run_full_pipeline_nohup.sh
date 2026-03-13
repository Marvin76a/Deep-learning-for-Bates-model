#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/full_pipeline_${TS}.log"

# Redirect both stdout and stderr to log file.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] Starting full SVJDM pipeline"
echo "Project dir: $PROJECT_DIR"
echo "Log file: $LOG_FILE"

cd "$PROJECT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate base || true
fi

echo "[$(date '+%F %T')] Installing dependencies"
python -m pip install -r requirements.txt

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START: $name"
  "$@"
  echo "[$(date '+%F %T')] DONE : $name"
}

run_step "Phase 1 accuracy" python experiments/phase1_accuracy.py
run_step "Phase 2 scaling" python experiments/phase2_scaling.py
run_step "Model ablation" python experiments/ablation_model.py
run_step "Structure ablation" python experiments/ablation_structure.py
run_step "Greeks extraction" python experiments/greeks_extraction.py
run_step "Phase 3 barrier level" python experiments/phase3_barrier_level.py
run_step "Phase 4 barrier jump" python experiments/phase4_barrier_jump.py
run_step "Robustness" python experiments/robustness.py

echo "[$(date '+%F %T')] Full SVJDM pipeline completed successfully"
echo "Final log: $LOG_FILE"
