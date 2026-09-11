#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${1:-mobike}"
GPU="${GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage: bash Models/FullyNN/run.sh DATASET

DATASET:
  mobike
  stackoverflow   (alias: stackov)

Examples:
  GPU=1 bash Models/FullyNN/run.sh mobike
  GPU=0 bash Models/FullyNN/run.sh stackoverflow
  GPU=1 EPOCHS=120 LEARNING_RATE=0.0005 bash Models/FullyNN/run.sh mobike
  GPU=-1 bash Models/FullyNN/run.sh mobike

GPU=-1 uses CPU. Every training parameter below can be overridden with an
environment variable.
EOF
}

if (( $# > 1 )); then
  usage >&2
  exit 2
fi

if [[ "$DATASET" == "stackov" ]]; then
  DATASET="stackoverflow"
fi

# Every dataset keeps its complete FullyNN configuration in one branch.
case "$DATASET" in
  mobike)
    RESULT_ROOT="Models/FullyNN/Result/MobikeData"
    RUN_NAME="fullynn_mobike_${STAMP}"

    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    NUM_LAYERS="${NUM_LAYERS:-1}"
    NUM_MLP_LAYERS="${NUM_MLP_LAYERS:-2}"
    HISTORY_WINDOW="${HISTORY_WINDOW:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-20}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-500}"
    DTIME_MAX="${DTIME_MAX:-336}"
    LR_PATIENCE="${LR_PATIENCE:-5}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    SEED="${SEED:-2024}"
    ;;

  stackoverflow)
    RESULT_ROOT="Models/FullyNN/Result/StackOverflow"
    RUN_NAME="fullynn_stackoverflow_${STAMP}"

    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    NUM_LAYERS="${NUM_LAYERS:-1}"
    NUM_MLP_LAYERS="${NUM_MLP_LAYERS:-2}"
    HISTORY_WINDOW="${HISTORY_WINDOW:-40}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-20}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-300}"
    DTIME_MAX="${DTIME_MAX:-20}"
    LR_PATIENCE="${LR_PATIENCE:-5}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    SEED="${SEED:-2024}"
    ;;

  *)
    usage >&2
    exit 2
    ;;
esac

# Prefer an explicitly selected or active environment, then the project venv.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 127
  fi
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv-rmtpp/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv-rmtpp/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python was not found. Activate .venv-rmtpp or set PYTHON_BIN." >&2
  exit 127
fi

if ! PYTHONPATH="$PROJECT_ROOT/Models/EasyTPP${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -c \
  "import numpy, torch, yaml, datasets, omegaconf, matplotlib, easy_tpp" \
  2>/dev/null; then
  echo "The selected Python is missing FullyNN/EasyTPP dependencies: $PYTHON_BIN" >&2
  echo "Activate the project environment first:" >&2
  echo "  source .venv-rmtpp/bin/activate" >&2
  exit 1
fi

OUTPUT_DIR="${RESULT_ROOT}/${RUN_NAME}"
ARCHIVE="${OUTPUT_DIR}.tar.gz"
BOOTSTRAP_LOG="/tmp/${RUN_NAME}_nohup.log"
CONSOLE_LOG="${OUTPUT_DIR}/log/${DATASET}_console.log"

mkdir -p "$RESULT_ROOT"

PYTHONUNBUFFERED=1 nohup "$PYTHON_BIN" Models/FullyNN/run_experiment.py \
  --dataset "$DATASET" \
  --gpu "$GPU" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-size "$HIDDEN_SIZE" \
  --num-layers "$NUM_LAYERS" \
  --num-mlp-layers "$NUM_MLP_LAYERS" \
  --history-window "$HISTORY_WINDOW" \
  --thinning-num-sample "$THINNING_NUM_SAMPLE" \
  --thinning-num-exp "$THINNING_NUM_EXP" \
  --dtime-max "$DTIME_MAX" \
  --lr-patience "$LR_PATIENCE" \
  --lr-factor "$LR_FACTOR" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --min-delta "$MIN_DELTA" \
  --selection-metric "$SELECTION_METRIC" \
  --output-dir "$OUTPUT_DIR" \
  --archive "$ARCHIVE" \
  --overwrite \
  >"$BOOTSTRAP_LOG" 2>&1 &

PID=$!

echo "PID: $PID"
echo "Python: $PYTHON_BIN"
echo "Dataset: $DATASET"
echo "Log: $CONSOLE_LOG"
echo "Bootstrap log: $BOOTSTRAP_LOG"
echo "Result: $OUTPUT_DIR"
echo "Archive: $ARCHIVE"
printf "Watch: while [ ! -f '%s' ]; do " "$CONSOLE_LOG"
printf "if ! kill -0 %s 2>/dev/null; then cat '%s'; exit 1; fi; " "$PID" "$BOOTSTRAP_LOG"
printf "sleep 1; done; tail -f '%s'\n" "$CONSOLE_LOG"
echo "Check: ps -fp $PID"
