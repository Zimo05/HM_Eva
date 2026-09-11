#!/usr/bin/env bash
set -euo pipefail

THP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THP_DIR"

usage() {
  cat <<'EOF'
Usage: bash run.sh DATASET

The command starts training with nohup in the background. The terminal only
prints the PID and output paths; epoch details are written to console.log.

Available commands (run inside Models/THP):
  bash run.sh amazon
  bash run.sh retweet
  bash run.sh taxi
  bash run.sh stackoverflow
  bash run.sh taobao
  bash run.sh mobike
  bash run.sh mimic
  bash run.sh covid_policy_tracker
  bash run.sh dws_8
  bash run.sh dws_10
  bash run.sh dws_13
  bash run.sh dws_15
  bash run.sh dws_17
  bash run.sh dws_20

Aliases:
  covid, covid-policy-tracker, stackov, dws8, dws10, dws13, dws15, dws17, dws20

Examples with overrides:
  GPU=1 bash run.sh amazon
  GPU=-1 bash run.sh taxi
  EPOCHS=2 BATCH_SIZE=4 DEVICE=cpu bash run.sh covid
  LEARNING_RATE=0.00005 D_MODEL=128 NUM_LAYERS=3 bash run.sh dws_8

GPU=-1 forces CPU. All hyperparameters below can be overridden through
environment variables; no file editing is needed.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

DATASET="$1"
case "$DATASET" in
  covid|covid-policy-tracker)
    DATASET="covid_policy_tracker"
    ;;
  stackov)
    DATASET="stackoverflow"
    ;;
  dws8|dws10|dws13|dws15|dws17|dws20)
    DATASET="dws_${DATASET#dws}"
    ;;
esac

# Complete dataset-specific defaults. Environment values always take priority.
case "$DATASET" in
  amazon)
    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  retweet)
    EPOCHS="${EPOCHS:-60}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  taxi)
    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  stackoverflow)
    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  taobao)
    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  mobike)
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  mimic)
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    SELECTION_METRIC="${SELECTION_METRIC:-accuracy}"
    ;;
  covid_policy_tracker)
    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  dws_8|dws_10|dws_13|dws_15|dws_17|dws_20)
    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SELECTION_METRIC="${SELECTION_METRIC:-ll}"
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    usage >&2
    exit 2
    ;;
esac

LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-64}"
D_RNN="${D_RNN:-256}"
D_INNER="${D_INNER:-128}"
D_K="${D_K:-16}"
D_V="${D_V:-16}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
INTEGRAL_METHOD="${INTEGRAL_METHOD:-trapezoid}"
MC_SAMPLES="${MC_SAMPLES:-20}"
EVENT_LOSS_WEIGHT="${EVENT_LOSS_WEIGHT:-1.0}"
TYPE_LOSS_WEIGHT="${TYPE_LOSS_WEIGHT:-1.0}"
TIME_LOSS_WEIGHT="${TIME_LOSS_WEIGHT:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SEED="${SEED:-2024}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-auto}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thp_matplotlib_${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

python_has_dependencies() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  "$candidate" -c "import numpy, torch, matplotlib" >/dev/null 2>&1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! python_has_dependencies "$PYTHON_BIN"; then
    echo "PYTHON_BIN is missing THP dependencies: $PYTHON_BIN" >&2
    exit 1
  fi
else
  PYTHON_CANDIDATES=()
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_CANDIDATES+=("$VIRTUAL_ENV/bin/python")
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON_CANDIDATES+=("$CONDA_PREFIX/bin/python")
  fi
  PYTHON_CANDIDATES+=(
    "$THP_DIR/../../.venv-rmtpp/bin/python"
    "$THP_DIR/../../.venv/bin/python"
  )
  while IFS= read -r candidate; do
    PYTHON_CANDIDATES+=("$candidate")
  done < <(type -a -p python3 2>/dev/null || true)
  while IFS= read -r candidate; do
    PYTHON_CANDIDATES+=("$candidate")
  done < <(type -a -p python 2>/dev/null || true)

  PYTHON_BIN=""
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if python_has_dependencies "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "No Python environment can import numpy, torch, and matplotlib." >&2
    echo "Activate the THP environment or set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

if [[ "$GPU" == "-1" ]]; then
  DEVICE="cpu"
else
  export CUDA_DEVICE_ORDER="PCI_BUS_ID"
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="thp_${DATASET}_${STAMP}"
OUTPUT_DIR="$THP_DIR/Result/$DATASET/$RUN_NAME"
ARCHIVE="$THP_DIR/Result/$DATASET/${RUN_NAME}.tar.gz"
BOOTSTRAP_LOG="/tmp/${RUN_NAME}_launcher.log"
CONSOLE_LOG="$OUTPUT_DIR/log/console.log"

nohup "$PYTHON_BIN" "$THP_DIR/run_experiment.py" \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --d-model "$D_MODEL" \
  --d-rnn "$D_RNN" \
  --d-inner "$D_INNER" \
  --d-k "$D_K" \
  --d-v "$D_V" \
  --num-heads "$NUM_HEADS" \
  --num-layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --label-smoothing "$LABEL_SMOOTHING" \
  --integral-method "$INTEGRAL_METHOD" \
  --mc-samples "$MC_SAMPLES" \
  --event-loss-weight "$EVENT_LOSS_WEIGHT" \
  --type-loss-weight "$TYPE_LOSS_WEIGHT" \
  --time-loss-weight "$TIME_LOSS_WEIGHT" \
  --grad-clip "$GRAD_CLIP" \
  --selection-metric "$SELECTION_METRIC" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR" \
  --archive "$ARCHIVE" \
  --overwrite \
  >"$BOOTSTRAP_LOG" 2>&1 </dev/null &

PID=$!

echo "THP started in background."
echo "PID: $PID"
echo "Dataset: $DATASET"
echo "Config: epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE d_model=$D_MODEL layers=$NUM_LAYERS"
echo "Result: $OUTPUT_DIR"
echo "Archive: $ARCHIVE"
echo "Startup log: $BOOTSTRAP_LOG"
echo "Training log: $CONSOLE_LOG"
echo ""
echo "Ctrl+C or closing this terminal will not stop PID $PID."
echo "Check process: ps -fp $PID"
printf "View log: while [ ! -f '%s' ]; do sleep 1; done; tail -f '%s'\n" \
  "$CONSOLE_LOG" "$CONSOLE_LOG"
