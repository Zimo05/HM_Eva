#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${1:-${DATASET:-dws}}"
VARIANT="${2:-${VARIANT:-13}}"
GPU="${GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage: bash run.sh [DATASET] [DWS_VARIANT]

Run this command from Baseline/Models/RMTPP. The script resolves the Baseline
project root automatically, so result and monitoring paths remain absolute.

DATASET:
  amazon
  retweet
  taxi
  stackoverflow
  taobao
  covid-policy-tracker   (aliases: covid, covid_policy_tracker)
  dws [13|15|17|20]      (aliases: dws_13, dws_15, dws_17, dws_20)

Examples:
  bash run.sh amazon
  bash run.sh taxi
  bash run.sh covid-policy-tracker
  bash run.sh dws 13
  bash run.sh dws_15
  GPU=1 EPOCHS=80 BATCH_SIZE=64 bash run.sh retweet
  GPU=-1 bash run.sh taxi

GPU=-1 runs on CPU. Other hyperparameters may also be overridden through
environment variables; see the dataset configuration branches in run.sh.
EOF
}

if (( $# > 2 )); then
  usage >&2
  exit 2
fi

case "$DATASET" in
  covid|covid-policy-tracker|covid_policy_tracker)
    DATASET="covid_policy_tracker"
    ;;
  dws_13|dws_15|dws_17|dws_20)
    VARIANT="${DATASET#dws_}"
    DATASET="dws"
    ;;
  dws13|dws15|dws17|dws20)
    VARIANT="${DATASET#dws}"
    DATASET="dws"
    ;;
esac

# Each branch contains the complete default configuration for one dataset.
# Every value can be overridden, for example:
#   GPU=1 LEARNING_RATE=0.0003 HIDDEN_SIZE=32 bash run.sh dws 13
case "$DATASET" in
  amazon)
    RESULT_ROOT="Models/RMTPP/Result/Amazon"
    RUN_NAME="rmtpp_amazon_${STAMP}"
    MODEL_RUN_NAME="amazon"

    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    MC_SAMPLES="${MC_SAMPLES:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-1}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-200}"
    DTIME_MAX="${DTIME_MAX:-2}"
    LR_PATIENCE="${LR_PATIENCE:-5}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SEED="${SEED:-2024}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    ;;

  dws)
    case "$VARIANT" in
      13|15|17|20) ;;
      *)
        echo "DWS variant must be one of: 13, 15, 17, 20" >&2
        exit 2
        ;;
    esac
    RESULT_ROOT="Models/RMTPP/Result/DWS"
    RUN_NAME="rmtpp_dws_${VARIANT}_${STAMP}"
    MODEL_RUN_NAME="dws_${VARIANT}"

    EPOCHS="${EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    MC_SAMPLES="${MC_SAMPLES:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-1}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-500}"
    DTIME_MAX="${DTIME_MAX:-120}"
    LR_PATIENCE="${LR_PATIENCE:-5}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SEED="${SEED:-2024}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    ;;

  covid_policy_tracker)
    RESULT_ROOT="Models/RMTPP/Result/Covid-Policy-Tracker"
    RUN_NAME="rmtpp_covid_policy_tracker_${STAMP}"
    MODEL_RUN_NAME="covid_policy_tracker"

    EPOCHS="${EPOCHS:-200}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
    MC_SAMPLES="${MC_SAMPLES:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-1}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-500}"
    DTIME_MAX="${DTIME_MAX:-60}"
    LR_PATIENCE="${LR_PATIENCE:-10}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-30}"
    MIN_DELTA="${MIN_DELTA:-0.0001}"
    SEED="${SEED:-2024}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    ;;

  taxi)
    RESULT_ROOT="Models/RMTPP/Result/Taxi"
    RUN_NAME="rmtpp_taxi_${STAMP}"
    MODEL_RUN_NAME="taxi"

    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    MC_SAMPLES="${MC_SAMPLES:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-1}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-300}"
    DTIME_MAX="${DTIME_MAX:-3}"
    LR_PATIENCE="${LR_PATIENCE:-5}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SEED="${SEED:-2024}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    ;;

  taobao)
    RESULT_ROOT="Models/RMTPP/Result/Taobao"
    RUN_NAME="rmtpp_taobao_${STAMP}"
    MODEL_RUN_NAME="taobao"

    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    MC_SAMPLES="${MC_SAMPLES:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-1}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-300}"
    DTIME_MAX="${DTIME_MAX:-1.5}"
    LR_PATIENCE="${LR_PATIENCE:-5}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SEED="${SEED:-2024}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    ;;

  retweet)
    RESULT_ROOT="Models/RMTPP/Result/Retweet"
    RUN_NAME="rmtpp_retweet_${STAMP}"
    MODEL_RUN_NAME="retweet"

    EPOCHS="${EPOCHS:-60}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    MC_SAMPLES="${MC_SAMPLES:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-1}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-100}"
    DTIME_MAX="${DTIME_MAX:-65000}"
    LR_PATIENCE="${LR_PATIENCE:-4}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SEED="${SEED:-2024}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    ;;

  stackoverflow)
    RESULT_ROOT="Models/RMTPP/Result/StackOverflow"
    RUN_NAME="rmtpp_stackoverflow_${STAMP}"
    MODEL_RUN_NAME="stackoverflow"

    EPOCHS="${EPOCHS:-80}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LEARNING_RATE="${LEARNING_RATE:-0.001}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
    MC_SAMPLES="${MC_SAMPLES:-20}"
    THINNING_NUM_SAMPLE="${THINNING_NUM_SAMPLE:-1}"
    THINNING_NUM_EXP="${THINNING_NUM_EXP:-300}"
    DTIME_MAX="${DTIME_MAX:-20}"
    LR_PATIENCE="${LR_PATIENCE:-5}"
    LR_FACTOR="${LR_FACTOR:-0.3}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
    MIN_DELTA="${MIN_DELTA:-0.001}"
    SEED="${SEED:-2024}"
    SELECTION_METRIC="${SELECTION_METRIC:-loglike}"
    ;;

  *)
    usage >&2
    exit 2
    ;;
esac

# The caller's shell stays in its original directory even though this script
# runs from PROJECT_ROOT.  Convert result paths to absolute paths so every
# printed watch/tail command works when copied from any directory.
RESULT_ROOT="$PROJECT_ROOT/$RESULT_ROOT"

# Select the first available Python that can actually import all dependencies.
python_has_dependencies() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  PYTHONPATH="$PROJECT_ROOT/Models/EasyTPP${PYTHONPATH:+:$PYTHONPATH}" \
    "$candidate" -c \
    "import numpy, torch, yaml, datasets, omegaconf, matplotlib, easy_tpp" \
    >/dev/null 2>&1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! python_has_dependencies "$PYTHON_BIN"; then
    echo "PYTHON_BIN cannot import the RMTPP/EasyTPP dependencies: $PYTHON_BIN" >&2
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
    "$PROJECT_ROOT/.venv-rmtpp/bin/python"
    "$PROJECT_ROOT/.venv/bin/python"
  )
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CANDIDATES+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_CANDIDATES+=("$(command -v python)")
  fi

  PYTHON_BIN=""
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if python_has_dependencies "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "No Python environment can import the RMTPP/EasyTPP dependencies." >&2
    echo "Create or activate the project environment:" >&2
    echo "  python3 -m venv .venv-rmtpp" >&2
    echo "  source .venv-rmtpp/bin/activate" >&2
    echo "  python -m pip install -e Models/EasyTPP" >&2
    echo "  python -m pip install matplotlib" >&2
    echo "Or set PYTHON_BIN=/absolute/path/to/python." >&2
    exit 1
  fi
fi

OUTPUT_DIR="${RESULT_ROOT}/${RUN_NAME}"
ARCHIVE="${OUTPUT_DIR}.tar.gz"
BOOTSTRAP_LOG="/tmp/${RUN_NAME}_nohup.log"
CONSOLE_LOG="${OUTPUT_DIR}/log/${MODEL_RUN_NAME}_console.log"

mkdir -p "$RESULT_ROOT"

PYTHONUNBUFFERED=1 nohup "$PYTHON_BIN" Models/RMTPP/run_experiment.py \
  --dataset "$DATASET" \
  --variant "$VARIANT" \
  --gpu "$GPU" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-size "$HIDDEN_SIZE" \
  --mc-samples "$MC_SAMPLES" \
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
if [[ "$DATASET" == "dws" ]]; then
  echo "Dataset: dws (variant $VARIANT)"
else
  echo "Dataset: $DATASET"
fi
echo "Config: epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE hidden=$HIDDEN_SIZE mc_samples=$MC_SAMPLES seed=$SEED"
echo "Eval: thinning_sample=$THINNING_NUM_SAMPLE thinning_exp=$THINNING_NUM_EXP dtime_max=$DTIME_MAX"
echo "Schedule: lr_patience=$LR_PATIENCE lr_factor=$LR_FACTOR early_stop=$EARLY_STOP_PATIENCE min_delta=$MIN_DELTA"
echo "Selection: validation_$SELECTION_METRIC"
echo "Log: $CONSOLE_LOG"
echo "Bootstrap log: $BOOTSTRAP_LOG"
echo "Result: $OUTPUT_DIR"
echo "Archive: $ARCHIVE"
printf "Watch: while [ ! -f '%s' ]; do " "$CONSOLE_LOG"
printf "if ! kill -0 %s 2>/dev/null; then cat '%s'; exit 1; fi; " "$PID" "$BOOTSTRAP_LOG"
printf "sleep 1; done; tail -f '%s'\n" "$CONSOLE_LOG"
echo "Check: ps -fp $PID"
