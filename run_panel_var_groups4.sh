#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] line $LINENO"; exit 1' ERR

# ===================== FIXED PATHS (robust) =====================
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"   # <-- critical: run from repo root

# Data & schema (override with env vars if needed)
ROOT_PATH="${ROOT_PATH:-${REPO}/dataset/panel}"
DATA_CSV="${DATA_CSV:-panel_gvkey_nasdow.csv}"
ID_COL="${ID_COL:-gvkey}"
TIME_COL="${TIME_COL:-DATE}"
Y_COL="${Y_COL:-actual}"

# Feature groups JSON
VAR_GROUP_JSON="${VAR_GROUP_JSON:-${ROOT_PATH}/var_group.json}"

# LLaMA (your local checkpoint)
LLAMA_DIR="${LLAMA_DIR:-${REPO}/.models/llama-3.1-8b}"

# Text embeddings index (override if stored elsewhere)
TEXT_INDEX="${TEXT_INDEX:-${ROOT_PATH}/10-Q/embeddings_nasdow/gvkey_text_index.csv}"
TEXT_DIM="${TEXT_DIM:-4096}"

# ===================== RUNTIME =====================
GPU_ID="${GPU_ID:-0}"
SEQ_LEN=36; LABEL_LEN=32; TOKEN_LEN=4
TEST_SEQ_LEN=36; TEST_LABEL_LEN=32; TEST_PRED_LEN=4
BATCH=1; EPOCHS=7; LR=1e-3

OUT_ROOT="${OUT_ROOT:-${REPO}/runs_nasdow}"
LOG_DIR="${OUT_ROOT}/logs"
RES_DIR="${OUT_ROOT}/results"
CKPT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "$LOG_DIR" "$RES_DIR" "$CKPT_DIR"

SETTING="long_term_forecast_PANELCOV_${SEQ_LEN}_${TOKEN_LEN}_AutoTimes_Llama"
TR_FOLDER="${REPO}/test_results/${SETTING}"
# ===================================================

# (free_gpu, KEYS reader, helpers) ... keep as-is ...

run_one () {
  local tag="$1"; shift
  local log="${LOG_DIR}/$(date +'%Y%m%d-%H%M%S')_${tag}.log"

  local -a args=(
    --model AutoTimes_Llama
    --data panel_cov
    --root_path "${ROOT_PATH}"
    --data_path "${DATA_CSV}"
    --seq_len ${SEQ_LEN} --label_len ${LABEL_LEN} --token_len ${TOKEN_LEN}
    --test_seq_len ${TEST_SEQ_LEN} --test_label_len ${TEST_LABEL_LEN} --test_pred_len ${TEST_PRED_LEN}
    --batch_size ${BATCH} --learning_rate ${LR} --train_epochs ${EPOCHS}
    --gpu 0
    --checkpoints "${CKPT_DIR}"
    --panel_id_col "${ID_COL}" --panel_time_col "${TIME_COL}" --panel_y_col "${Y_COL}"
    --panel_cov_cols "${FEATS}"
    --llama_model_name "${LLAMA_DIR}"
    --cosine --tmax ${EPOCHS}
  )
  if [[ "$#" -gt 0 ]]; then args+=( "$@" ); fi

  echo; echo "=============================================="
  echo "=== RUN: ${tag} | GPU (masked) = ${GPU_ID}"
  echo "=============================================="; echo

  free_gpu
  rm -rf "${TR_FOLDER}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u "${REPO}/run.py" "${args[@]}" 2>&1 | tee "${log}"
  free_gpu
}