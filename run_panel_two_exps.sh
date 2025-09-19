#!/usr/bin/env bash
set -euo pipefail

# ====== USER PATHS ======
ROOT_PATH="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel"
DATA_CSV="panel_gvkey_filtered.csv"
TEXT_INDEX="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/10-Q/embeddings_llama/gvkey_text_index.csv"
LLAMA_DIR="/ssd1/muntasir/Desktop/AutoTimes/llama-7b"

# ====== RUNTIME / EXP SETTINGS ======
GPU_ID="${GPU_ID:-0}"   # set to 0 or 1 when you call the script
SEQ_LEN=36
LABEL_LEN=32
TOKEN_LEN=4
TEST_SEQ_LEN=36
TEST_LABEL_LEN=32
TEST_PRED_LEN=4
BATCH=2
EPOCHS=2
LR=1e-3

OUT_ROOT="${OUT_ROOT:-./runs_panel}"
LOG_DIR="${OUT_ROOT}/logs"
RES_DIR="${OUT_ROOT}/results"
CKPT_DIR="${OUT_ROOT}/checkpoints"

mkdir -p "$LOG_DIR" "$RES_DIR" "$CKPT_DIR"

SETTING="long_term_forecast_PANELCOV_${SEQ_LEN}_${TOKEN_LEN}_AutoTimes_Llama"
TR_FOLDER="./test_results/${SETTING}"

timestamp() { date +"%Y%m%d-%H%M%S"; }

collect_and_package () {
  local tag="$1"
  local stamp; stamp="$(timestamp)"
  local dest="${RES_DIR}/${stamp}_${tag}"
  mkdir -p "${dest}/raw"

  if [[ -d "${TR_FOLDER}" ]]; then
    cp -r "${TR_FOLDER}/." "${dest}/raw/"
  else
    echo "WARN: expected ${TR_FOLDER} not found; skipping copy." >&2
  fi

  if [[ -f "${dest}/raw/predictions.npy" && -f "${dest}/raw/ground_truth.npy" ]]; then
    python - <<'PY'
import json, numpy as np, sys, os
base = sys.argv[1]
pred = np.load(os.path.join(base, "raw", "predictions.npy"))
true = np.load(os.path.join(base, "raw", "ground_truth.npy"))
yhat = pred.reshape(-1); y = true.reshape(-1)
eps = 1e-8
mse = float(np.mean((yhat - y)**2))
mae = float(np.mean(np.abs(yhat - y)))
rmse = float(np.sqrt(mse))
mape = float(np.mean(np.abs((yhat - y) / (np.abs(y) + eps))))
mspe = float(np.mean(((yhat - y) / (np.abs(y) + eps))**2))
sst = float(np.sum((y - y.mean())**2)) or eps
sse = float(np.sum((y - yhat)**2))
r2  = float(1.0 - sse/sst)
den = float(np.sum(y**2)) or eps
kelly_r2 = float(1.0 - sse/den)
metrics = dict(mse=mse, mae=mae, rmse=rmse, mape=mape, mspe=mspe, r2=r2, kelly_r2=kelly_r2)
np.savez(os.path.join(base, "results_bundle.npz"),
         predictions=pred, ground_truth=true, **metrics)
with open(os.path.join(base, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved:", os.path.join(base, "results_bundle.npz"))
print("Metrics:", json.dumps(metrics))
PY
    "${dest}"
  else
    echo "NOTE: no predictions found to package in ${dest}" >&2
  fi

  [[ -f "result_long_term_forecast.txt" ]] && cp "result_long_term_forecast.txt" "${dest}/"
  echo "Packaged results -> ${dest}"
}

run_one () {
  local tag="$1"; shift
  local log="${LOG_DIR}/$(timestamp)_${tag}.log"

  echo "=== Running: ${tag} on GPU ${GPU_ID} (masked) ==="
  # Mask to a single physical GPU; inside the process, GPU '0' means this device.
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python -u run.py \
    --model AutoTimes_Llama \
    --data panel_cov \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_CSV}" \
    --seq_len ${SEQ_LEN} --label_len ${LABEL_LEN} --token_len ${TOKEN_LEN} \
    --test_seq_len ${TEST_SEQ_LEN} --test_label_len ${TEST_LABEL_LEN} --test_pred_len ${TEST_PRED_LEN} \
    --batch_size ${BATCH} --learning_rate ${LR} --train_epochs ${EPOCHS} \
    --gpu 1 \                # <<<<<<<<<<<<<< numeric index (NOT "cuda:0")
    --checkpoints "${CKPT_DIR}" \
    --panel_id_col "GVKEY" --panel_time_col "DATE" --panel_y_col "actual" \
    --llama_model_name "${LLAMA_DIR}" \
    "$@" 2>&1 | tee "${log}"

  collect_and_package "${tag}"
}

rm -rf "${TR_FOLDER}"

# 1) No text
run_one "no_text"

# 2) With text
run_one "with_text" \
  --use_text \
  --text_mode emb \
  --text_dim 4096 \
  --text_index_csv "${TEXT_INDEX}"