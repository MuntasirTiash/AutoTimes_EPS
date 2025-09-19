#!/usr/bin/env bash
set -euo pipefail

# ===== Paths you gave =====
ROOT_PATH="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel"
DATA_CSV="panel_gvkey_filtered.csv"
TEXT_INDEX="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/10-Q/embeddings_llama/gvkey_text_index.csv"
LLAMA_DIR="/ssd1/muntasir/Desktop/AutoTimes/llama-7b"

# ===== Runtime =====
GPU_ID="${GPU_ID:-0}"   # 0 or 1
SEQ_LEN=36; LABEL_LEN=32; TOKEN_LEN=4
TEST_SEQ_LEN=36; TEST_LABEL_LEN=32; TEST_PRED_LEN=4
BATCH=2; EPOCHS=5; LR=1e-3
TEXT_DIM="${TEXT_DIM:-4096}"

# Which runs to execute (space-separated): "no_text with_text" | "no_text" | "with_text"
RUNS="${RUNS:-no_text with_text}"

OUT_ROOT="${OUT_ROOT:-./runs_panel}"
LOG_DIR="${OUT_ROOT}/logs"; RES_DIR="${OUT_ROOT}/results"; CKPT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "$LOG_DIR" "$RES_DIR" "$CKPT_DIR"

SETTING="long_term_forecast_PANELCOV_${SEQ_LEN}_${TOKEN_LEN}_AutoTimes_Llama"
TR_FOLDER="./test_results/${SETTING}"

timestamp() { date +"%Y%m%d-%H%M%S"; }

package_results () {
  local dest="$1"

  # Copy repo's rolling summary if present
  [[ -f "result_long_term_forecast.txt" ]] && cp "result_long_term_forecast.txt" "${dest}/" || true

  # Bundle predictions + truth + metrics if present
  if [[ -f "${dest}/raw/predictions.npy" && -f "${dest}/raw/ground_truth.npy" ]]; then
    # Pass $dest as argv[1] to the here-doc'd Python via "python - "$dest" <<'PY'"
    python - "$dest" <<'PY'
import json, numpy as np, os, sys
base = sys.argv[1]
p = os.path.join(base, "raw", "predictions.npy")
t = os.path.join(base, "raw", "ground_truth.npy")
pred = np.load(p); true = np.load(t)
yhat = pred.reshape(-1); y = true.reshape(-1)
eps = 1e-8
mse = float(np.mean((yhat - y)**2))
mae = float(np.mean(np.abs(yhat - y)))
rmse = float(np.sqrt(mse))
mape = float(np.mean(np.abs((yhat - y)/(np.abs(y)+eps))))
mspe = float(np.mean(((yhat - y)/(np.abs(y)+eps))**2))
sst = float(np.sum((y - y.mean())**2)) or eps
sse = float(np.sum((y - yhat)**2))
r2  = float(1.0 - sse/sst)
kelly_r2 = float(1.0 - sse/(float(np.sum(y**2)) or eps))
metrics = dict(mse=mse, mae=mae, rmse=rmse, mape=mape, mspe=mspe, r2=r2, kelly_r2=kelly_r2)
np.savez(os.path.join(base, "results_bundle.npz"),
         predictions=pred, ground_truth=true, **metrics)
with open(os.path.join(base, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved NPZ + metrics to", base)
PY
  else
    echo "NOTE: predictions/ground_truth not found in ${dest}/raw — skipping NPZ bundle."
  fi
}

run_one () {
  local tag="$1"
  local log="${LOG_DIR}/$(timestamp)_${tag}.log"

  # Shared args as an array (safe)
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
    --panel_id_col gvkey --panel_time_col DATE --panel_y_col actual
    --llama_model_name "${LLAMA_DIR}"
    --cosine --tmax ${EPOCHS}
  )

  if [[ "$tag" == "with_text" ]]; then
    args+=( --use_text --text_mode emb --text_dim "${TEXT_DIM}" --text_index_csv "${TEXT_INDEX}" )
  fi

  echo
  echo "=============================================="
  echo "=== RUN: ${tag} | GPU (masked) = ${GPU_ID}"
  echo "=== TEXT: $([[ "$tag" == "with_text" ]] && echo ON || echo OFF)"
  echo "=============================================="
  echo

  # Freshen the repo's test output folder so we know what we copy
  rm -rf "${TR_FOLDER}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u run.py "${args[@]}" 2>&1 | tee "${log}"

  # Package artifacts from this run
  local dest="${RES_DIR}/$(timestamp)_${tag}"
  mkdir -p "${dest}/raw"
  if [[ -d "${TR_FOLDER}" ]]; then
    cp -r "${TR_FOLDER}/." "${dest}/raw/"
  else
    echo "WARN: expected ${TR_FOLDER} not found; copying nothing."
  fi

  package_results "${dest}"
  echo "Packaged -> ${dest}"
}

# Execute requested runs
for r in ${RUNS}; do
  run_one "${r}"
done