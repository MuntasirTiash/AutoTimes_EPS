#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] line $LINENO"; exit 1' ERR

# ===================== FIXED PATHS (auto-detect repo root) ===================
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Data & schema
ROOT_PATH="${REPO}/dataset/panel"
DATA_CSV="panel_gvkey_cleaned.csv"     # <- your CSV
ID_COL="gvkey"                        # <- id column in CSV
TIME_COL="DATE"                       # <- time column in CSV
Y_COL="actual"                        # <- target column in CSV

# LLaMA (your local checkpoint under repo)
LLAMA_DIR="${REPO}/.models/llama-3.1-8b"

# ===================== RUNTIME =====================
GPU_ID="${GPU_ID:-0}"
SEQ_LEN=36
LABEL_LEN=32
TOKEN_LEN=4
TEST_SEQ_LEN=36
TEST_LABEL_LEN=32
TEST_PRED_LEN=4
BATCH=8
EPOCHS=2
LR=1e-3

OUT_ROOT="./runs_all_no_text_2"
LOG_DIR="${OUT_ROOT}/logs"
RES_DIR="${OUT_ROOT}/results"
CKPT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "$LOG_DIR" "$RES_DIR" "$CKPT_DIR"

SETTING="long_term_forecast_PANELCOV_${SEQ_LEN}_${TOKEN_LEN}_AutoTimes_Llama"
TR_FOLDER="./test_results/${SETTING}"

timestamp() { date +"%Y%m%d-%H%M%S"; }
sanitize()  { echo "$1" | tr ' /,:;|&' '_' | tr -cd '[:alnum:]_,-' | cut -c1-80; }

# --- Clear GPU between runs to reduce OOM risk (safe no-op if CPU) ---
free_gpu () {
  echo "Clearing GPU ${GPU_ID} cache..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<'PY' || true
import gc, sys
try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    print("GPU cache cleared.")
except Exception as e:
    print(f"GPU clear skipped: {e}", file=sys.stderr)
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${GPU_ID}" --gpu-reset >/dev/null 2>&1 || true
  fi
  sleep 2
}

# ---- Package predictions + metrics if test artifacts exist ----
package_results () {
  local dest="$1"
  mkdir -p "${dest}/raw"
  if [[ -d "${TR_FOLDER}" ]]; then
    cp -r "${TR_FOLDER}/." "${dest}/raw/"
  else
    echo "WARN: expected ${TR_FOLDER} not found; copying nothing."
  fi

  if [[ -f "${dest}/raw/predictions.npy" && -f "${dest}/raw/ground_truth.npy" ]]; then
    python - "$dest" <<'PY'
import json, numpy as np, os, sys
base=sys.argv[1]
p=os.path.join(base,"raw","predictions.npy"); t=os.path.join(base,"raw","ground_truth.npy")
pred=np.load(p); truth=np.load(t)
yhat=pred.reshape(-1); y=truth.reshape(-1); eps=1e-8
mse=float(np.mean((yhat-y)**2)); mae=float(np.mean(np.abs(yhat-y))); rmse=float(np.sqrt(mse))
mape=float(np.mean(np.abs((yhat-y)/(np.abs(y)+eps)))); mspe=float(np.mean(((yhat-y)/(np.abs(y)+eps))**2))
sst=float(np.sum((y-y.mean())**2)) or eps; sse=float(np.sum((y-yhat)**2))
r2=float(1.0 - sse/sst); kelly_r2=float(1.0 - sse/(float(np.sum(y**2)) or eps))
metrics=dict(mse=mse,mae=mae,rmse=rmse,mape=mape,mspe=mspe,r2=r2,kelly_r2=kelly_r2)
np.savez(os.path.join(base,"results_bundle.npz"), predictions=pred, ground_truth=truth, **metrics)
with open(os.path.join(base,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
print("Saved metrics ->", os.path.join(base,"metrics.json"))
PY
  else
    echo "NOTE: predictions/ground_truth not found in ${dest}/raw — skipping NPZ bundle."
  fi
}

# ---- Build FEATS = all columns except ID/TIME/Y (keeps order) ----
FEATS="$(python - "${ROOT_PATH}" "${DATA_CSV}" "${ID_COL}" "${TIME_COL}" "${Y_COL}" <<'PY'
import os, sys, pandas as pd
root, csv, idc, tc, yc = sys.argv[1:]
df = pd.read_csv(os.path.join(root, csv), nrows=1)
cols = list(df.columns)
drop = {idc, tc, yc}
feats = [c for c in cols if c not in drop]
if not feats:
    raise SystemExit("No covariates found after dropping ID/TIME/Y.")
print(",".join(feats))
PY
)"

NFEAT=$(( $(awk -F, '{print NF}' <<<"$FEATS") ))
echo "Using all covariates (${NFEAT}): ${FEATS}"

# ---- Single run: NO TEXT ----
TAG="all_no_text"
DEST="${RES_DIR}/$(timestamp)_${TAG}"
LOG="${LOG_DIR}/$(timestamp)_${TAG}.log"

echo; echo "=============================================="
echo "=== RUN: ${TAG} | GPU (masked) = ${GPU_ID}"
echo "=============================================="; echo

free_gpu
rm -rf "${TR_FOLDER}"

# Assemble args cleanly
declare -a ARGS=(
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

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u run.py "${ARGS[@]}" 2>&1 | tee "${LOG}"
free_gpu

# ---- Save config + metrics bundle ----
mkdir -p "${DEST}"
package_results "${DEST}"

cat > "${DEST}/run_config.json" <<JSON
{
  "group": "all",
  "text": false,
  "n_features": ${NFEAT},
  "features": "$(echo "${FEATS}" | sed 's/"/\\"/g')",
  "id_col": "${ID_COL}", "time_col": "${TIME_COL}", "y_col": "${Y_COL}",
  "root_path": "${ROOT_PATH}", "data_csv": "${DATA_CSV}",
  "llama_dir": "${LLAMA_DIR}",
  "seq_len": ${SEQ_LEN}, "label_len": ${LABEL_LEN}, "token_len": ${TOKEN_LEN},
  "test_seq_len": ${TEST_SEQ_LEN}, "test_label_len": ${TEST_LABEL_LEN}, "test_pred_len": ${TEST_PRED_LEN},
  "batch_size": ${BATCH}, "epochs": ${EPOCHS}, "lr": ${LR}
}
JSON

echo; echo "=== Done. See:"
echo " - ${DEST}"
echo " - ${LOG}"
