#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] line $LINENO"; exit 1' ERR

# ===================== FIXED PATHS (edit here if needed) =====================
# JSON with feature groups (at repo root, created earlier)
VAR_GROUP_JSON="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/var_group.json"

# Data & schema
ROOT_PATH="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel"
DATA_CSV="panel_gvkey_nasdow.csv"
ID_COL="gvkey"
TIME_COL="DATE"
Y_COL="actual"

# LLaMA & text embeddings
LLAMA_DIR="/ssd1/muntasir/Desktop/AutoTimes/llama-7b"
TEXT_INDEX="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/10-Q/embeddings_nasdow/gvkey_text_index.csv"
TEXT_DIM=4096

# Training/runtime knobs
GPU_ID=0
SEQ_LEN=36
LABEL_LEN=32
TOKEN_LEN=4
TEST_SEQ_LEN=36
TEST_LABEL_LEN=32
TEST_PRED_LEN=4
BATCH=1
EPOCHS=7
LR=1e-3

# Output roots
OUT_ROOT="./runs_nasdow"
LOG_DIR="${OUT_ROOT}/logs"
RES_DIR="${OUT_ROOT}/results"
CKPT_DIR="${OUT_ROOT}/checkpoints"
mkdir -p "$LOG_DIR" "$RES_DIR" "$CKPT_DIR"
# ============================================================================

# Derived repo test output folder (where run.py writes predictions)
SETTING="long_term_forecast_PANELCOV_${SEQ_LEN}_${TOKEN_LEN}_AutoTimes_Llama"
TR_FOLDER="./test_results/${SETTING}"

timestamp() { date +"%Y%m%d-%H%M%S"; }
sanitize()  { echo "$1" | tr ' /,:;|&' '_' | tr -cd '[:alnum:]_,-' | cut -c1-80; }

# ---- read ordered keys from var_group.json (put 'all' first if exists) ----
[[ -f "${VAR_GROUP_JSON}" ]] || { echo "Missing ${VAR_GROUP_JSON}"; exit 2; }

readarray -t KEYS < <(python - "$VAR_GROUP_JSON" <<'PY'
import json, sys
vg=json.load(open(sys.argv[1]))
keys=list(vg.keys())
if 'all' in keys:
    keys.insert(0, keys.pop(keys.index('all')))
print("\n".join(keys))
PY
)

# ---- helpers ----
package_results () {
  local dest="$1"
  mkdir -p "${dest}/raw"
  if [[ -d "${TR_FOLDER}" ]]; then
    cp -r "${TR_FOLDER}/." "${dest}/raw/"
  else
    echo "WARN: expected ${TR_FOLDER} not found; copying nothing."
  fi

  # bundle arrays + metrics if present
  if [[ -f "${dest}/raw/predictions.npy" && -f "${dest}/raw/ground_truth.npy" ]]; then
    python - "$dest" <<'PY'
import json, numpy as np, os, sys
base=sys.argv[1]
p=os.path.join(base,"raw","predictions.npy"); t=os.path.join(base,"raw","ground_truth.npy")
pred=np.load(p); true=np.load(t)
yhat=pred.reshape(-1); y=true.reshape(-1); eps=1e-8
mse=float(np.mean((yhat-y)**2)); mae=float(np.mean(np.abs(yhat-y))); rmse=float(np.sqrt(mse))
mape=float(np.mean(np.abs((yhat-y)/(np.abs(y)+eps)))); mspe=float(np.mean(((yhat-y)/(np.abs(y)+eps))**2))
sst=float(np.sum((y-y.mean())**2)) or eps; sse=float(np.sum((y-yhat)**2))
r2=float(1.0 - sse/sst); kelly_r2=float(1.0 - sse/(float(np.sum(y**2)) or eps))
metrics=dict(mse=mse,mae=mae,rmse=rmse,mape=mape,mspe=mspe,r2=r2,kelly_r2=kelly_r2)
np.savez(os.path.join(base,"results_bundle.npz"), predictions=pred, ground_truth=true, **metrics)
with open(os.path.join(base,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
print("Saved metrics ->", os.path.join(base,"metrics.json"))
PY
  else
    echo "NOTE: predictions/ground_truth not found in ${dest}/raw — skipping NPZ bundle."
  fi
}

append_summary () {
  local dest="$1" group="$2" textflag="$3" nfeat="$4"
  local mfile="${dest}/metrics.json"
  [[ -f "$mfile" ]] || { echo "WARN: $mfile missing; skip summary row"; return; }
  python - "$mfile" "$group" "$textflag" "$nfeat" "$dest" <<'PY'
import json, sys, os
m=json.load(open(sys.argv[1]))
group=sys.argv[2]; textflag=sys.argv[3]; nfeat=int(sys.argv[4]); dest=sys.argv[5]
row=[group,textflag,str(nfeat),dest,str(m.get("mse")),str(m.get("mae")),str(m.get("rmse")),str(m.get("r2")),str(m.get("kelly_r2"))]
out="runs_panel/summary.csv"
if not os.path.exists(out):
    open(out,"w").write("group,text,n_features,dest,mse,mae,rmse,r2,kelly_r2\n")
with open(out,"a") as f: f.write(",".join(row)+"\n")
print("Appended to", out, "->", ",".join(row))
PY
}

run_one () {
  local tag="$1"; shift
  local log="${LOG_DIR}/$(timestamp)_${tag}.log"

  # shared args
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

  if [[ "$#" -gt 0 ]]; then
    args+=( "$@" )   # e.g., text flags
  fi

  echo; echo "=============================================="
  echo "=== RUN: ${tag} | GPU (masked) = ${GPU_ID}"
  echo "=============================================="; echo

  rm -rf "${TR_FOLDER}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u run.py "${args[@]}" 2>&1 | tee "${log}"
}

# Preflight: require text index for with_text runs
[[ -f "${TEXT_INDEX}" ]] || { echo "Missing TEXT_INDEX: ${TEXT_INDEX}"; exit 2; }

# ===================== MAIN LOOP =====================
for KEY in "${KEYS[@]}"; do
  GROUP="$(sanitize "${KEY}")"
  echo; echo ">>>> Processing group: ${GROUP} <<<<"

  # Build FEATS string from JSON, keep only columns present in CSV header
  set +e
  FEATS="$(python - "${VAR_GROUP_JSON}" "${ROOT_PATH}" "${DATA_CSV}" "${KEY}" <<'PY'
import json, os, sys, pandas as pd
vg=json.load(open(sys.argv[1])); root=sys.argv[2]; csv=sys.argv[3]; group=sys.argv[4]
cols=set(pd.read_csv(os.path.join(root,csv), nrows=1).columns)
want=list(dict.fromkeys(vg[group]))  # preserve insertion order, dedupe
feats=[c for c in want if c in cols]
missing=[c for c in want if c not in cols]
if missing:
    print(f"WARN[{group}]: dropping missing columns: {', '.join(missing)}", file=sys.stderr)
if not feats:
    print(f"SKIP[{group}]: no features exist in dataset header", file=sys.stderr); sys.exit(3)
print(",".join(feats))
PY
  )"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "Skipping group ${GROUP} due to no valid features."
    continue
  fi

  NFEAT=$(( $(awk -F, '{print NF}' <<<"$FEATS") ))

  # -------- 1) No text --------
  TAG_NO="${GROUP}_no_text"
  DEST_NO="${RES_DIR}/$(timestamp)_${TAG_NO}"
  run_one "${TAG_NO}"
  mkdir -p "${DEST_NO}"; package_results "${DEST_NO}"

  cat > "${DEST_NO}/run_config.json" <<JSON
{
  "group": "${GROUP}",
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
  append_summary "${DEST_NO}" "${GROUP}" "no_text" "${NFEAT}"

  # -------- 2) With text (embeddings) --------
  TAG_TX="${GROUP}_with_text"
  DEST_TX="${RES_DIR}/$(timestamp)_${TAG_TX}"
  run_one "${TAG_TX}" \
    --use_text \
    --text_mode emb \
    --text_dim "${TEXT_DIM}" \
    --text_index_csv "${TEXT_INDEX}"
  mkdir -p "${DEST_TX}"; package_results "${DEST_TX}"

  cat > "${DEST_TX}/run_config.json" <<JSON
{
  "group": "${GROUP}",
  "text": true,
  "n_features": ${NFEAT},
  "features": "$(echo "${FEATS}" | sed 's/"/\\"/g')",
  "id_col": "${ID_COL}", "time_col": "${TIME_COL}", "y_col": "${Y_COL}",
  "root_path": "${ROOT_PATH}", "data_csv": "${DATA_CSV}",
  "llama_dir": "${LLAMA_DIR}",
  "text_index_csv": "${TEXT_INDEX}", "text_dim": ${TEXT_DIM},
  "seq_len": ${SEQ_LEN}, "label_len": ${LABEL_LEN}, "token_len": ${TOKEN_LEN},
  "test_seq_len": ${TEST_SEQ_LEN}, "test_label_len": ${TEST_LABEL_LEN}, "test_pred_len": ${TEST_PRED_LEN},
  "batch_size": ${BATCH}, "epochs": ${EPOCHS}, "lr": ${LR}
}
JSON
  append_summary "${DEST_TX}" "${GROUP}" "with_text" "${NFEAT}"
done

echo; echo "=== Done. See ${RES_DIR}/* and runs_panel/summary.csv ==="