# make_fake_text_embeds.py
import os, hashlib
import numpy as np
import pandas as pd

PANEL_CSV   = "/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/panel_short.csv"    
OUT_DIR     = "/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/embedding"    
INDEX_CSV   = os.path.join(OUT_DIR, "text_index.csv")
EMB_DIM     = 256  # must match --text_dim

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(PANEL_CSV)
if "DATE" not in df.columns or "PERMNO" not in df.columns:
    raise ValueError("Panel CSV must have DATE and PERMNO columns")

df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
pairs = df[["PERMNO", "DATE"]].drop_duplicates().reset_index(drop=True)

def det_vec(permno, date_str, dim):
    # deterministic normal vector using a stable hash seed
    h = hashlib.sha256(f"{permno}_{date_str}".encode()).digest()
    seed = int.from_bytes(h[:8], "little") % (2**31 - 1)
    rng = np.random.RandomState(seed)
    return rng.normal(0, 0.5, size=(1, dim)).astype(np.float32)  # [1, dim]

rows = []
for _, r in pairs.iterrows():
    permno = r["PERMNO"]
    d_str  = str(r["DATE"])  # YYYY-MM-DD
    emb    = det_vec(permno, d_str, EMB_DIM)
    # build filename; ensure permno in filename is clean
    try:
        perm_int = int(permno) if float(permno).is_integer() else permno
    except Exception:
        perm_int = permno
    fname = f"{perm_int}_{d_str}.npy"
    fpath = os.path.join(OUT_DIR, fname)
    np.save(fpath, emb)
    rows.append({"PERMNO": permno, "FILING_DATE": d_str, "EMB_PATH": fpath})

index_df = pd.DataFrame(rows).sort_values(["PERMNO", "FILING_DATE"]).reset_index(drop=True)
index_df.to_csv(INDEX_CSV, index=False)
print("Wrote:", INDEX_CSV)
print("Example row:\n", index_df.head(3))