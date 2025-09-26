#!/usr/bin/env python3
import os, argparse, re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from types import SimpleNamespace

# Repo-local: wrapper that exposes .llama_tokenizer and .llama.model
from models.Preprocess_Llama import Model as LlamaEmbedder  # repo-local


# ---------------------------
# Utilities
# ---------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def parse_dates_robust(series: pd.Series) -> pd.Series:
    """
    Parse date strings robustly. Handles:
      - YYYYMMDD (e.g., 20180510)
      - epoch seconds (10 digits)
      - epoch milliseconds (13 digits)
      - ISO-like (YYYY-MM-DD) and other pandas-friendly strings
    Returns: series of 'YYYY-MM-DD' strings; NaT rows become NaN (drop later).
    """
    s = series.astype(str).str.strip()

    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    mask_8  = s.str.fullmatch(r"\d{8}")     # 20180510
    mask_10 = s.str.fullmatch(r"\d{10}")    # epoch seconds
    mask_13 = s.str.fullmatch(r"\d{13}")    # epoch milliseconds
    mask_else = ~(mask_8 | mask_10 | mask_13)

    dt.loc[mask_8]  = pd.to_datetime(s.loc[mask_8],  format="%Y%m%d", errors="coerce")
    # cast to int64 safely; non-numeric to NaT
    if mask_10.any():
        sec = pd.to_numeric(s.loc[mask_10], errors="coerce")
        dt.loc[mask_10] = pd.to_datetime(sec, unit="s", errors="coerce")
    if mask_13.any():
        ms = pd.to_numeric(s.loc[mask_13], errors="coerce")
        dt.loc[mask_13] = pd.to_datetime(ms, unit="ms", errors="coerce")
    dt.loc[mask_else] = pd.to_datetime(s.loc[mask_else], errors="coerce")

    return dt.dt.strftime("%Y-%m-%d")

def pick_device(gpu_arg: str) -> torch.device:
    if gpu_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(gpu_arg)
    return torch.device("cpu")


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/10-Q/nasdow_executive_summaries_10Q_with_gvkey.csv")
    p.add_argument("--out_dir",
                   default="/ssd1/muntasir/Desktop/AutoTimes/dataset/panel/10-Q/embeddings_nasdow")
    p.add_argument("--llm_ckp_dir",
                   default="/ssd1/muntasir/Desktop/AutoTimes/llama-7b")
    p.add_argument("--gpu", default="cuda:0")  # also accepts "cpu"
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=2048)  # reduce if OOM
    p.add_argument("--text_col", default="executive_summary")  # tolerates 'execute_summary' below
    p.add_argument("--id_col", default="gvkey")
    p.add_argument("--date_col", default="date")
    p.add_argument("--pool", choices=["last", "mean"], default="last",
                   help="Pooling over token embeddings: 'last' non-pad token or attention-masked 'mean'.")
    p.add_argument("--dedup_policy", choices=["longest", "first"], default="longest",
                   help="If multiple rows share (id,date), keep the longest text or the first occurrence.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    index_csv = os.path.join(args.out_dir, "gvkey_text_index.csv")

    # --------------- Load CSV ---------------
    df = pd.read_csv(args.csv)

    # tolerate 'execute_summary' typo
    if args.text_col not in df.columns and "execute_summary" in df.columns:
        args.text_col = "execute_summary"

    needed = [args.id_col, args.date_col, args.text_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {args.csv}")

    df = df[needed].copy()

    # --------------- Clean text ---------------
    before = len(df)
    df[args.text_col] = df[args.text_col].map(clean_text)
    df = df[df[args.text_col].str.len() > 0]
    print(f"[clean] kept non-empty `{args.text_col}`: {before} -> {len(df)}")

    # --------------- Robust date parsing ---------------
    parsed_dates = parse_dates_robust(df[args.date_col])
    bad = parsed_dates.isna().sum()
    if bad:
        print(f"[date] could not parse {bad} dates; dropping those rows")
    df = df[parsed_dates.notna()].copy()
    df[args.date_col] = parsed_dates.loc[df.index]

    print("[date] sample parsed (id, date):")
    print(df[[args.id_col, args.date_col]].head(5))

    # --------------- De-dup per (id, date) ---------------
    before = len(df)
    if args.dedup_policy == "longest":
        df["_len"] = df[args.text_col].str.len()
        df = (
            df.sort_values("_len", ascending=False)
              .drop_duplicates(subset=[args.id_col, args.date_col])
              .drop(columns="_len")
        )
    else:  # "first"
        df = df.drop_duplicates(subset=[args.id_col, args.date_col])
    print(f"[dedup] by ({args.id_col}, {args.date_col}): {before} -> {len(df)}")

    # safety: ensure sorted (nice for deterministic batches and index CSV)
    df = df.sort_values([args.id_col, args.date_col]).reset_index(drop=True)

    # --------------- LLaMA init ---------------
    device = pick_device(args.gpu)
    print(f"[device] using {device}")

    cfg = SimpleNamespace(gpu=args.gpu, llm_ckp_dir=args.llm_ckp_dir)
    llama = LlamaEmbedder(cfg)  # exposes .llama_tokenizer and .llama.model
    tokenizer = llama.llama_tokenizer
    backbone = llama.llama.model  # transformer module (HF)

    # place model on device; safe-guard if wrapper hasn’t already moved it
    try:
        backbone.to(device)
    except Exception as e:
        print(f"[warn] could not move model to device ({e}); proceeding")

    backbone.eval()

    # --------------- Batch embed ---------------
    texts = df[args.text_col].tolist()
    ids   = df[args.id_col].astype(str).tolist()
    dates = df[args.date_col].astype(str).tolist()

    B = max(1, args.batch_size)
    rows = []

    # helper for pooling
    def pool_hidden(last_hidden_state: torch.Tensor,
                    attn_mask: torch.Tensor,
                    mode: str) -> torch.Tensor:
        """
        last_hidden_state: [B,T,H], attn_mask: [B,T]
        returns: [B,H]
        """
        if mode == "last":
            last_idx = (attn_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
            b_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            return last_hidden_state[b_idx, last_idx, :]  # [B,H]
        else:  # "mean" (masked)
            # avoid division by zero
            lens = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
            # mask pads to zero, then mean
            masked = last_hidden_state * attn_mask.unsqueeze(-1)
            return masked.sum(dim=1) / lens  # [B,H]

    total_batches = (len(texts) + B - 1) // B
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), B), total=total_batches, desc="Embedding 10-Q summaries"):
            batch_txt = texts[i:i+B]
            enc = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(args.max_length, getattr(tokenizer, "model_max_length", args.max_length))
            )
            # move to device
            for k in enc:
                enc[k] = enc[k].to(device)

            # forward -> last hidden states [B, T, H]
            out = backbone(**enc).last_hidden_state  # dtype may be fp16 on GPU
            pooled = pool_hidden(out, enc["attention_mask"], args.pool)  # [B,H]

            # move to cpu/float32 for saving
            pooled = pooled.detach().float().cpu().numpy()  # [B,H]

            # save each vector and append to index
            for j, vec in enumerate(pooled):
                k = i + j
                gv = ids[k]
                d  = dates[k]
                fname = f"{gv}_{d}.npy"
                fpath = os.path.join(args.out_dir, fname)
                np.save(fpath, vec[None, :].astype(np.float32))  # shape [1, H]

                rows.append({
                    "PERMNO": gv,          # keep for current loader compatibility
                    "GVKEY": gv,           # convenience (future-proofing)
                    "FILING_DATE": d,
                    "EMB_PATH": fpath
                })

    # --------------- Write index CSV ---------------
    index_df = pd.DataFrame(rows).sort_values(["PERMNO", "FILING_DATE"]).reset_index(drop=True)
    index_df.to_csv(index_csv, index=False)
    print(f"\nWrote index: {index_csv}")
    print(index_df.head(5))

    # Sanity: report counts
    uniq_pairs = index_df[["PERMNO", "FILING_DATE"]].drop_duplicates().shape[0]
    print(f"[done] embeddings written: {len(index_df)} (unique (gvkey,date): {uniq_pairs})")


if __name__ == "__main__":
    main()