# data_provider/panel_cov_loader_text.py
# -*- coding: utf-8 -*-
"""
PanelCovDataset (temporal-aware):
- Builds sliding windows on the FULL df (per gvkey), then filters windows by an
  OPTIONAL set of allowed anchor dates for the split (train/val/test).
- Anchor date = last horizon date (t + seq_len + pred_len - 1).
- Optional text embedding lookup (same_day | nearest_prev).

Return per item:
  use_text=False:  x_num [T,C] (or [C,T] if channel_first), y [H], meta{gvkey,date}
  use_text=True:   x_num, x_text [D], y, meta
"""

from __future__ import annotations
import os, re, bisect, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------
# Robust date parsing helpers
# ---------------------------
_DATE_RE_8  = re.compile(r"^\d{8}$")    # YYYYMMDD
_DATE_RE_10 = re.compile(r"^\d{10}$")   # epoch s
_DATE_RE_13 = re.compile(r"^\d{13}$")   # epoch ms

def parse_dates_robust(s: pd.Series) -> pd.Series:
    """Return series of 'YYYY-MM-DD' strings; NaT -> NaN."""
    x = s.astype(str).str.strip()
    dt = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")

    m8  = x.str.match(_DATE_RE_8)
    m10 = x.str.match(_DATE_RE_10)
    m13 = x.str.match(_DATE_RE_13)
    mel = ~(m8 | m10 | m13)

    dt.loc[m8]  = pd.to_datetime(x.loc[m8],  format="%Y%m%d", errors="coerce")
    if m10.any():
        sec = pd.to_numeric(x.loc[m10], errors="coerce")
        dt.loc[m10] = pd.to_datetime(sec, unit="s", errors="coerce")
    if m13.any():
        ms = pd.to_numeric(x.loc[m13], errors="coerce")
        dt.loc[m13] = pd.to_datetime(ms, unit="ms", errors="coerce")
    dt.loc[mel]  = pd.to_datetime(x.loc[mel], errors="coerce")

    return dt.dt.strftime("%Y-%m-%d")

# ---------------------------
# Text embedding lookup
# ---------------------------
@dataclass
class _EmbRow:
    date: str
    path: str

class TextEmbeddingLookup:
    def __init__(self, index_csv: str, policy: str = "same_day",
                 text_dim: int = 4096, max_back_days: Optional[int] = None):
        assert policy in ("same_day", "nearest_prev")
        self.policy = policy
        self.text_dim = int(text_dim)
        self.max_back_days = max_back_days

        if not os.path.exists(index_csv):
            raise FileNotFoundError(f"Text index CSV not found: {index_csv}")

        idx = pd.read_csv(index_csv)
        req = ["GVKEY", "FILING_DATE", "EMB_PATH"]
        missing = [c for c in req if c not in idx.columns]
        if missing:
            raise ValueError(f"Index CSV missing columns: {missing}")

        idx["GVKEY"] = idx["GVKEY"].astype(str)
        idx["FILING_DATE"] = parse_dates_robust(idx["FILING_DATE"])
        idx = idx.dropna(subset=["GVKEY", "FILING_DATE", "EMB_PATH"])

        self._by_gv: Dict[str, List[_EmbRow]] = {}
        for gv, sub in idx.groupby("GVKEY"):
            rows = [_EmbRow(d, p) for d, p in zip(sub["FILING_DATE"], sub["EMB_PATH"])]
            rows.sort(key=lambda r: r.date)
            self._by_gv[str(gv)] = rows

        self._cache: Dict[str, np.ndarray] = {}

    def _load_vec(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Embedding at {path} must be 1D, got {arr.shape}")
        if arr.shape[0] != self.text_dim:
            warnings.warn(f"[TextEmbeddingLookup] dim mismatch: expected {self.text_dim}, got {arr.shape[0]}")
        self._cache[path] = arr.astype(np.float32, copy=False)
        return self._cache[path]

    def get(self, gvkey: str, anchor_date: str) -> Optional[np.ndarray]:
        rows = self._by_gv.get(str(gvkey))
        if not rows:
            return None
        dates = [r.date for r in rows]

        if self.policy == "same_day":
            i = bisect.bisect_left(dates, anchor_date)
            if i < len(dates) and dates[i] == anchor_date:
                return self._load_vec(rows[i].path)
            return None

        # nearest_prev
        i = bisect.bisect_right(dates, anchor_date) - 1
        if i < 0:
            return None
        sel = rows[i]
        if self.max_back_days is not None:
            da = pd.to_datetime(anchor_date); ds = pd.to_datetime(sel.date)
            if (da - ds).days > int(self.max_back_days):
                return None
        return self._load_vec(sel.path)

# ---------------------------
# Dataset
# ---------------------------
class PanelCovDataset(Dataset):
    """
    Sliding-window panel dataset with optional text embeddings and meta.
    Windows are created on FULL df; optional `allowed_anchor_dates` filters which
    windows belong to this split (train/val/test) by anchor date.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 id_col: str = "GVKEY",
                 time_col: str = "DATE",
                 y_col: str = "actual",
                 feature_cols: Optional[List[str]] = None,
                 seq_len: int = 36,
                 pred_len: int = 1,
                 channel_first: bool = True,
                 # text
                 use_text: bool = False,
                 text_index_csv: Optional[str] = None,
                 text_dim: int = 4096,
                 text_policy: str = "same_day",
                 text_max_back_days: Optional[int] = None,
                 # safety
                 drop_na_targets: bool = True,
                 # NEW: set of 'YYYY-MM-DD' dates allowed as ANCHOR dates for this split
                 allowed_anchor_dates: Optional[set] = None):
        super().__init__()
        self.id_col = id_col
        self.time_col = time_col
        self.y_col = y_col
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channel_first = bool(channel_first)

        self.use_text = bool(use_text)
        self.text_dim = int(text_dim)
        self._text_lookup: Optional[TextEmbeddingLookup] = None
        if self.use_text:
            if text_index_csv is None:
                raise ValueError("use_text=True but text_index_csv is None")
            self._text_lookup = TextEmbeddingLookup(
                text_index_csv, policy=text_policy, text_dim=text_dim,
                max_back_days=text_max_back_days
            )

        # normalize frame
        req = [id_col, time_col, y_col]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"DataFrame missing required columns: {miss}")

        df = df.copy()
        parsed = parse_dates_robust(df[time_col])
        bad = parsed.isna().sum()
        if bad:
            warnings.warn(f"[PanelCovDataset] dropping {bad} rows with unparseable dates")
        df = df[parsed.notna()].copy()
        df[time_col] = parsed.loc[df.index]

        if feature_cols is None:
            cand = df.select_dtypes(include=["number", "bool"]).columns.tolist()
            for c in [id_col, time_col, y_col]:
                if c in cand: cand.remove(c)
            if not cand:
                raise ValueError("Could not infer numeric feature columns; pass feature_cols.")
            feature_cols = cand
        self.feature_cols = feature_cols

        df[id_col] = df[id_col].astype(str)
        df = df.sort_values([id_col, time_col]).reset_index(drop=True)

        if drop_na_targets:
            n0 = len(df)
            df = df[~pd.isna(df[y_col])]
            if len(df) < n0:
                warnings.warn(f"[PanelCovDataset] dropped {n0-len(df)} rows with NaN targets")

        # store allowed anchor set (strings)
        self._allowed_anchor_dates = None
        if allowed_anchor_dates is not None:
            self._allowed_anchor_dates = set(pd.Series(list(allowed_anchor_dates)).astype(str).str.strip())

        self._groups: List[Dict[str, np.ndarray]] = []
        self._windows: List[Tuple[int, int]] = []  # (group_idx, start_t)
        self._build_groups_and_windows(df)

    def _build_groups_and_windows(self, df: pd.DataFrame):
        W, H = self.seq_len, self.pred_len
        for gk, sub in df.groupby(self.id_col, sort=False):
            X = sub[self.feature_cols].to_numpy(np.float32, copy=True)  # [N,C]
            y = sub[self.y_col].to_numpy(np.float32, copy=True)         # [N]
            dates = sub[self.time_col].astype(str).to_numpy()           # [N]
            ids = sub[self.id_col].astype(str).to_numpy()               # [N]

            N = X.shape[0]
            nwin = N - (W + H) + 1
            if nwin <= 0:
                continue

            g = {"X": X, "y": y, "dates": dates, "ids": ids, "gvkey": str(gk)}
            gidx = len(self._groups)
            self._groups.append(g)

            for t0 in range(nwin):
                anchor_idx = t0 + W + H - 1
                anchor_date = dates[anchor_idx]
                if self._allowed_anchor_dates is not None and anchor_date not in self._allowed_anchor_dates:
                    continue
                self._windows.append((gidx, t0))

        if not self._windows:
            raise ValueError(
                "No windows could be constructed. "
                "Likely cause: split’s allowed anchor dates don’t leave enough history "
                f"for seq_len({W})+pred_len({H}). Consider lowering seq_len/pred_len "
                "or widening the temporal split."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def _slice_item(self, g: Dict[str, np.ndarray], t0: int):
        W, H = self.seq_len, self.pred_len
        x_num = g["X"][t0:t0+W, :]           # [W,C]
        y     = g["y"][t0+W:t0+W+H]          # [H]
        anchor_idx = t0 + W + H - 1
        anchor_date = g["dates"][anchor_idx]
        gvkey = g["ids"][anchor_idx]

        if self.channel_first:
            x_num = np.ascontiguousarray(x_num.T)  # [C,W]
        return x_num, y, gvkey, anchor_date

    def __getitem__(self, idx: int):
        gidx, t0 = self._windows[idx]
        g = self._groups[gidx]
        x_num, y, gvkey, anchor_date = self._slice_item(g, t0)

        x_num = torch.from_numpy(x_num).float()
        y_t = torch.tensor(y[0], dtype=torch.float32) if y.shape[0] == 1 else torch.from_numpy(y).float()
        meta = {"gvkey": str(gvkey), "date": str(anchor_date)}

        if not self.use_text:
            return x_num, y_t, meta

        vec = None
        if self._text_lookup is not None:
            vec = self._text_lookup.get(str(gvkey), str(anchor_date))
        if vec is None:
            vec = np.zeros((self.text_dim,), dtype=np.float32)
        x_text = torch.from_numpy(vec).float()
        return x_num, x_text, y_t, meta