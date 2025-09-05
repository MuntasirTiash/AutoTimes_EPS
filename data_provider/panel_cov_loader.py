# data_provider/panel_cov_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from bisect import bisect_right


class Dataset_PanelCov(Dataset):
    """
    Long-format panel with (DATE, PERMNO, covariates..., actual).

    Uses 55+ covariates as lagged inputs and predicts ONLY 'actual'.
    Guarantees: lookbacks come ONLY from the same PERMNO.
    - seq_x: [seq_len, C_total], where the LAST column is past 'actual' (ffilled)
    - seq_y: [label_len + pred_len, 1] from the RAW 'actual' (no ffill for targets)
    - *_mark: zeros (keep --mix_embeds off unless you add time-text embeddings)

    Splits by TARGET END date (encoder may borrow history across boundaries):
      • Train: target_end <  2012-01-01
      • Val  : 2012-01-01 ≤ target_end < 2015-01-01
      • Test : target_end ≥ 2015-01-01

    Missing-data policy:
      • Covariates: forward-fill within each PERMNO (no backfill to avoid future leak).
      • 'actual' in encoder (past): forward-fill for inputs only.
      • 'actual' in target window: must be fully observed; windows with NaN target are skipped.
    """
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,                 # [seq_len, label_len, pred_len]
        data_path=None,
        id_col='PERMNO',
        time_col='DATE',
        y_col='actual',
        cov_cols=None,             # list[str] or None -> infer as all non (id,time,y)
        scale=True,
        seasonal_patterns=None,
        drop_short=False,
        split_by='date',
        require_full_x=True,       # skip windows if any NaN remains in seq_x after ffill
        # ---- text fusion options ----
        text_index_csv=None,       # CSV: PERMNO,FILING_DATE,EMB_PATH  (precomputed embeddings)
        text_default_dim=4096,     # embedding dim if no text is found
        text_mode='emb',           # 'emb' (default) or 'ids' (pre-tokenized npz with input_ids/attention_mask)
    ):
        super().__init__()
        assert flag in ['train', 'val', 'test']
        self.seq_len, self.label_len, self.pred_len = size
        self.token_len = self.seq_len - self.label_len
        self.token_num = max(1, self.seq_len // max(1, self.token_len))
        self.flag = flag
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.id_col = id_col
        self.time_col = time_col
        self.y_col = y_col
        self.cov_cols = cov_cols
        self.scale = scale
        self.drop_short = drop_short
        self.split_by = split_by
        self.require_full_x = require_full_x

        self.text_index_csv = text_index_csv
        self.text_default_dim = int(text_default_dim)
        self.text_mode = text_mode

        self._read_and_index()
        self._build_text_index()

    # ------------ TEXT INDEX ------------
    def _build_text_index(self):
        """Build per-PERMNO map to ordered filings and paths."""
        self.text_index = None
        if not self.text_index_csv:
            return
        meta = pd.read_csv(self.text_index_csv)
        # For 'emb' mode, expect: PERMNO, FILING_DATE, EMB_PATH
        # For 'ids' mode, expect: PERMNO, FILING_DATE, NPZ_PATH  (with input_ids/attention_mask)
        date_col = 'FILING_DATE'
        path_col = 'EMB_PATH' if self.text_mode == 'emb' else 'NPZ_PATH'
        if date_col not in meta.columns or path_col not in meta.columns:
            raise ValueError(f"{self.text_index_csv} must have columns {date_col} and {path_col}")

        meta[date_col] = pd.to_datetime(meta[date_col])
        self.text_index = {}
        for pid, sub in meta.groupby('PERMNO'):
            sub = sub.sort_values(date_col)
            self.text_index[pid] = {
                'dates': sub[date_col].to_numpy(),
                'paths': sub[path_col].tolist(),
            }

    def _lookup_text_for(self, permno, asof_date):
        """
        Return most-recent 10-Q payload (<= asof_date).
        emb-mode: float32 array [T_text, D] (often [1, D])
        ids-mode: dict {'input_ids': LongTensor[T], 'attention_mask': LongTensor[T]}
        """
        if self.text_index is None or permno not in self.text_index:
            if self.text_mode == 'emb':
                return np.zeros((1, self.text_default_dim), dtype=np.float32)
            else:
                return {'input_ids': torch.zeros(1, dtype=torch.long),
                        'attention_mask': torch.zeros(1, dtype=torch.long)}
        dates = self.text_index[permno]['dates']
        paths = self.text_index[permno]['paths']
        i = bisect_right(dates, asof_date) - 1
        if i < 0:
            if self.text_mode == 'emb':
                return np.zeros((1, self.text_default_dim), dtype=np.float32)
            else:
                return {'input_ids': torch.zeros(1, dtype=torch.long),
                        'attention_mask': torch.zeros(1, dtype=torch.long)}

        path = paths[i]
        if self.text_mode == 'emb':
            arr = np.asarray(np.load(path), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr  # [T_text, D]
        else:
            # npz with arrays 'input_ids', 'attention_mask'
            npz = np.load(path)
            ids = torch.as_tensor(npz['input_ids'], dtype=torch.long)
            attn = torch.as_tensor(npz['attention_mask'], dtype=torch.long)
            return {'input_ids': ids, 'attention_mask': attn}

    # ------------ CORE ------------
    def _read_and_index(self):
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df[self.time_col] = pd.to_datetime(df[self.time_col])

        # Hard-coded cutoffs
        CUTOFF_TRAIN = pd.Timestamp('2012-01-01')
        CUTOFF_VAL   = pd.Timestamp('2015-01-01')

        base = [self.time_col, self.id_col, self.y_col]
        if self.cov_cols is None:
            self.cov_cols = [c for c in df.columns if c not in set(base)]

        use_cols = [self.time_col, self.id_col] + self.cov_cols + [self.y_col]
        df = df[use_cols].sort_values([self.id_col, self.time_col])

        self.series_X = []      # standardized later
        self.series_y_raw = []  # raw 'actual'
        self.series_dates = []
        self.series_ids = []

        for gid, g in df.groupby(self.id_col, sort=False):
            g = g.sort_values(self.time_col)

            cov = g[self.cov_cols].copy().ffill()
            y_raw = g[self.y_col].copy()
            y_ffill = y_raw.ffill()

            X = np.column_stack([cov.values.astype(np.float32),
                                 y_ffill.values.astype(np.float32)])
            y = y_raw.values.astype(np.float32)
            d = g[self.time_col].values

            if len(g) < (self.seq_len + self.pred_len):
                continue

            self.series_X.append(X)
            self.series_y_raw.append(y)
            self.series_dates.append(d)
            self.series_ids.append(gid)

        # Standardize using ONLY TRAIN dates
        if self.scale and len(self.series_X):
            train_chunks = []
            for X, d in zip(self.series_X, self.series_dates):
                mask_train = d < CUTOFF_TRAIN
                if mask_train.any():
                    train_chunks.append(X[mask_train])
            if not train_chunks:
                raise ValueError("No training rows before 2012-01-01; adjust cutoff or verify data.")
            train_all = np.concatenate(train_chunks, axis=0)

            self.scaler = StandardScaler().fit(train_all)
            zero_scale = (self.scaler.scale_ == 0) | ~np.isfinite(self.scaler.scale_)
            if zero_scale.any():
                self.scaler.scale_[zero_scale] = 1.0

            self.series_X = [ (X - self.scaler.mean_) / self.scaler.scale_ for X in self.series_X ]
            self.y_mean_ = float(self.scaler.mean_[-1])
            self.y_scale_ = float(self.scaler.scale_[-1]) if float(self.scaler.scale_[-1]) != 0 else 1.0
        else:
            self.scaler = None
            self.y_mean_, self.y_scale_ = 0.0, 1.0

        # Build (entity, start) index with TARGET-END based splits
        self.index_pairs = []
        for i, (X, y, d) in enumerate(zip(self.series_X, self.series_y_raw, self.series_dates)):
            n = len(X)
            max_start = n - (self.seq_len + self.pred_len) + 1
            if max_start <= 0:
                continue
            for s in range(max_start):
                s_end   = s + self.seq_len
                r_begin = s_end - self.label_len
                r_end   = s_end + self.pred_len

                tgt_end = d[r_end - 1]
                if self.flag == 'train':
                    if not (tgt_end < CUTOFF_TRAIN):
                        continue
                elif self.flag == 'val':
                    if not (CUTOFF_TRAIN <= tgt_end < CUTOFF_VAL):
                        continue
                else:
                    if not (tgt_end >= CUTOFF_VAL):
                        continue

                tgt_window = y[r_begin:r_end]
                if np.isnan(tgt_window).any():
                    continue

                enc_window = X[s:s_end]
                if not np.isfinite(enc_window).all():
                    continue

                self.index_pairs.append((i, s))

        self.C_total = len(self.cov_cols) + 1

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        ent, s = self.index_pairs[idx]
        s_end   = s + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = s_end + self.pred_len

        X     = self.series_X[ent]                # standardized
        y_raw = self.series_y_raw[ent]           # raw
        d     = self.series_dates[ent]
        seq_x = X[s:s_end]                        # [seq_len, C_total]

        # STANDARDIZE TARGETS to match model output scale
        y_target = y_raw[r_begin:r_end]
        y_target_std = ((y_target - self.y_mean_) / self.y_scale_)[:, None]  # [T,1]

        # simple zeros for marks (you can enhance later)
        x_mark = torch.zeros((self.token_num, 1), dtype=torch.float32)
        y_mark = torch.zeros((self.token_num, 1), dtype=torch.float32)

        # TEXT as of encoder end
        permno = self.series_ids[ent]
        enc_end_date = d[s_end - 1]
        text_payload = self._lookup_text_for(permno, enc_end_date)

        # sanitize numeric arrays
        seq_x = np.nan_to_num(seq_x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        y_target_std = np.nan_to_num(y_target_std, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        if self.text_mode == 'emb':
            text_payload = np.nan_to_num(text_payload, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
            return seq_x, y_target_std, x_mark, y_mark, text_payload
        else:
            # dict of LongTensors
            return seq_x, y_target_std, x_mark, y_mark, text_payload

    # Helper to invert scaling of the last channel
    def inverse_transform_y(self, arr_last_channel):
        if self.scaler is None:
            return arr_last_channel
        return arr_last_channel * self.y_scale_ + self.y_mean_