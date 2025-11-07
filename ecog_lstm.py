# -*- coding: utf-8 -*-
"""
Ecog-LSTM toolkit — consolidated module for import.

Includes:
  • Data utilities: set_seed, infer_time_hours, choose_feature_columns
  • Config dataclass
  • Sequence builder: preprocess_and_build_sequences
  • Model: LSTMBinary (BiLSTM + attention)
  • Training/eval: _fit_one_fold, run_time_increasing_loo, run_time_increasing_kfold
  • Class imbalance helpers: balance_by_duplication
  • Label distribution plots: plot_hourly_epilepsy_distribution
  • Saliency: Grad×Input (batched), run_saliency_only, topk_table_per_block, heatmap utils

Usage example:
    from ecog_lstm import Config, run_time_increasing_kfold, run_saliency_only
    cfg = Config(path_csv="Data/all_patients_60min_aggregated.csv", start_hr=12, end_hr=120,
                 step_minutes=60, features=FEATURE_LIST, eval_stride_hr=5.0,
                 epochs=60, batch_size=64, hidden_size=64, num_layers=2,
                 bidirectional=True, dropout=0.3)
    df_k = run_time_increasing_kfold(cfg, n_splits=5, repeats=1, plot_curves=False)
    sal = run_saliency_only(cfg, blocks=list(range(12,121,12)), train_epochs=20)

Notes:
  - Saliency uses batched Grad×Input to avoid OOM.
  - KFold/LOO scale per-fold using MinMaxScaler fitted on train only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import sys

# Public API
__all__ = [
    "set_seed","infer_time_hours","choose_feature_columns","Config",
    "preprocess_and_build_sequences","LSTMBinary","balance_by_duplication",
    "run_time_increasing_loo","run_time_increasing_kfold",
    "plot_hourly_epilepsy_distribution",
    "grad_x_input_saliency_batched","run_saliency_only","topk_table_per_block",
    "saliency_block_matrix_from_out","plot_saliency_heatmap_from_out"
]

# ----------------------- Constants -----------------------
NON_FEATURE_CANDIDATES = [
    'study_id','h5_id','h5_folder_id','channel',
    'epilepsy_label','has_epilepsy',
    'tbi_time_10min_start_hr','bin_start_hr','bin_end_hr','time_hr','time_hours',
    'bin_start_min','bin_end_min','time_min','time_minutes','time',
    'n_non_na','n_total','nan_skipped'
]

# ----------------------- Utils ---------------------------
def set_seed(s: int = 42) -> None:
    """Set Python/NumPy/PyTorch RNG seeds (CPU/GPU).
    """
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def infer_time_hours(df: pd.DataFrame) -> pd.Series:
    """Infer time in hours from available columns.
    Priority:
      1) hour columns: tbi_time_10min_start_hr, bin_end_hr, bin_start_hr, time_hr, time_hours
      2) minute columns: bin_start_min, bin_end_min, time_min, time_minutes (converted to hours)
      3) generic 'time' (interpreted as hours)
    """
    for c in ['tbi_time_10min_start_hr','bin_end_hr','bin_start_hr','time_hr','time_hours']:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                return s
    for c in ['bin_start_min','bin_end_min','time_min','time_minutes']:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                return s / 60.0
    if 'time' in df.columns:
        s = pd.to_numeric(df['time'], errors='coerce')
        if s.notna().any():
            return s
    raise ValueError("Failed to infer time column.")


def choose_feature_columns(df: pd.DataFrame, explicit_features: List[str]) -> List[str]:
    """Intersect explicit feature list with actual columns; drop admin/time/label and all-NaN.
    Prints a warning if some explicit features are missing.
    """
    exist = [c for c in explicit_features if c in df.columns]
    missing = [c for c in explicit_features if c not in df.columns]
    if missing:
        print(f"[Warn] {len(missing)} explicit features are missing (first 12): {missing[:12]}{' ...' if len(missing)>12 else ''}")
    non_feat_present = set([c for c in NON_FEATURE_CANDIDATES if c in df.columns])
    feat_cols = [c for c in exist if c not in non_feat_present and not df[c].isna().all()]
    if len(feat_cols) == 0:
        raise ValueError("Empty intersection between explicit features and data columns.")
    return feat_cols

# ----------------------- Config --------------------------
@dataclass
class Config:
    path_csv: str
    start_hr: float = 0.0
    end_hr: float = 72.0
    step_minutes: int = 60
    features: Optional[List[str]] = None
    label_col: str = 'epilepsy_label'
    agg_label: str = 'any_true'  # or 'majority'
    min_nonzero_frac: float = 0.10
    restrict_to_intersection: bool = True
    min_patient_coverage: float = 0.95
    # imbalance
    use_duplication: bool = False
    use_pos_weight: bool = True
    # training
    epochs: int = 80
    batch_size: int = 64
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.3
    out_csv: str = "has_epilepsy_lstm.csv"
    # evaluation stride (in hours)
    eval_stride_hr: float = 1.0

# ---------------- Sequence builder -----------------------
def preprocess_and_build_sequences(cfg: Config):
    """Load CSV, infer time (hours), filter, select/clean features,
    aggregate patient-level labels, and assemble (N,T,F) tensors.
    Returns (X, y, feat_cols, pids, T)
    """
    df = pd.read_csv(cfg.path_csv)
    if 'study_id' not in df.columns:
        raise ValueError("Missing 'study_id' column.")
    if cfg.label_col not in df.columns:
        raise ValueError(f"Missing label column '{cfg.label_col}'.")

    # Time → hours
    time_hr = infer_time_hours(df)
    df = df.assign(_time_hr=pd.to_numeric(time_hr, errors='coerce'))
    df = df.dropna(subset=['study_id','_time_hr',cfg.label_col]).copy()
    df = df[(df['_time_hr']>=cfg.start_hr)&(df['_time_hr']<=cfg.end_hr)].copy()
    if df.empty:
        raise ValueError("No data within the selected time window.")

    # Features
    feat_orig = choose_feature_columns(df, cfg.features or [])
    print(f"[Info] Explicit features requested: {len(cfg.features or [])}, present: {len(feat_orig)}")

    feat_cols = feat_orig
    if cfg.restrict_to_intersection:
        coverage_per_pid = df.groupby('study_id')[feat_cols].apply(lambda g: g.notna().any(axis=0))
        feat_cov = coverage_per_pid.mean(axis=0)
        kept = feat_cov[feat_cov >= cfg.min_patient_coverage].index.tolist()
        if len(kept) == 0:
            top_k = min(50, len(feat_cols))
            kept = feat_cov.sort_values(ascending=False).index[:top_k].tolist()
            print(f"[Warn] Strict intersection empty → fallback to top-{top_k} by coverage.")
        else:
            print(f"[Info] Kept after coverage intersection: {len(kept)} / {len(feat_cols)} (threshold={cfg.min_patient_coverage:.2f})")
        feat_cols = kept

    print(f"[Info] Final feature dimension: {len(feat_cols)}")

    # Numeric cleaning
    df[feat_cols] = df[feat_cols].replace([np.inf,-np.inf], np.nan)
    df[feat_cols] = df[feat_cols].astype(float).fillna(df[feat_cols].mean(numeric_only=True))
    df[feat_cols] = df[feat_cols].apply(lambda c: np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))

    # Patient-level label aggregation
    if cfg.agg_label=='any_true':
        lab_agg = df.groupby('study_id')[cfg.label_col].apply(lambda s: (pd.to_numeric(s, errors='coerce')>0).any()).astype(int)
    elif cfg.agg_label=='majority':
        lab_agg = df.groupby('study_id')[cfg.label_col].apply(lambda s: (pd.to_numeric(s, errors='coerce')>0).mean()).round().astype(int)
    else:
        raise ValueError("agg_label must be 'any_true' or 'majority'.")

    y_overall = lab_agg.values
    n_pos = int((y_overall==1).sum()); n_neg = int((y_overall==0).sum())
    print(f"[Info] Patient distribution: positive={n_pos}, negative={n_neg}, total={n_pos+n_neg}")

    # Build time grid
    step = cfg.step_minutes
    T = int(round((cfg.end_hr - cfg.start_hr)*60/step)) + 1  # inclusive
    def hour_to_idx(h): return int(round((h - cfg.start_hr)*60/step))

    X_list, y_list, pids = [], [], []
    F = len(feat_cols)
    for pid, g in df.groupby('study_id'):
        seq = np.zeros((T,F), dtype=np.float32)
        for _,row in g.iterrows():
            i = hour_to_idx(float(row['_time_hr']))
            if 0<=i<T:
                seq[i] = row[feat_cols].values.astype(np.float32)
        if (seq!=0).any():
            X_list.append(seq); y_list.append(int(lab_agg.get(pid,0))); pids.append(pid)

    if not X_list:
        raise ValueError("No usable samples (all-zero sequences).")

    X = np.stack(X_list,0)
    y = np.array(y_list, dtype=np.int64)
    print(f"[Info] Tensor shapes: N={len(pids)}, T={T} (step={step}min), F={F}")
    return X, y, feat_cols, pids, T

# ----------------------- Model ---------------------------
class LSTMBinary(nn.Module):
    """BiLSTM + attention + binary logit head."""
    def __init__(self, input_dim, hidden_size=64, num_layers=1, bidirectional=True, dropout=0.2):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn_proj = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
        self.attn_score = nn.Linear(hidden_size * self.num_directions, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        H, _ = self.lstm(x)                 # [B, T, D*H]
        H = self.dropout(H)
        u = torch.tanh(self.attn_proj(H))   # [B, T, D*H]
        scores = self.attn_score(u).squeeze(-1)       # [B, T]
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        z = torch.sum(alpha * H, dim=1)     # [B, D*H]
        return self.fc(z).view(-1)          # logits [B]

# ---------------- Imbalance helper -----------------------
def balance_by_duplication(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Simple duplication resampling to balance classes within the training fold."""
    classes = np.unique(y); max_n = max((y==c).sum() for c in classes)
    X_out, y_out = [], []
    for c in classes:
        Xc = X[y==c]; need = max_n - len(Xc)
        if need>0:
            X_aug = resample(Xc, replace=True, n_samples=need, random_state=42)
            Xc = np.concatenate([Xc, X_aug],0)
        X_out.append(Xc); y_out.append(np.full(len(Xc), c, dtype=np.int64))
    return np.concatenate(X_out,0), np.concatenate(y_out,0)

# ----------------- Train one fold ------------------------
def _fit_one_fold(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                  cfg: Config, L: int, F: int, device) -> np.ndarray:
    """Fit one model on (Xtr, ytr) and return predicted probabilities for Xte."""
    # scale on train only
    scaler = MinMaxScaler().fit(Xtr.reshape(-1, F))
    Xtr = scaler.transform(Xtr.reshape(-1, F)).reshape(Xtr.shape[0], L, F)
    Xte = scaler.transform(Xte.reshape(-1, F)).reshape(Xte.shape[0], L, F)

    # optional duplication
    if cfg.use_duplication:
        Xtr_bal, ytr_bal = balance_by_duplication(Xtr, ytr)
    else:
        Xtr_bal, ytr_bal = Xtr, ytr

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr_bal, dtype=torch.float32),
                      torch.tensor(ytr_bal, dtype=torch.float32)),
        batch_size=cfg.batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )

    model = LSTMBinary(F, cfg.hidden_size, cfg.num_layers, cfg.bidirectional, cfg.dropout).to(device)

    if cfg.use_pos_weight:
        pos = float((ytr_bal==1).sum()); neg = float((ytr_bal==0).sum())
        pw = max(1.0, neg/max(1.0, pos))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32).to(device)).view(-1)).cpu().numpy()
    return prob

# ------------- Eval schedule utilities ------------------
def _compute_eval_Ls(T: int, step_minutes: int, eval_stride_hr: float) -> List[int]:
    stride_steps = max(1, int(round(eval_stride_hr * 60.0 / step_minutes)))
    Ls = list(range(stride_steps, T + 1, stride_steps))
    if len(Ls) == 0 or Ls[-1] != T:
        Ls.append(T)
    return Ls

# --------------------- LOO eval --------------------------
def run_time_increasing_loo(cfg: Config) -> pd.DataFrame:
    """Time-increasing Leave-One-Out evaluation; returns metrics DataFrame."""
    X_np, y_np, feat_cols, pids, T = preprocess_and_build_sequences(cfg)
    N, F = X_np.shape[0], X_np.shape[2]
    if N < 2: raise ValueError("LOO requires at least 2 patients.")

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info][LOO] Device: {device}, N={N}, T={T}, F={F}, step={cfg.step_minutes}min, stride={cfg.eval_stride_hr}hr")

    results = []
    eval_Ls = _compute_eval_Ls(T, cfg.step_minutes, cfg.eval_stride_hr)

    for L in tqdm(eval_Ls, desc="Time steps (LOO)", dynamic_ncols=True, leave=False, file=sys.stdout):
        Xw = X_np[:, :L, :].copy(); yw = y_np.copy()
        # usable filter
        nonzero = (Xw!=0).any(axis=2).sum(axis=1)
        min_steps = max(1, int(cfg.min_nonzero_frac * L))
        mask = nonzero >= min_steps
        Xw, yw = Xw[mask], yw[mask]
        if len(np.unique(yw)) < 2 or len(yw) < 2:
            results.append({'time_hr': cfg.start_hr + (L-1)*(cfg.step_minutes/60.0),
                            'AUC': np.nan, 'Accuracy': np.nan,
                            'Acc_class_0': np.nan, 'Acc_class_1': np.nan})
            continue

        all_probs, all_preds, all_true = [], [], []
        loo = LeaveOneOut()
        for tr, te in loo.split(Xw):
            Xtr, Xte = Xw[tr].copy(), Xw[te].copy()
            ytr, yte = yw[tr], yw[te]

            if len(np.unique(ytr)) < 2:
                maj = int(np.bincount(ytr).argmax())
                prob = float(maj)
            else:
                prob = _fit_one_fold(Xtr, ytr, Xte, cfg, L=L, F=F, device=device)[0]

            pred = int(prob >= 0.5)
            all_true.append(int(yte[0])); all_probs.append(float(prob)); all_preds.append(pred)

        y_true = np.array(all_true, int); y_pred = np.array(all_preds, int); y_prob = np.array(all_probs, float)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        acc0 = (cm[0,0]/cm[0].sum()) if cm[0].sum()>0 else np.nan
        acc1 = (cm[1,1]/cm[1].sum()) if cm[1].sum()>0 else np.nan

        results.append({'time_hr': cfg.start_hr + (L-1)*(cfg.step_minutes/60.0),
                        'AUC': auc, 'Accuracy': acc,
                        'Acc_class_0': acc0, 'Acc_class_1': acc1})

    dfm = pd.DataFrame(results)
    out_csv = cfg.out_csv.replace(".csv", "_loo.csv")
    dfm.to_csv(out_csv, index=False)
    print(f"[OK] LOO metrics saved: {out_csv}")
    return dfm

# -------------------- KFold eval -------------------------
def run_time_increasing_kfold(cfg: Config,
                              n_splits: int = 5,
                              repeats: int = 1,
                              csv_suffix: str = "_5fold.csv",
                              plot_curves: bool = True) -> pd.DataFrame:
    """Time-increasing stratified K-fold evaluation (optionally with repeats)."""
    import matplotlib.pyplot as plt

    X_np, y_np, feat_cols, pids, T = preprocess_and_build_sequences(cfg)
    N, F = X_np.shape[0], X_np.shape[2]
    if N < n_splits: raise ValueError(f"N={N} too small for {n_splits}-fold.")

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info][KFold] Device: {device}, N={N}, T={T}, F={F}, step={cfg.step_minutes}min, stride={cfg.eval_stride_hr}hr, folds={n_splits}, repeats={repeats}")

    time_axis = [cfg.start_hr + i*(cfg.step_minutes/60.0) for i in range(T)]
    records = []

    eval_Ls = _compute_eval_Ls(T, cfg.step_minutes, cfg.eval_stride_hr)

    for L in tqdm(eval_Ls, desc="Time steps (KFold)", dynamic_ncols=True, leave=False, file=sys.stdout):
        Xw = X_np[:, :L, :].copy(); yw = y_np.copy()
        nonzero = (Xw!=0).any(axis=2).sum(axis=1)
        min_steps = max(1, int(cfg.min_nonzero_frac * L))
        mask = nonzero >= min_steps
        Xw, yw = Xw[mask], yw[mask]
        if len(np.unique(yw)) < 2 or len(yw) < n_splits:
            records.append({'time_hr': time_axis[L-1],
                            'AUC_mean': np.nan, 'AUC_std': np.nan,
                            'ACC_mean': np.nan, 'ACC_std': np.nan,
                            'ACC0_mean': np.nan, 'ACC0_std': np.nan,
                            'ACC1_mean': np.nan, 'ACC1_std': np.nan})
            continue

        auc_list, acc_list, acc0_list, acc1_list = [], [], [], []
        for r in range(repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+r)
            for tr_idx, te_idx in skf.split(Xw, yw):
                Xtr, Xte = Xw[tr_idx].copy(), Xw[te_idx].copy()
                ytr, yte = yw[tr_idx], yw[te_idx]

                if len(np.unique(ytr)) < 2:
                    probs = np.full(len(yte), float(int(np.bincount(ytr).argmax())))
                else:
                    probs = _fit_one_fold(Xtr, ytr, Xte, cfg, L=L, F=F, device=device)

                preds = (probs >= 0.5).astype(int)
                auc = roc_auc_score(yte, probs) if len(np.unique(yte))>1 else np.nan
                acc = accuracy_score(yte, preds)
                cm = confusion_matrix(yte, preds, labels=[0,1])
                acc0 = (cm[0,0]/cm[0].sum()) if cm[0].sum()>0 else np.nan
                acc1 = (cm[1,1]/cm[1].sum()) if cm[1].sum()>0 else np.nan

                auc_list.append(auc); acc_list.append(acc); acc0_list.append(acc0); acc1_list.append(acc1)

        def mean_std(arr):
            arr = np.array(arr, dtype=float)
            m = np.nanmean(arr)
            s = np.nanstd(arr, ddof=1) if np.isfinite(arr).sum()>1 else np.nan
            return m, s

        auc_m, auc_s = mean_std(auc_list)
        acc_m, acc_s = mean_std(acc_list)
        acc0_m, acc0_s = mean_std(acc0_list)
        acc1_m, acc1_s = mean_std(acc1_list)

        records.append({
            'time_hr': time_axis[L-1],
            'AUC_mean': auc_m,  'AUC_std': auc_s,
            'ACC_mean': acc_m,  'ACC_std': acc_s,
            'ACC0_mean': acc0_m,'ACC0_std': acc0_s,
            'ACC1_mean': acc1_m,'ACC1_std': acc1_s
        })

    dfm = pd.DataFrame(records)
    out_csv = cfg.out_csv.replace(".csv", csv_suffix)
    dfm.to_csv(out_csv, index=False)
    print(f"[OK] K-fold metrics saved: {out_csv}")

    if plot_curves:
        plt.figure(figsize=(9,5))
        for key, label in [('AUC', 'AUC'), ('ACC','Accuracy'),
                           ('ACC0','Acc class 0'), ('ACC1','Acc class 1')]:
            m = dfm[f'{key}_mean'].values
            s = dfm[f'{key}_std'].values
            x = dfm['time_hr'].values
            plt.plot(x, m, label=label)
            if np.isfinite(s).any():
                plt.fill_between(x, m - s, m + s, alpha=0.15)
        plt.xlabel('Hours since start'); plt.ylabel('Score')
        plt.title(f"Stratified {n_splits}-fold (repeats={repeats}) — step={cfg.step_minutes}min, stride={cfg.eval_stride_hr}hr")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return dfm

# ---------------- Label distribution --------------------
# Helpers reused from earlier logic

def _load_windowed_df_and_feats(cfg: Config):
    df = pd.read_csv(cfg.path_csv)
    if 'study_id' not in df.columns:
        raise ValueError("Missing 'study_id'.")
    if cfg.label_col not in df.columns:
        raise ValueError(f"Missing label column '{cfg.label_col}'.")

    time_hr = infer_time_hours(df)
    df['_time_hr'] = pd.to_numeric(time_hr, errors='coerce')
    df = df.dropna(subset=['study_id', '_time_hr', cfg.label_col]).copy()
    df = df[(df['_time_hr'] >= cfg.start_hr) & (df['_time_hr'] <= cfg.end_hr)].copy()
    if df.empty:
        raise ValueError("No data in selected window.")

    feat_cols = choose_feature_columns(df, cfg.features or [])
    if cfg.restrict_to_intersection:
        coverage_per_pid = df.groupby('study_id')[feat_cols].apply(lambda g: g.notna().any(axis=0))
        feat_cov = coverage_per_pid.mean(axis=0)
        kept = feat_cov[feat_cov >= cfg.min_patient_coverage].index.tolist()
        if len(kept) == 0:
            top_k = min(50, len(feat_cols))
            kept = feat_cov.sort_values(ascending=False).index[:top_k].tolist()
        feat_cols = kept

    if cfg.agg_label == 'any_true':
        lab_agg = df.groupby('study_id')[cfg.label_col].apply(lambda s: (pd.to_numeric(s, errors='coerce') > 0).any()).astype(int)
    elif cfg.agg_label == 'majority':
        lab_agg = df.groupby('study_id')[cfg.label_col].apply(lambda s: (pd.to_numeric(s, errors='coerce') > 0).mean()).round().astype(int)
    else:
        raise ValueError("agg_label must be 'any_true' or 'majority'.")

    step = cfg.step_minutes
    T = int(round((cfg.end_hr - cfg.start_hr) * 60 / step)) + 1
    time_axis = [cfg.start_hr + i * (step / 60.0) for i in range(T)]
    def hour_to_idx(h): return int(round((h - cfg.start_hr) * 60 / step))

    return df, feat_cols, lab_agg, T, time_axis, hour_to_idx


def _build_observed_mask(df: pd.DataFrame, feat_cols: List[str], T: int, hour_to_idx) -> Tuple[np.ndarray, List[str]]:
    pids = sorted(df['study_id'].unique())
    pid2idx = {p: i for i, p in enumerate(pids)}
    obs = np.zeros((len(pids), T), dtype=bool)
    for pid, g in df.groupby('study_id'):
        i = pid2idx[pid]
        present_row = g[feat_cols].notna().any(axis=1).values
        hours = g['_time_hr'].astype(float).values
        idxs = np.array([hour_to_idx(h) for h in hours], dtype=int)
        ok = (idxs >= 0) & (idxs < T)
        obs[i, idxs[ok]] |= present_row[ok]
    return obs, pids


def plot_hourly_epilepsy_distribution(cfg: Config,
                                      mode: str = "usable_cumulative",
                                      normalize: bool = False,
                                      save_prefix: str = "hourly_epilepsy_dist") -> pd.DataFrame:
    import matplotlib.pyplot as plt

    X_np, y_np, feat_cols_used, pids_used, T = preprocess_and_build_sequences(cfg)
    time_axis = [cfg.start_hr + i*(cfg.step_minutes/60.0) for i in range(T)]
    rows = []

    if mode == "usable_cumulative":
        nonzero_cum = (X_np != 0).any(axis=2).cumsum(axis=1)  # [N, T]
        for L in range(1, T+1):
            min_steps = max(1, int(cfg.min_nonzero_frac * L))
            mask = nonzero_cum[:, L-1] >= min_steps
            yL = y_np[mask]
            pos = int((yL == 1).sum()); neg = int((yL == 0).sum())
            tot = int(len(yL))
            rows.append({
                "time_hr": time_axis[L-1],
                "usable_total": tot,
                "label_0_count": neg,
                "label_1_count": pos,
                "label_0_frac": (neg/tot) if tot>0 else np.nan,
                "label_1_frac": (pos/tot) if tot>0 else np.nan,
            })
        out_csv = f"{save_prefix}_{mode}_{cfg.step_minutes}min_{int(cfg.start_hr)}_{int(cfg.end_hr)}.csv"

    elif mode == "valid_exact":
        df_win, feat_cols_raw, lab_agg, T2, time_axis2, hour_to_idx = _load_windowed_df_and_feats(cfg)
        assert T2 == T and np.allclose(time_axis2, time_axis), "Time grid mismatch."
        obs_mask, pids = _build_observed_mask(df_win, feat_cols_raw, T, hour_to_idx)
        labs = np.array([int(lab_agg.get(pid, 0)) for pid in pids], dtype=np.int64)

        for t in range(T):
            mask = obs_mask[:, t]
            y_t = labs[mask]
            pos = int((y_t == 1).sum()); neg = int((y_t == 0).sum())
            tot = int(len(y_t))
            rows.append({
                "time_hr": time_axis[t],
                "valid_total": tot,
                "label_0_count": neg,
                "label_1_count": pos,
                "label_0_frac": (neg/tot) if tot>0 else np.nan,
                "label_1_frac": (pos/tot) if tot>0 else np.nan,
            })
        out_csv = f"{save_prefix}_{mode}_{cfg.step_minutes}min_{int(cfg.start_hr)}_{int(cfg.end_hr)}.csv"

    else:
        raise ValueError("mode must be 'usable_cumulative' or 'valid_exact'.")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved hourly label distribution: {out_csv}")

    x = df["time_hr"].values
    if normalize:
        y0 = df["label_0_frac"].values
        y1 = df["label_1_frac"].values
        y_label = "Proportion"; title_mode = "proportions"
    else:
        y0 = df["label_0_count"].values
        y1 = df["label_1_count"].values
        y_label = "Count"; title_mode = "counts"

    plt.figure(figsize=(10, 4.5))
    plt.stackplot(x, y0, y1, labels=["Label 0 (no epilepsy)", "Label 1 (epilepsy)"])
    plt.xlabel("Hours since start"); plt.ylabel(y_label)
    plt.title(f"{mode.replace('_',' ')} — {title_mode} by label (step={cfg.step_minutes}min)")
    plt.grid(True, alpha=0.3); plt.legend(loc="upper left")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, df["label_0_count"].values, label="Label 0 (count)")
    plt.plot(x, df["label_1_count"].values, label="Label 1 (count)")
    total_col = "usable_total" if mode=="usable_cumulative" else "valid_total"
    if total_col in df.columns:
        plt.plot(x, df[total_col].values, linestyle="--", label="Total")
    plt.xlabel("Hours since start"); plt.ylabel("Count")
    plt.title(f"{mode.replace('_',' ')} — counts (step={cfg.step_minutes}min)")
    plt.grid(True, alpha=0.3); plt.legend();
    plt.tight_layout(); plt.show()

    return df

# ----------------- Saliency (Grad×Input) -----------------

def grad_x_input_saliency_batched(model: nn.Module,
                                  X: np.ndarray,
                                  device: torch.device,
                                  abs_value: bool = True,
                                  batch_size: int = 32) -> np.ndarray:
    """Batched Grad×Input per-sample saliency: returns (N, T, F)."""
    was_training = model.training
    saved_dropout = getattr(model, 'dropout', None)
    if saved_dropout is not None:
        try:
            model.dropout = nn.Identity()
        except Exception:
            pass
    model.train()

    N, T, F = X.shape
    out = np.empty((N, T, F), dtype=np.float32)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        xb = torch.tensor(X[i:j], dtype=torch.float32, device=device, requires_grad=True)
        logits = model(xb).view(-1)
        loss = logits.sum()
        grads = torch.autograd.grad(loss, xb, retain_graph=False, create_graph=False)[0]
        sal = grads * xb
        if abs_value:
            sal = sal.abs()
        out[i:j] = sal.detach().cpu().numpy()

    if saved_dropout is not None:
        try:
            model.dropout = saved_dropout
        except Exception:
            pass
    if not was_training:
        model.eval()
    return out


def make_blocks_cumulative(start_hr: float, end_hr: float,
                           block_hours: List[int]) -> List[Tuple[int,int,int]]:
    """Cumulative windows (start_hr, be]; return list of tuples (be, t0, t1)."""
    out = []
    for be in block_hours:
        if be <= start_hr or be > end_hr:
            continue
        t0 = 0
        t1 = int(round(be - start_hr))
        out.append((be, t0, t1))
    return out


def aggregate_saliency_per_block(saliency_ntf: np.ndarray,
                                 blocks: List[Tuple[int,int,int]],
                                 feat_names: List[str],
                                 agg_over_time: str = 'mean',
                                 agg_over_samples: str = 'mean') -> pd.DataFrame:
    """Aggregate (N,T,F) saliency → per-block (F,) importance.
    Returns long DF: [block_end_hr, feature_index, feature, importance, norm_importance]
    """
    N, T, F = saliency_ntf.shape
    rows = []
    for be, t0, t1 in blocks:
        s = saliency_ntf[:, t0:(t1+1), :]  # (N, L, F)
        s_time = s.sum(axis=1) if agg_over_time == 'sum' else s.mean(axis=1)  # (N,F)
        vec = np.median(s_time, axis=0) if agg_over_samples == 'median' else np.mean(s_time, axis=0)  # (F,)
        for j, v in enumerate(vec):
            rows.append({'block_end_hr': be,
                         'feature_index': j,
                         'feature': feat_names[j] if j < len(feat_names) else f'f{j}',
                         'importance': float(v)})
    df = pd.DataFrame(rows)
    df['norm_importance'] = df.groupby('block_end_hr')['importance'].transform(lambda v: v / (v.sum() + 1e-8))
    return df


def run_saliency_only(cfg: Config,
                      blocks: List[int] = list(range(12, 121, 12)),
                      saliency_abs: bool = True,
                      train_epochs: Optional[int] = None,
                      batch_size: Optional[int] = None,
                      model: Optional[nn.Module] = None,
                      scaler: Optional[MinMaxScaler] = None,
                      return_trained_model: bool = True) -> Dict[str, object]:
    """Run Grad×Input (cumulative) per block; returns dict with saliency and artifacts."""
    X, y, feat_cols, pids, T = preprocess_and_build_sequences(cfg)  # X: (N,T,F)
    N, F = X.shape[0], X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # scaling (global by default; pass in fold scaler if needed)
    if scaler is None:
        scaler = MinMaxScaler().fit(X.reshape(-1, F))
    Xs = scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)

    # model
    if model is None:
        model = LSTMBinary(input_dim=F,
                           hidden_size=getattr(cfg, 'hidden_size', 32),
                           num_layers=getattr(cfg, 'num_layers', 1),
                           bidirectional=getattr(cfg, 'bidirectional', True),
                           dropout=getattr(cfg, 'dropout', 0.2)).to(device)
        epochs = int(getattr(cfg, 'epochs', 40) if train_epochs is None else train_epochs)
        if epochs > 0:
            ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                               torch.tensor(y, dtype=torch.float32))
            dl = DataLoader(
                ds,
                batch_size=(getattr(cfg, 'batch_size', 64) if batch_size is None else batch_size),
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=0
            )
            criterion = nn.BCEWithLogitsLoss()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb).view(-1)
                    loss = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
        else:
            warnings.warn("epochs=0 → skipping model training; using randomly initialized weights.")

    # saliency (batched)
    sal_ntf = grad_x_input_saliency_batched(
        model, Xs, device=device, abs_value=saliency_abs,
        batch_size=(batch_size or getattr(cfg, 'batch_size', 32))
    )

    # blocks & aggregation
    blocks_cum = make_blocks_cumulative(cfg.start_hr, cfg.end_hr, blocks)
    if len(blocks_cum) == 0:
        raise ValueError("No valid saliency blocks produced. Check start/end hours and 'blocks'.")

    sal_df = aggregate_saliency_per_block(
        saliency_ntf=sal_ntf,
        blocks=blocks_cum,
        feat_names=list(feat_cols),
        agg_over_time='mean',
        agg_over_samples='mean'
    )

    out: Dict[str, object] = {
        'saliency_df': sal_df,
        'feature_names': list(feat_cols),
        'blocks_saliency': blocks_cum,
        'saliency_ntf': sal_ntf,
        'scaler': scaler
    }
    if return_trained_model:
        out['model'] = model
    return out

# ---------------- Convenience tables ---------------------
def topk_table_per_block(df: pd.DataFrame, value_col: str, k: int = 10) -> pd.DataFrame:
    out = []
    for be, g in df.groupby('block_end_hr'):
        g = g.sort_values(value_col, ascending=False).head(k)
        g = g[['block_end_hr', 'feature', value_col]].reset_index(drop=True)
        out.append(g)
    return pd.concat(out, axis=0, ignore_index=True) if out else pd.DataFrame()

# ---------------- Heatmap utilities ----------------------

def saliency_block_matrix_from_out(out: dict,
                                   block_end_hr: int,
                                   sample_reduce: str = "mean",
                                   cfg=None  
                                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract LSTM Grad×Input saliency matrix for a cumulative window (start_hr, block_end_hr]:
    Returns (S_tf, time_hr, feat_names)
    """
    if "saliency_ntf" not in out:
        raise ValueError("Missing 'saliency_ntf' in output.")
    sal_ntf = out["saliency_ntf"]
    blocks_sal = out["blocks_saliency"]
    feat_names = out["feature_names"]

    hit = [b for b in blocks_sal if b[0] == block_end_hr]
    if not hit:
        raise ValueError(f"No cumulative block for be={block_end_hr}.")
    _, t0, t1 = hit[0]

    Sl = sal_ntf[:, t0:(t1+1), :]
    S_tf = np.mean(Sl, axis=0) if sample_reduce == "mean" else np.median(Sl, axis=0)

    if cfg is not None:
        time_hr = [cfg.start_hr + (i * cfg.step_minutes / 60.0) for i in range(S_tf.shape[0])]
    else:
        time_hr = np.arange(S_tf.shape[0])

    return S_tf, np.array(time_hr), list(feat_names)


def plot_saliency_heatmap_from_out(out: dict,
                                   block_end_hr: int,
                                   sample_reduce: str = "mean",
                                   clip_percentile: float = 99.0,
                                   figsize: tuple = (12, 6),
                                   save_path: Optional[str] = None,
                                   show: bool = True,
                                   title: Optional[str] = None,
                                   cfg=None):  # 新增参数
    import matplotlib.pyplot as plt
    S_tf, t_hr, feat_names = saliency_block_matrix_from_out(out, block_end_hr,
                                                            sample_reduce=sample_reduce,
                                                            cfg=cfg)

    vmin = vmax = None
    if clip_percentile is not None:
        hi = np.percentile(S_tf, clip_percentile)
        lo = np.percentile(S_tf, 100.0 - clip_percentile)
        vmin, vmax = float(lo), float(hi)

    plt.figure(figsize=figsize)
    im = plt.imshow(S_tf, aspect="auto", vmin=vmin, vmax=vmax,
                    origin="lower", extent=[0, len(feat_names), t_hr[0], t_hr[-1]])

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(ticks=np.linspace(t_hr[0], t_hr[-1], 5),
               labels=[f"{h:.0f}h" for h in np.linspace(t_hr[0], t_hr[-1], 5)])
    plt.xticks(ticks=np.arange(len(feat_names)), labels=feat_names, rotation=90)
    ttl = title if title is not None else f"LSTM Grad×Input Heatmap (cumulative {t_hr[0]:.0f}–{t_hr[-1]:.0f}h)"
    plt.title(ttl)
    plt.xlabel("Features")
    plt.ylabel("Time (hr)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

# End of module
