# -*- coding: utf-8 -*-
"""
Epilepsy LSTM pipeline (supports 10-min and 60-min aggregated tables)
--------------------------------------------------------------------
- Choose time resolution (10min or 60min) by pointing to the CSV and specifying `step_minutes`.
- Explicitly pass the feature list (we'll intersect with available columns and warn on missing).
- Select a time window in hours, e.g., 0–72h (inclusive end).
- Preprocess: padding to full window, missing/inf handling, patient-level label aggregation,
  assemble (N, T, F) sequences.
- Class imbalance options + Time-increasing + Leave-One-Out (patient-level) evaluation.
- Same LSTM(+attention) model structure as before (without feature-importance in this version).

Example usage is at the bottom of this file.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# -------------------------
# Globals
# -------------------------
NON_FEATURE_COLS = ['study_id', 'h5_folder_id', 'has_epilepsy',
                    'bin_start_hr', 'bin_end_hr',
                    'bin_start_min', 'bin_end_min']  # include minute-based variants


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@dataclass
class PipelineConfig:
    path_csv: str
    start_hr: int = 0
    end_hr: int = 72
    step_minutes: int = 60         # 10 or 60
    features: Optional[List[str]] = None   # explicit feature list; if None -> auto(all except NON_FEATURE_COLS)
    agg_label: str = 'any_true'    # 'any_true' or 'majority'
    restrict_to_intersection: bool = True
    min_patient_coverage: float = 0.95     # keep features with >= coverage across patients
    min_nonzero_frac: float = 0.10         # require >=10% nonzero time-steps per patient (dynamic with L)
    epochs: int = 60
    batch_size: int = 64
    hidden_size: int = 32
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.2
    use_duplication: bool = True
    use_pos_weight: bool = True
    out_csv: str = 'has_epilepsy_lstm_LOO.csv'
    focus_ylim: Optional[Tuple[float, float]] = (0.7, 1.0)


# -------------------------
# Time column inference
# -------------------------
def infer_time_in_hours(df: pd.DataFrame) -> pd.Series:
    """
    Try to infer a 'time (hours since start)' column.
    Priority order:
      1) 'bin_start_hr' (already hours)
      2) 'bin_start_min'/'bin_start_minutes' (convert /60)
      3) 'time_hr' / 'time_hours' (already hours)
      4) numeric 'time' (assume hours)
    """
    candidates_hr = ['bin_start_hr', 'time_hr', 'time_hours']
    for c in candidates_hr:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                return s

    candidates_min = ['bin_start_min', 'bin_start_minutes', 'time_min', 'time_minutes']
    for c in candidates_min:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                return s / 60.0

    if 'time' in df.columns:
        s = pd.to_numeric(df['time'], errors='coerce')
        if s.notna().any():
            return s

    raise ValueError("无法推断时间列。请确保存在 'bin_start_hr' 或 'bin_start_min' 等可解析的列。")


# -------------------------
# Feature selection helper
# -------------------------
def choose_feature_columns(df: pd.DataFrame, explicit_features: Optional[List[str]]) -> List[str]:
    if explicit_features is None:
        feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    else:
        exist = [c for c in explicit_features if c in df.columns]
        missing = [c for c in explicit_features if c not in df.columns]
        if missing:
            print(f"[Warn] 显式指定的特征中有 {len(missing)} 列缺失：{missing[:10]}{' ...' if len(missing)>10 else ''}")
        feat_cols = exist
    # drop all-NaN
    feat_cols = [c for c in feat_cols if not df[c].isna().all()]
    return feat_cols


# -------------------------
# Preprocess & assemble sequences
# -------------------------
def preprocess_and_build_sequences(cfg: PipelineConfig):
    # read
    df = pd.read_csv(cfg.path_csv)

    # sanity
    needed = ['study_id', 'has_epilepsy']
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")

    # infer time in hours
    time_hr = infer_time_in_hours(df)
    df = df.assign(_time_hr=time_hr)
    df = df.dropna(subset=['study_id', '_time_hr', 'has_epilepsy']).copy()

    # restrict window [start_hr, end_hr] inclusive
    df = df[(df['_time_hr'] >= cfg.start_hr) & (df['_time_hr'] <= cfg.end_hr)].copy()
    if df.empty:
        raise ValueError("所选时间窗内没有数据")

    # choose features
    feat_cols = choose_feature_columns(df, cfg.features)
    if len(feat_cols) == 0:
        raise ValueError("没有可用特征列。请检查显式特征列表或数据表。")

    # patient-level coverage -> keep features over threshold
    if cfg.restrict_to_intersection:
        coverage_per_pid = (
            df.groupby('study_id')[feat_cols]
              .apply(lambda g: g.notna().any(axis=0))
        )  # index=pid, columns=feat, bool
        feat_coverage = coverage_per_pid.mean(axis=0)  # [0,1]
        kept_feats = feat_coverage[feat_coverage >= cfg.min_patient_coverage].index.tolist()
        if len(kept_feats) == 0:
            top_k = min(50, len(feat_cols))
            kept_feats = feat_coverage.sort_values(ascending=False).index[:top_k].tolist()
            print(f"[Warn] 严格交集为空，已退化为覆盖率 Top-{top_k} 特征。")
        print(f"[Info] 交集特征数: {len(kept_feats)} / {len(feat_cols)} (阈值={cfg.min_patient_coverage:.2f}); 示例: {kept_feats[:8]}")
        feat_cols = kept_feats

    # clean values for the chosen features
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    # per-feature mean impute
    df[feat_cols] = df[feat_cols].astype(float).fillna(df[feat_cols].mean(numeric_only=True))
    # final safety
    df[feat_cols] = df[feat_cols].apply(lambda col: np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0))

    # label aggregation per patient
    lab_df = df[['study_id', 'has_epilepsy']].copy()
    if cfg.agg_label == 'any_true':
        lab_agg = lab_df.groupby('study_id')['has_epilepsy'].any().astype(int).rename('y')
    elif cfg.agg_label == 'majority':
        lab_agg = (lab_df.groupby('study_id')['has_epilepsy'].mean().round().astype(int).rename('y'))
    else:
        raise ValueError("agg_label 仅支持 'any_true' 或 'majority'")

    # time index resolution
    step_min = cfg.step_minutes
    total_steps = int((cfg.end_hr - cfg.start_hr) * 60 / step_min) + 1  # inclusive end

    def hour_to_index(h: float) -> int:
        # convert hours -> step index
        return int(round((h - cfg.start_hr) * 60.0 / step_min))

    # build sequences
    X_list, y_list, pids = [], [], []
    F = len(feat_cols)

    for pid, g in df.groupby('study_id'):
        seq = np.zeros((total_steps, F), dtype=np.float32)
        for _, row in g.iterrows():
            idx = hour_to_index(float(row['_time_hr']))
            if 0 <= idx < total_steps:
                seq[idx] = row[feat_cols].values.astype(np.float32)
        if (seq != 0).any():
            X_list.append(seq)
            y_list.append(int(lab_agg.get(pid, 0)))
            pids.append(pid)

    if not X_list:
        X = np.empty((0, total_steps, F), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        return X, y, feat_cols, pids, total_steps

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    print(f"[Info] 最终样本: N={len(pids)}, T={total_steps} (step={step_min}min), F={F}")
    return X, y, feat_cols, pids, total_steps


# -------------------------
# Model
# -------------------------
class LSTMBinary(nn.Module):
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
        # x: [B, T, F]
        H, _ = self.lstm(x)                       # [B, T, D*H]
        H = self.dropout(H)
        u = torch.tanh(self.attn_proj(H))         # [B, T, D*H]
        scores = self.attn_score(u).squeeze(-1)   # [B, T]
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        z = torch.sum(alpha * H, dim=1)           # [B, D*H]
        return self.fc(z).view(-1)                # logits [B]


def balance_by_duplication(X, y):
    classes = np.unique(y)
    counts = {c: int((y == c).sum()) for c in classes}
    max_n = max(counts.values())

    X_out, y_out = [], []
    for c in classes:
        Xc = X[y == c]
        n_needed = max_n - len(Xc)
        if n_needed > 0:
            X_aug = resample(Xc, replace=True, n_samples=n_needed, random_state=42)
            Xc = np.concatenate([Xc, X_aug], axis=0)
        X_out.append(Xc)
        y_out.append(np.full((len(Xc),), c, dtype=np.int64))
    return np.concatenate(X_out, axis=0), np.concatenate(y_out, axis=0)


# -------------------------
# LOO time-increasing evaluation
# -------------------------
def run_time_increasing_loo(cfg: PipelineConfig) -> pd.DataFrame:
    X_np, y_np, feat_cols, pids, T = preprocess_and_build_sequences(cfg)
    N, F = X_np.shape[0], X_np.shape[2]
    if N < 2:
        raise ValueError("LOO 需要至少 2 名患者。")

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info][LOO] Device: {device}, N={N}, T={T}, F={F}, step={cfg.step_minutes}min")

    loo = LeaveOneOut()
    records = []

    # time-increasing: L = 1..T (inclusive window end in steps)
    for L in tqdm(range(1, T + 1), desc="Time steps (LOO)"):
        # aggregate per-hour label doesn't change across L; but input is truncated
        Xw = X_np[:, :L, :].copy()
        yw = y_np.copy()

        # filter patients with too few nonzero steps
        nonzero_counts = (Xw != 0).any(axis=2).sum(axis=1)
        min_steps = max(1, int(cfg.min_nonzero_frac * L))
        mask_pat = nonzero_counts >= min_steps
        Xw, yw = Xw[mask_pat], yw[mask_pat]

        if len(np.unique(yw)) < 2 or len(yw) < 2:
            # still add a record to keep the time-axis consistent
            hour_tp = cfg.start_hr + (L - 1) * (cfg.step_minutes / 60.0)
            records.append({'time_hr': hour_tp, 'AUC': np.nan, 'Accuracy': np.nan,
                            'Acc_class_0': np.nan, 'Acc_class_1': np.nan})
            continue

        all_probs, all_preds, all_true = [], [], []

        for train_idx, test_idx in loo.split(Xw):
            Xtr, Xte = Xw[train_idx].copy(), Xw[test_idx].copy()
            ytr, yte = yw[train_idx], yw[test_idx]

            # if training has one class only -> majority baseline for this test fold
            if len(np.unique(ytr)) < 2:
                majority = int(np.bincount(ytr).argmax())
                all_true.append(int(yte[0]))
                all_preds.append(majority)
                all_probs.append(float(majority))
                continue

            # fit scaler on training only
            scaler = MinMaxScaler()
            scaler.fit(Xtr.reshape(-1, F))
            Xtr = scaler.transform(Xtr.reshape(-1, F)).reshape(Xtr.shape[0], L, F)
            Xte = scaler.transform(Xte.reshape(-1, F)).reshape(Xte.shape[0], L, F)

            # balance
            if cfg.use_duplication:
                Xtr_bal, ytr_bal = balance_by_duplication(Xtr, ytr)
            else:
                Xtr_bal, ytr_bal = Xtr, ytr

            train_loader = DataLoader(
                TensorDataset(torch.tensor(Xtr_bal, dtype=torch.float32),
                              torch.tensor(ytr_bal, dtype=torch.float32)),
                batch_size=cfg.batch_size, shuffle=True
            )

            # model
            model = LSTMBinary(
                input_dim=F, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                bidirectional=cfg.bidirectional, dropout=cfg.dropout
            ).to(device)

            # loss
            if cfg.use_pos_weight:
                pos = float((ytr_bal == 1).sum())
                neg = float((ytr_bal == 0).sum())
                pw = max(1.0, neg / max(1.0, pos))
                pos_weight = torch.tensor([pw], dtype=torch.float32).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()

            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # train
            for _ in range(cfg.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb).view(-1)
                    loss = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # test one patient
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(Xte, dtype=torch.float32).to(device)
                prob = torch.sigmoid(model(xb).view(-1)).cpu().numpy()[0]
                pred = int(prob >= 0.5)
            all_true.append(int(yte[0]))
            all_probs.append(float(prob))
            all_preds.append(pred)

        # aggregate metrics at this time step
        y_true = np.array(all_true, dtype=int)
        y_pred = np.array(all_preds, dtype=int)
        y_prob = np.array(all_probs, dtype=float)

        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception:
                auc = np.nan

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc_c0 = (cm[0,0] / cm[0].sum()) if cm[0].sum() > 0 else np.nan
        acc_c1 = (cm[1,1] / cm[1].sum()) if cm[1].sum() > 0 else np.nan

        hour_tp = cfg.start_hr + (L - 1) * (cfg.step_minutes / 60.0)
        records.append({'time_hr': hour_tp, 'AUC': auc, 'Accuracy': acc,
                        'Acc_class_0': acc_c0, 'Acc_class_1': acc_c1})

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(cfg.out_csv, index=False)
    print(f"[OK][LOO] Metrics saved to {cfg.out_csv}")

    # plot
    plt.figure(figsize=(9,5))
    plt.plot(df_metrics['time_hr'], df_metrics['AUC'], '-o', label='AUC')
    plt.plot(df_metrics['time_hr'], df_metrics['Accuracy'], '-s', label='Accuracy')
    plt.plot(df_metrics['time_hr'], df_metrics['Acc_class_0'], '--', label='Acc class 0 (no epilepsy)')
    plt.plot(df_metrics['time_hr'], df_metrics['Acc_class_1'], '--', label='Acc class 1 (has epilepsy)')
    plt.xlabel('Time since start (hours)'); plt.ylabel('Score')
    plt.title(f"LSTM (has_epilepsy) over time - LOO (step={cfg.step_minutes} min)")
    if cfg.focus_ylim is not None:
        plt.ylim(*cfg.focus_ylim)
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(cfg.out_csv.replace('.csv', '.png'), dpi=180)
    plt.close()
    print(f"[OK] Plot saved to {cfg.out_csv.replace('.csv', '.png')}")

    return df_metrics


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    set_seed(42)

    # === Choose one of the following ===
    # 1) 60-min resolution
    # csv_path = "Data/all_patients_60min_aggregated.csv"
    # step_minutes = 60

    # 2) 10-min resolution
    # csv_path = "Data/all_patients_10min_aggregated.csv"
    # step_minutes = 10

    # You can also use absolute paths like "/mnt/data/all_patients_60min_aggregated.csv"
    csv_path = "Data/all_patients_60min_aggregated.csv"
    step_minutes = 60

    # Explicit feature list (example). Replace with your own list:
    # If None, the pipeline will auto-use all columns except NON_FEATURE_COLS.
    explicit_feats = None
    # explicit_feats = ["BCI", "entropy", "amplitude", "spike_rate", "GPD_prob", "LPD_prob"]

    cfg = PipelineConfig(
        path_csv=csv_path,
        start_hr=0, end_hr=72,
        step_minutes=step_minutes,
        features=explicit_feats,
        agg_label='any_true',
        restrict_to_intersection=True,
        min_patient_coverage=0.95,
        min_nonzero_frac=0.10,
        epochs=80,
        batch_size=64,
        hidden_size=32,
        num_layers=1,
        bidirectional=True,
        dropout=0.2,
        use_duplication=True,
        use_pos_weight=True,
        out_csv=f"has_epilepsy_lstm_{step_minutes}min_0_72_loo.csv",
        focus_ylim=(0.7, 1.0)
    )

    run_time_increasing_loo(cfg)
