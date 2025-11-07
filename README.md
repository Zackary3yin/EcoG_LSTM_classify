# EcoG_LSTM_classify

BiLSTM‑attention pipeline for EEG/ECoG epilepsy classification with time‑increasing evaluation, Grad×Input feature importance, and clean artifacts for analysis and reporting.

---

## 🗂 Repository layout

```
EcoG_LSTM_classify/
├─ ecog_lstm.py            # Importable toolkit (data → model → eval → saliency → plots)
├─ run_eeg_pipeline.py     # One‑click runner for 60‑min & 10‑min setups
├─ Main_Pipline.ipynb      # Notebook workflow (interactive)
├─ Data/                   # <put your CSVs here>
└─ outputs_lstm/           # Results written here
   ├─ time60min/
   └─ time10min/
```

---

## ✨ Features

* **Time‑increasing evaluation**: stratified K‑Fold (optional LOO), metrics across growing time windows.
* **BiLSTM + attention** model with class imbalance options (`pos_weight`, duplication).
* **Grad×Input saliency** per cumulative window, Top‑K tables, and **heatmaps with real hour y‑axis**.
* **Data sanity views**: hourly label distributions (usable‑cumulative & valid‑exact).
* Clean separation of **60‑min** and **10‑min** resolutions.

---

## 🔧 Requirements

Python 3.9+ and:

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
```

> For CUDA, install PyTorch from pytorch.org per your GPU/CUDA version.

---

## 📄 Data format

A single CSV containing:

* `study_id` (patient ID)
* Label column: `epilepsy_label` (0/1)
* Any **one** time column (hours preferred):

  * Hours: `tbi_time_10min_start_hr`, `bin_end_hr`, `bin_start_hr`, `time_hr`, `time_hours`
  * Minutes (auto‑converted): `bin_start_min`, `bin_end_min`, `time_min`, `time_minutes`
  * Fallback: `time` (treated as hours)
* EEG feature columns (see `FEATURE_LIST`).

Pipeline steps: infer time → filter `[start_hr, end_hr]` → select present features → mean impute → grid to `(N,T,F)` with `step_minutes`.

---

## 🚀 Quick start (script)

Run **both** resolutions and produce artifacts to separate folders:

```bash
python run_eeg_pipeline.py
```

Artifacts appear in:

```
outputs_lstm/
  ├─ time60min/
  └─ time10min/
```

> Edit `Data/...csv` paths and `FEATURE_LIST` in the runner if needed.

### Only one resolution

Comment out the other block in `__main__` and run the script again.

---

## 📓 Quick start (notebook)

Open **`Main_Pipline.ipynb`** and run cells in order:

1. Imports & paths (define `FEATURE_LIST`)
2. Build `Config` (60‑min and/or 10‑min)
3. Data stats → Model performance → Save perf plots → Saliency → Heatmaps

Tips:

```python
%matplotlib inline
from tqdm.notebook import tqdm  # nicer progress bars
```

Pass `cfg` to heatmap calls so the y‑axis shows real hours:

```python
plot_saliency_heatmap_from_out(out, block_end_hr=24, cfg=cfg60,
                               save_path="outputs_lstm/time60min/heatmaps/heatmap_cumulative_to_24h.png",
                               show=False)
```

---

## ⚙️ Configuration (key fields)

Defined via `Config` in `ecog_lstm.py` (see `run_eeg_pipeline.py` for examples):

* `path_csv`: input CSV path
* `start_hr`, `end_hr`: analysis window (hours)
* `step_minutes`: time resolution (e.g., 60 or 10)
* `features`: explicit feature list (intersected with columns)
* `eval_stride_hr`: spacing of evaluation timepoints (hours)
* Training: `epochs`, `batch_size`, `hidden_size`, `num_layers`, `bidirectional`, `dropout`
* Imbalance: `use_duplication`, `use_pos_weight`
* Feature filtering: `restrict_to_intersection`, `min_patient_coverage`
* Data sufficiency: `min_nonzero_frac` (min fraction of observed steps required)

**Saliency windows are cumulative**: for block end hour `be`, the window is `[start_hr, be]`. Example: `start_hr=12`, `be=24` → 12–24 h.

---

## 📊 Outputs (per resolution)

Inside `outputs_lstm/time60min/` and `outputs_lstm/time10min/`:

* **Performance**

  * `has_epilepsy_lstm_[60|10]min_5fold.csv`
  * `perf_kfold.png` (AUC/ACC/ACC0/ACC1 mean±std vs time)
* **Saliency**

  * `saliency_per_block.csv` (all features × windows)
  * `top15_per_block.csv`
  * `heatmaps/heatmap_cumulative_to_[24|48|72|120]h.png`
* **Label distributions**

  * `hourly_usable_cumulative_*.csv`
  * `hourly_valid_exact_*.csv`

> Filenames may differ if you change `out_csv`, ranges, or blocks.

---

## ♻️ Reproducibility

* Seeds are set internally (42) for eval; feel free to expose `set_seed(42)`.
* Consider pinning versions in `requirements.txt`.
* Commit your data snapshot/CSV schema and the exact `FEATURE_LIST` used.

---

## 🧯 Troubleshooting

* **Progress bar not visible**: ensure loops use `tqdm(..., file=sys.stdout, dynamic_ncols=True)`; run unbuffered in logs: `python -u run_eeg_pipeline.py`.
* **Heatmap y‑axis starts at 0**: pass `cfg` to `plot_saliency_heatmap_from_out` (runner & notebook examples do this).
* **`plt` not defined**: the module’s plotting functions import `matplotlib` where needed; add `import matplotlib.pyplot as plt` if you write new plot code.

---

## 📜 License

MIT (or update to your preferred license).
