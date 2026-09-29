# ML training

How every model in the system is trained: which data it uses, how that data is prepared and
split, which models are compared and tuned, and where the results go. The measured results are
in [MODEL_EVALUATION.md](MODEL_EVALUATION.md). What data exists and why each dataset is (or is
not) used is in [DATASET_INVENTORY.md](DATASET_INVENTORY.md).

All training is **offline** (`python -m traffic_intel.ai.training ...`). The API process never
trains and never imports PyTorch. The pipeline loads trained artifacts through the existing
interfaces (`Detector`, `Tracker`, `Forecaster`, series anomaly detectors, `CongestionClassifier`),
selected in `configs/pipeline.yaml`.

```
raw dataset --adapter--> canonical representation --preprocessing (fit on TRAIN)--> features
   (datasets/adapters)       (core/feature_schema.py,      (forecasting/data/preprocess.py,
                              Tracks, IncidentAnnotation)   one definition for training and serving)
          --> chronological / by-sequence / by-video split --> models + small tuning on VALIDATION
          --> TEST metrics, plots --> models/<task>/<name>/<version>/ (+ preprocessor, metadata)
          --> experiments/<task>/<id>/ record --> promotion --> configs/pipeline.yaml --> live inference
```

---

## 1. Before training: discovery and data quality

```bash
cd backend
python -m traffic_intel.datasets inventory   # reports/dataset_inventory.json: every dataset on disk
python -m traffic_intel.datasets quality     # reports/data_quality_report.{json,md}
```

The quality report checks:

- missing values, gaps and duplicates;
- irregular sampling and DST handling;
- values outside physical ranges, and stuck sensors;
- invalid, out-of-image and tiny boxes;
- label validity and class balance;
- adjacent-frame redundancy;
- train/test overlap, plus the actual chronological split of every training config.

---

## 2. Every experiment starts with a smoke run

```bash
python -m traffic_intel.ai.training train <task> --config ../configs/training/<file>.yaml --smoke
```

A smoke run exercises the whole pipeline in seconds to minutes: one epoch, a subset of the data,
the first value of each grid. It writes `experiments/<task>/<id>_smoke/` (figures included, never
`reports/`), **registers nothing and is not indexed**. Its numbers are pipeline checks, never
results.

---

## 3. Tasks

### 3.1 Vehicle detection: YOLO11 fine-tuning on UA-DETRAC

| | |
|---|---|
| Config | `configs/training/detection_ua_detrac.yaml` |
| Code | `ai/training/detection.py` |
| Data | UA-DETRAC official **train** (60 sequences) / **test** (40 sequences), through the `ua_detrac` adapter (`split=train|test`) |
| Classes | project `car`, `bus`, `truck` (manifest mapping: van → car, others → truck). No pedestrian / bicycle / motorcycle labels exist in UA-DETRAC. |
| Split | validation = 10 **whole** training sequences (weather-stratified, deterministic). Frames of one sequence are never in two splits. |
| Preprocessing | every 10th annotated frame (adjacent frames are near-duplicates, median box IoU 0.85–0.94). Ignored regions greyed out (value 114) and labels centred in them dropped. Images without an annotation entry are excluded, never assumed empty. Output: `data/processed/ua_detrac_yolo/` (YOLO format, `split.json` documents the split). |
| Models | COCO-pretrained YOLO11s and YOLO11m, **evaluated zero-shot** (baseline), then **fine-tuned** (transfer learning, not from scratch) |
| Training | Ultralytics, `imgsz` 960 (native width), up to 30 epochs, early stopping on validation mAP50-95 (patience 8), cosine LR, AMP. Batch 8 / 4 on the 8 GB GPU; Ultralytics accumulates gradients to a nominal batch of 64. |
| Evaluation | the held-out **test** split through the **production inference path** (`YOLODetector.detect` + `ClassMapper`), so COCO and fine-tuned models are scored identically. Precision / recall at the pipeline's confidence threshold 0.25; mAP50 and mAP50-95 with Ultralytics' `ap_per_class` (10 IoU thresholds). Per class, at 960 and 1280 px, plus latency. Detections centred in ignored regions do not count. |
| Class imbalance | follow-up config `detection_ua_detrac_rfs.yaml`: **repeat-factor sampling** (LVIS) of training images, r = max(1, √(t / f_c)) for the rarest class in the image, t = 0.5 fixed from training statistics (trucks in 3.5% of training images → ×3.8; buses in 31% → ×1.3). Only the training list changes (`data/processed/ua_detrac_yolo/train_rfs_t0.5.txt`); validation and test are untouched. The first run showed that plain fine-tuning loses truck AP. |
| Zero-shot only | `--baselines-only` evaluates the COCO weights without training (used for YOLO11m, whose first fine-tuning run was interrupted before its baseline rows were written). Rows of all runs are merged by (model, input size), newest first. |
| Artifacts | `models/detection/<name>/<version>/model.pt` + `metadata.json`. `configs/pipeline.yaml` can name a registered model: `weights: registry:detection/<name>@production`. |

### 3.2 Tracking: ByteTrack vs BoT-SORT on UA-DETRAC

| | |
|---|---|
| Config | `configs/training/tracking_ua_detrac.yaml` |
| Code | `ai/training/tracking.py` |
| Data | the 10 validation sequences of 3.1 (tuning) and every 2nd official test sequence (20 sequences, evaluation) |
| Method | trackers are not learned. The registered `Tracker` implementations are compared, and their association thresholds (`track_high_thresh`, `match_thresh`, `new_track_thresh`) are chosen with a small grid on **validation** MOTA. Every other parameter comes from `configs/pipeline.yaml`. Detections come from the production detector at confidence 0.1 (ByteTrack's second round uses low-score boxes), cached as Parquet in `data/interim/ua_detrac_detections/`. |
| Metrics | CLEAR-MOT at IoU 0.5: MOTA, MOTP, precision, recall, identity switches, plus tracker time per frame. Ground truth and predictions centred in ignored regions are removed. |

### 3.3 Traffic forecasting (+5 / +15 / +30 min)

| | |
|---|---|
| Configs | `forecasting_metr_la_{traffic_only,context}.yaml`, `forecasting_pems_bay_{traffic_only,context}.yaml`, `forecasting_pems08.yaml`, `forecasting.yaml` (synthetic junction = live feature set) |
| Code | `ai/training/experiment.py`, `forecasting/data/{sources,preprocess,windows}.py`, `forecasting/models/*` |
| Data | adapters: METR-LA / PEMS-BAY (5-min speed; zeros = outage → missing; local time → UTC, non-existent DST times dropped), PEMS08 (5-min flow / occupancy / speed), synthetic junction (1-min live feature set) |
| Features | the canonical schema (`core/feature_schema.py`) through one `Preprocessor`: gaps ≤ `max_gap_fill_steps` interpolated, longer gaps never filled (windows touching them are excluded); lags, rolling means, calendar (hour, weekday, holiday, rush hour) and weather. Scaling statistics come from **train only**. |
| Context variants | `*_context` configs add calendar and Open-Meteo archive weather for the dataset's **own** location and dates (fetched once, a few KB, CC BY 4.0, cached). The traffic-only twin is identical except for these columns. |
| Split | **chronological** train / val / test (70/10/20 for METR-LA / PEMS-BAY, 60/20/20 otherwise). A sample belongs to a period only if its forecast origin **and** its furthest target lie inside it; inputs may reach back into the past. No shuffling. |
| Models | persistence, seasonal naive (same time yesterday), linear regression, random forest, XGBoost, LSTM, GRU, Transformer. All models see the **same windows**. |
| Tuning | a small matrix: `random_forest` vs `random_forest_deep`; XGBoost at depth 4 / 6 / 8 with early stopping. Deep models: early stopping on validation loss, 2–3 seeds each (the best **validation** seed is kept; mean ± std over seeds is reported). |
| Selection | lowest **validation** RMSE on the primary target. Test is reporting only. |
| Metrics | MAE, RMSE, R², MAPE (only where \|y\| ≥ 1 unit), skill vs persistence; per horizon and averaged |
| Artifacts | `models/forecasting/<model>/<experiment>/` (model, `preprocessor.json`, `params.json`, `metadata.json` with the feature-schema hash). A model whose preprocessing hash does not match refuses to load. |

**Which forecaster serves the live system.** Only a model trained on the **live feature set and
resolution** (synthetic junction today; own recordings in future) can serve the pipeline. The
real-data experiments (METR-LA, PEMS-BAY, PEMS08) are the evidence for *which model family* to
use. They stay CANDIDATE and are never promoted: `pipeline.yaml` names the live model explicitly
instead of relying on `auto`.

### 3.4 Congestion level

| | |
|---|---|
| Configs | `congestion_level_{metr_la,pems_bay,synthetic}.yaml` |
| Code | `ai/training/congestion.py`, `traffic/congestion/rules.py` (`levels_frame`) |
| Why not "classify the current state" | Current levels are **defined** by the explainable rules (`configs/congestion_rules.yaml`) and no dataset carries independent congestion labels. A classifier trained on rule labels of the same window would only imitate the rules, so it is **not trained**. |
| What is trained | the **future** level (+5 / +15 / +30 min). The label is the rule level of the **observed** value at t + h. Labels are computed with `RuleBasedCongestionClassifier.levels_frame`, the vectorised twin of the live `classify()` (identical on 400 random windows, overrides included, tested), using only the indicators the dataset measures. |
| Free-flow speed | per series: the 85th percentile of **train-period** speeds (loop detectors), or the network's value for live approaches |
| Models | persistence of the current level; *regression → rules* (XGBoost forecasts the values, then the same rules: this is what the live service does); logistic regression, random forest and XGBoost classifiers. Small grids (C, class weights, depth) are chosen on **validation macro-F1**. |
| Metrics | accuracy, balanced accuracy, macro / weighted precision, recall and F1, per class, confusion matrix |

### 3.5 Anomaly detection: live feature set (synthetic junction)

| | |
|---|---|
| Config | `configs/training/anomaly.yaml` |
| Code | `ai/training/anomaly.py`, `anomaly/detectors/{statistical,learned}.py` |
| Data | synthetic junction series with injected capacity drops (ground truth). Chronological: the learned detectors are fitted on **normal** windows of the first 14 days, and everything is evaluated on the remaining days. |
| Methods | statistical detectors (sudden slowdown, queue growth, density z-score, flow imbalance, EWMA, z-score), Isolation Forest, autoencoder, and the fused incident score |
| Metrics | PR-AUC, ROC-AUC, precision / recall / F1 at the configured alarm threshold (window level; accuracy is not used, since incidents are under 1% of windows), event recall (all and impactful incidents), false alarms per hour, detection delay |

### 3.6 Incident detection on real video: AI City 2021 Track 4

| | |
|---|---|
| Config | `configs/training/incident_aic21.yaml` |
| Code | `ai/training/incident_video.py` |
| Data | the 100 labelled training videos (18 "anomaly" intervals). The 150 test videos have no public labels and are excluded. |
| Perception | production detector and tracker through the `Detector` / `Tracker` interfaces, 1280 px, 5 processed frames per second. Tracker thresholds are the ones tuned on UA-DETRAC validation (3.2): a different dataset, so nothing is tuned on Track 4 labels. Frame-level detections and tracks go to `data/interim/aic21_track4/*.parquet`. |
| Features | uncalibrated cameras, so the live metric features cannot be computed. Scale-normalised **pixel** features per 10-s window (sliding 5 s): tracks visible, speed in box heights per second, stationary tracks and their longest stationary duration, and an **ID-free stationary persistence** (how long a box has kept overlapping the same place, robust to detector flicker and track-id switches). The persistence feature was added after a 6-video pipeline check showed that distant stopped vehicles fragment into short tracks; that check is recorded as a smoke run. |
| Split | **by video**, stratified so labelled videos appear in train, val and test |
| Methods | stationary rule (longest persistence, threshold chosen on validation), Isolation Forest and autoencoder (fitted on normal windows of training videos) |
| Metrics | window: ROC-AUC, PR-AUC, precision / recall / F1. Event (video): first alarm → precision / recall / F1 and detection delay. AI City timing score (F1 within 10 s of the start, RMSE, S4) for the rule with its back-dated onset. |
| Calibration | Platt and isotonic maps of the best score, fitted on validation windows and judged on test windows (Brier score, reliability plot) |
| Deployment | **none.** These models use pixel features and are evaluation artefacts in the experiment folder. The live incident score stays **UNCALIBRATED**; see MODEL_EVALUATION.md. |

### 3.7 Risk

No dataset contains risk labels, so no supervised risk model is trained. The explainable,
weighted risk engine (`configs/risk.yaml`) is kept.

---

## 4. Experiment records

Every run writes `experiments/<task>/<experiment_id>/` and appends to
`experiments/<task>/index.csv`. The record contains:

- `config.yaml` (resolved configuration);
- the dataset: source, series or sequences, periods / splits, sample counts, features, targets,
  preprocessing hash;
- per model: type, hyperparameters, seed(s), fit time, inference time per sample or image,
  epochs, and the artifact path;
- validation metrics, used for selection;
- test metrics;
- grid-search tables (`grid_search_val.csv`);
- the report (`comparison.md` / `report.md`);
- figures in `reports/figures/<task>/<experiment_id>/`.

Two commands turn the records into the evaluation material. Neither retrains anything, and neither
types a number by hand.

```bash
python -m traffic_intel.ai.training summary   # reports/model_evaluation_tables.md (latest completed run per config)
python -m traffic_intel.ai.training figures   # reports/figures/ + INDEX.md
```

`figures` re-scores each forecasting experiment's validation-selected model and persistence from
the **registry**, on test windows rebuilt from the recorded config. It fails if the recomputed RMSE
differs from the record. The resulting error-over-time and error-by-hour figures therefore come
from the exact artifacts that are served. Anomaly runs save their test scores
(`scores.parquet`) for the score timeline.

Models go to `models/<task>/<name>/<version>/` with `metadata.json` (task, framework,
feature-schema hash, training data and periods, parameters, metrics, status). Versions are never
overwritten. `PRODUCTION` in a model folder names the promoted version.

```bash
python -m traffic_intel.ai.training list forecasting
python -m traffic_intel.ai.training promote forecasting <model> <version|latest>
python -m traffic_intel.ai.training promote detection <name> <version>
```

## 5. GPU

PyTorch and Ultralytics use CUDA when it is available and fall back to the CPU otherwise.
Detection batch sizes are set for an 8 GB card. On Windows, every data-loader worker loads the
CUDA runtime (memory commit), so `workers: 2`. Run one GPU-heavy job at a time: two jobs
oversubscribing GPU memory spill into shared memory (about 10× slower) and can crash.

## 6. Reproduce everything

```bash
cd backend
python -m traffic_intel.datasets inventory && python -m traffic_intel.datasets quality
for c in forecasting_metr_la_traffic_only forecasting_metr_la_context forecasting_pems_bay_traffic_only \
         forecasting_pems_bay_context forecasting_pems08 forecasting; do
  python -m traffic_intel.ai.training train forecasting --config ../configs/training/$c.yaml; done
for c in congestion_level_metr_la congestion_level_pems_bay congestion_level_synthetic; do
  python -m traffic_intel.ai.training train congestion --config ../configs/training/$c.yaml; done
python -m traffic_intel.ai.training train anomaly --config ../configs/training/anomaly.yaml
python -m traffic_intel.ai.training train detection --config ../configs/training/detection_ua_detrac.yaml
python -m traffic_intel.ai.training train detection --config ../configs/training/detection_ua_detrac.yaml \
       --only yolo11m_ua_detrac --baselines-only
python -m traffic_intel.ai.training train detection --config ../configs/training/detection_ua_detrac_rfs.yaml
python -m traffic_intel.ai.training train tracking --config ../configs/training/tracking_ua_detrac.yaml
python -m traffic_intel.ai.training train incident --config ../configs/training/incident_aic21.yaml
python -m traffic_intel.ai.training summary && python -m traffic_intel.ai.training figures
```

Seeds are fixed in every config. Deep-learning results on GPU can differ in the last digits
between runs (non-deterministic CUDA kernels). Deep models are therefore trained with several
seeds and the spread is reported.
