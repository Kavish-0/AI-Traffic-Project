# Model evaluation

The measured results of every trained model, and why each model that serves the live system was
chosen. How the models are trained is in [ML_TRAINING.md](ML_TRAINING.md). What data exists, and why
each dataset is or is not used, is in [DATASET_INVENTORY.md](DATASET_INVENTORY.md).

Every number here comes from an experiment record under `experiments/<task>/<experiment_id>/`:

- the full tables are generated from those records: `python -m traffic_intel.ai.training summary`
  writes `reports/model_evaluation_tables.md`;
- the figures are generated the same way: `python -m traffic_intel.ai.training figures` writes
  `reports/figures/` (index: `reports/figures/INDEX.md`).

The forecasting figures re-score the **registered** models on rebuilt test windows. The figure
command stops if the recomputed RMSE differs from the recorded one. All six forecasting
experiments passed that check, so the artifacts in `models/` reproduce the reported numbers.

## How models are chosen

1. **Validation first.** Every model is selected on the validation split (chronological periods,
   whole video sequences, or whole videos). Test metrics are reported only and never used to pick a
   model or a threshold.
2. **Then speed, complexity and compatibility.** A model can serve the live system only if it:
   - uses the live feature set at the live resolution;
   - runs inside the pipeline runtime;
   - keeps the API process free of PyTorch (the API reads model metadata only).

   A model that is only marginally better but costs more in any of these is not preferred.
3. **Candidate by default.** Training registers every model as CANDIDATE. Promotion is an explicit
   command, and `configs/pipeline.yaml` names the serving model. A model is never picked because it
   is "latest".

## What serves the live system

| Task | Serving now | Evidence | Status |
|---|---|---|---|
| Vehicle detection | YOLO11m, COCO-pretrained, 1280 px | UA-DETRAC test: zero-shot COCO beats fine-tuning overall; fine-tuning cannot keep pedestrian / bicycle / motorcycle (§5) | unchanged |
| Tracking | ByteTrack, `new_track_thresh` 0.4 (was 0.6) | UA-DETRAC test MOTA 0.737 → 0.749; BoT-SORT equal MOTA at 33× the time (§6) | **tuned** |
| Forecasting (+5/+15/+30 min) | `xgboost_d8` trained on the live feature set | best on validation for the live domain, 33% lower RMSE than persistence; tree ensembles match or beat deep models on the public datasets (§1) | **new, PRODUCTION** |
| Congestion level, now | explainable rules (`configs/congestion_rules.yaml`) | no independent labels exist; a classifier would imitate the rules (§2) | unchanged |
| Congestion level, future | forecast → the same rules | best on validation for the live domain at every horizon (§2) | unchanged |
| Traffic anomalies | statistical detectors, fused | event recall 0.83–0.92 on incidents that affect traffic; learned detectors are at chance level (§3) | unchanged; learned detectors stay off |
| Incident probability | fused evidence, **UNCALIBRATED** | no labelled incidents in the live domain; the real-video study (§4) does not transfer | label stays "UNCALIBRATED" |
| Risk | explainable weighted engine | no risk labels exist in any dataset | unchanged |

---

## 1. Traffic forecasting

Six experiments, identical protocol:

- chronological train / validation / test periods;
- every model sees the same windows;
- selection by validation RMSE on the primary target;
- deep models are trained with 2–3 seeds, and the best validation seed is kept.

The public datasets answer *which model family* to use. Only the synthetic junction has the live
feature set and resolution, so only its models can serve.

**Test RMSE averaged over +5 / +15 / +30 min** (the model selected on validation, then the
baselines):

| Dataset (experiment) | Target | Selected on validation | RMSE | MAE | R² | Linear | Persistence | Skill vs persistence |
|---|---|---|--:|--:|--:|--:|--:|--:|
| METR-LA, traffic only (`20260925-025205`) | speed, km/h | transformer | 10.68 | 6.24 | 0.785 | 10.99 | 11.63 | 8.2% |
| METR-LA, + calendar & weather (`20260925-020635`) | speed, km/h | xgboost_d8 | 10.10 | 5.96 | 0.809 | 10.89 | 11.63 | 12.8% |
| PEMS-BAY, traffic only (`20260925-021348`) | speed, km/h | gru | 4.88 | 2.36 | 0.881 | 5.21 | 5.39 | 9.1% |
| PEMS-BAY, + calendar & weather (`20260925-023203`) | speed, km/h | random_forest_deep | 4.51 | 2.22 | 0.900 | 5.16 | 5.39 | 14.9% |
| PEMS08 (`20260925-023948`) | flow, veh/h | xgboost_d8 | 240.9 | 148.1 | 0.979 | 283.4 | 320.3 | 23.7% |
| Synthetic junction, live feature set (`20260925-024730`) | speed, km/h | **xgboost_d8** | 3.35 | 1.87 | 0.844 | 3.82 | 5.10 | 32.7% |

Skill = 1 − RMSE / RMSE(persistence). The seasonal-naive baseline (same time yesterday) is worse
than persistence everywhere (skill −0.49 to −1.78).

**By horizon** (the selected model vs persistence, test RMSE):

| Dataset | +5 min | +15 min | +30 min |
|---|---|---|---|
| METR-LA, traffic only | 7.79 vs 8.54 | 10.81 vs 11.70 | 13.43 vs 14.66 |
| METR-LA, + context | 7.61 vs 8.54 | 10.30 vs 11.70 | 12.38 vs 14.66 |
| PEMS-BAY, + context | 2.44 vs 2.75 | 4.68 vs 5.45 | 6.42 vs 7.97 |
| PEMS08 (flow) | 210.6 vs 252.9 | 241.4 vs 312.1 | 270.5 vs 396.0 |
| Synthetic junction | 3.09 vs 4.02 | 3.41 vs 4.97 | 3.57 vs 6.31 |

**What the results show**

- **Every learned model beats persistence**, and the gap grows with the horizon, as expected:
  the next 5 minutes look like now, the next 30 do not.
- **Model families are close; context matters more.**
  - On METR-LA and PEMS-BAY, the best tree ensemble and the best deep model are within 1.5% RMSE
    of each other. On PEMS08 and the synthetic junction, the trees lead by 3–5%.
  - Adding calendar and weather features lowers RMSE by 5.4% (METR-LA) and 7.6% (PEMS-BAY).
  - The hour-of-day figures show where that gain comes from. With context, the model beats
    persistence at the predictable transitions: at +30 min at 19:00 on METR-LA, MAE is about 10.6
    vs 13.6 km/h. Without context, the transformer barely beats persistence in the rush hours
    (`reports/figures/forecasting/<experiment>/error_by_hour_*.png`).
  - The 15:00–17:00 build-up stays hard for every model.
- **Deep models do not pay for their cost here.**
  - On METR-LA and PEMS-BAY they train 5–26× longer than xgboost_d8 (METR-LA transformer: 122 s
    vs 4.7 s).
  - They need PyTorch in the runtime.
  - They were not better once context features were present.

  Every model predicts in about 0.01 ms per sample or less (batched), so inference speed does not
  separate them.
- **Live model.** `xgboost_d8` is selected on validation for the synthetic junction and serves the
  runtime (`configs/pipeline.yaml` names it). The public-data models stay CANDIDATE: they were
  trained on other features, resolutions and places.

**Limits.**

- The live model is trained on **synthetic** traffic. Its accuracy on real cameras is unknown until
  the system records its own history; retraining on those recordings is the intended path.
- METR-LA and PEMS-BAY cover 11 selected sensors each (see the inventory), not the full networks,
  so the numbers are not comparable to published full-network benchmarks.

## 2. Congestion level

The *current* level is defined by the explainable rules, and no dataset has independent congestion
labels, so no classifier is trained for it. The *future* level is trained: the label is the rule
level of the **observed** value at t + h. Five approaches are compared:

- persistence of the current level;
- *regression → rules* (XGBoost forecasts the indicators, then the same rules apply; this is what
  the live service does);
- logistic regression, random forest and XGBoost classifiers.

**Test macro-F1** (the model best on validation marked **bold**):

| Dataset | Horizon | Persistence | Regression → rules | Logistic | Random forest | XGBoost |
|---|---|--:|--:|--:|--:|--:|
| METR-LA | +5 | 0.649 | 0.666 | 0.633 | **0.666** | 0.650 |
| METR-LA | +15 | 0.589 | 0.597 | 0.569 | **0.621** | 0.577 |
| METR-LA | +30 | 0.542 | 0.547 | 0.500 | **0.575** | 0.531 |
| PEMS-BAY | +5 | 0.829 | **0.848** | 0.837 | 0.846 | 0.845 |
| PEMS-BAY | +15 | 0.697 | 0.715 | 0.689 | **0.720** | 0.723 |
| PEMS-BAY | +30 | 0.615 | 0.632 | 0.560 | 0.651 | **0.656** |
| Synthetic junction | +5 | 0.562 | **0.586** | 0.547 | 0.571 | 0.555 |
| Synthetic junction | +15 | 0.495 | **0.549** | 0.510 | 0.523 | 0.513 |
| Synthetic junction | +30 | 0.415 | **0.541** | 0.476 | 0.508 | 0.510 |

**What the results show**

- Macro-F1 is the right summary here, because the classes are very unbalanced (METR-LA test, +15
  min: 50,179 free vs 2,884 high). Accuracy (0.76–0.98) is dominated by the free class.
- **Direct classifiers win on the loop-detector data**, where the labels come from **speed only**:
  - on METR-LA the random forest is best on validation at every horizon (on test: equal at +5,
    and +0.02 to +0.03 macro-F1 over regression → rules at +15 / +30);
  - on PEMS-BAY the classifiers win at +15 / +30.
- **Regression → rules wins on the live feature set** at every horizon. There, the level
  combines speed, density, occupancy and queue, and a classifier must learn that rule from few
  high / severe examples. The live service therefore keeps forecast → rules. The per-horizon
  classifiers are registered as candidates (`models/congestion/level_h{5,15,30}`).
- Confusion matrices: `reports/figures/congestion/<experiment>/confusion_h*.png`.

## 3. Traffic anomaly detection (live feature set, synthetic)

The synthetic junction has injected capacity drops as ground truth: 40 incidents in the test
period, 12 of which occur under enough demand to affect traffic. The learned detectors are fitted on
normal windows of the first 14 days only. Experiment `20260930-014744` (a rerun that also saves the
scores for the figures; its metrics are identical to `20260925-021606`).

| Detector | PR-AUC | ROC-AUC | F1 (window) | Event recall | Recall, incidents with traffic impact | False alarms / h | Median delay |
|---|--:|--:|--:|--:|--:|--:|--:|
| flow imbalance | 0.062 | 0.540 | 0.085 | 0.13 | 0.42 | 0.01 | 2 min |
| fused probability | 0.015 | 0.571 | 0.027 | 0.38 | 0.83 | 0.31 | 2 min |
| density z-score | 0.011 | 0.524 | 0.047 | 0.38 | 0.83 | 0.41 | 2 min |
| queue growth | 0.011 | 0.525 | 0.044 | 0.28 | 0.92 | 0.25 | 2 min |
| sudden slowdown | 0.010 | 0.519 | 0.038 | 0.40 | 0.83 | 0.49 | 3 min |
| autoencoder | 0.009 | 0.512 | 0.004 | 0.05 | 0.17 | 0.03 | 7.5 min |
| z-score | 0.009 | 0.515 | 0.024 | 0.55 | 0.83 | 0.65 | 4 min |
| EWMA | 0.009 | 0.509 | 0.019 | 0.50 | 0.83 | 0.68 | 4 min |
| isolation forest | 0.009 | 0.512 | 0.000 | 0.00 | 0.00 | 0.00 | – |

The base rate (the PR-AUC of a random score) is 0.0085.

**What the results show**

- **Window-level scores are low for every detector, and the score timeline shows why**
  (`reports/figures/anomaly/<experiment>/score_timeline.png`):
  - the label marks only the capacity-drop interval;
  - the detectors respond to its *consequences*, which lag it and outlast it (a queue builds, then
    drains);
  - they also respond to recurrent peak-hour congestion.
- **Event-level results are what matters for an operator.** The speed, queue and density
  detectors, and the fused score, find 83–92% of the incidents that change traffic within 2–4
  minutes, at 0.25–0.7 false alarms per hour. Incidents at low demand do not change traffic and cannot be seen in traffic data.
- **The learned detectors are at chance level.** Isolation Forest and the autoencoder learn
  "unusual feature vectors", and in this data unusual vectors are mostly ordinary peaks. They stay
  **commented out** in `configs/pipeline.yaml`, and the version promoted earlier was **retired**
  (archived), so the registry no longer lists them as production models.
- **Caveat.** This is a queueing simulation. It validates detector behaviour and the evaluation
  machinery, not real-world detection rates. §4 is the real-data check.

## 4. Incident detection on real video (AI City 2021 Track 4)

_In progress: the 100 labelled training videos are being processed (detections and tracks at
1280 px, 5 fps). This section will report the window-level and event-level metrics, the AI City
timing score and the calibration study (Platt / isotonic, Brier score) from that run._

## 5. Vehicle detection (UA-DETRAC)

_The first run is complete (below). A zero-shot evaluation of YOLO11m (the production model) and a
follow-up fine-tune with repeat-factor sampling are queued behind the video extraction, because
only one GPU-heavy job runs at a time. This section will be completed with them._

Test = the official UA-DETRAC test split (40 sequences, every 10th frame, 5,625 images, 63,377
boxes), scored through the production detector adapter. Precision and recall are measured at the
pipeline's confidence threshold of 0.25.

| Model | P | R | mAP50 | mAP50-95 | car | bus | truck | ms / image |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| YOLO11s COCO, zero-shot @960 | 0.850 | 0.858 | 0.781 | 0.551 | 0.618 | 0.605 | 0.428 | 12.6 |
| YOLO11s COCO, zero-shot @1280 | 0.805 | 0.879 | 0.803 | 0.561 | 0.620 | 0.578 | 0.485 | 22.0 |
| YOLO11s fine-tuned @960 | 0.848 | 0.834 | 0.592 | 0.397 | 0.638 | 0.506 | 0.045 | 13.5 |
| YOLO11s fine-tuned @1280 | 0.846 | 0.847 | 0.599 | 0.399 | 0.645 | 0.498 | 0.053 | 23.2 |

Per-class columns are mAP50-95. Fine-tuning took 4,092 s; the best epoch was 10 of 18 (early
stopping).

**What the first run shows**

- Fine-tuning improves **cars** (0.62 → 0.64 mAP50-95) but loses most of **truck** AP (0.43 →
  0.05) and some **bus** AP.
- The training data explains it (`reports/figures/detection/class_distribution.png`):
  - trucks are 0.5% of training boxes but 2.6% of test boxes, and appear in only 233 of 6,669
    training images;
  - buses are 5.6% of training boxes vs 10.4% of test boxes.
- A fine-tuned UA-DETRAC model also cannot detect pedestrians, bicycles or motorcycles (no labels),
  and the pipeline uses those classes. COCO weights therefore stay in production unless a
  fine-tuned model is clearly better on the classes it has.

## 6. Tracking (UA-DETRAC)

Twenty test sequences (28,787 frames, 327,191 ground-truth boxes), CLEAR-MOT at IoU 0.5.
Detections come from the production weights (YOLO11m, 960 px, confidence 0.1) and are cached, so
every tracker sees identical input. Thresholds were chosen by a grid on the 10 validation sequences
(experiment `20260925-023637`).

| Tracker | Setting | Validation MOTA | Test MOTA | Precision | Recall | ID switches | ms / frame |
|---|---|--:|--:|--:|--:|--:|--:|
| ByteTrack | production before tuning (`new_track_thresh` 0.6) | – | 0.737 | 0.943 | 0.785 | 174 | 1.0 |
| ByteTrack | tuned: high 0.5, match 0.8, new 0.4 | 0.834 | **0.749** | 0.936 | 0.804 | 193 | 1.1 |
| BoT-SORT | tuned: high 0.5, new 0.4 | 0.834 | 0.749 | 0.936 | 0.804 | 196 | 35.4 |

**Decision.**

- The tuned `new_track_thresh` of 0.4 generalises from validation to test: +1.2 MOTA points, and
  recall 0.785 → 0.804 (precision 0.943 → 0.936), for 19 more identity switches. It is now set in `configs/pipeline.yaml`.
- BoT-SORT reaches the same MOTA at 33× the time per frame (mostly camera-motion compensation,
  which fixed cameras do not need), so ByteTrack stays.
- The production baseline was scored on the same 20 sequences:
  `experiments/tracking/20260925-023637_tracker_ua_detrac/production_baseline_test.json`.

## 7. What is not trained, and why

| Component | Reason |
|---|---|
| Current congestion level | defined by explainable rules; no independent labels (§2) |
| Risk | no dataset has risk labels; the explainable weighted engine stays |
| Incident probability calibration | no labelled incidents in the live domain. The label stays **"Incident probability: UNCALIBRATED"** in the API and dashboard. |
| Re-identification (CityFlowV2-ReID) | no ReID or multi-camera module consumes it |
| Traffic-light state (LISA) | dashcam view; the system has no signal-phase module |
| Szeged weather | wrong place for every traffic dataset; the context features use Open-Meteo archive weather for each dataset's own location and dates |
