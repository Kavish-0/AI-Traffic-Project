# Dataset strategy

This project does **not** rely on one dataset for every task. Each component is developed and
evaluated on a dataset suited to its task. The components are then connected through **one
canonical representation**: the project's own types and `core/feature_schema.py`.

> The integrated system combines *outputs* of components that were trained and evaluated on
> different datasets. It never pretends those outputs came from one dataset. Every experiment
> record states which dataset it used.

- **Machine-readable source of truth:** [`configs/datasets/manifest.yaml`](../configs/datasets/manifest.yaml)
- **Code:** `backend/traffic_intel/datasets/`

---

## 1. Datasets and their roles

| Task | Dataset | Manifest id | Access | Status in this repository |
|---|---|---|---|---|
| Vehicle detection, classification, tracking | **UA-DETRAC** | `ua_detrac` | registration; non-commercial research | **present** (60 train + 40 test sequences, 140k frames); adapter + tests |
| Multi-camera / intersection tracking, camera calibration | **AI City Challenge CityFlowV2 (Track 1)** | `aicity_mtmc` | request + data licence agreement | adapter + tests; **still missing** (not the same as the Track 2 ReID data below) |
| Traffic anomalies / incidents | **AI City Challenge 2021 Track 4** | `aicity21_anomaly` | request + data licence agreement (release `license.txt`: CC BY 4.0) | **present** (100 labelled train videos, 18 intervals; 150 unlabelled test videos kept apart) |
| Forecasting (speed) | **METR-LA** | `metr_la` | open (DCRNN release), no explicit data licence | **present** (207 sensors, 34,272 steps); legacy HDF5 read by the adapter |
| Forecasting (speed) | **PEMS-BAY** | `pems_bay` | open (DCRNN release), no explicit data licence | **present** (325 sensors, 52,116 steps) |
| Forecasting (flow, occupancy, speed) | PeMS District 8 (PEMS08) | `pems08` | open, small, checksum-verified script | **present**; used in Prompt 3 results |
| Context | Open-Meteo historical weather | `open_meteo_weather` | open, CC BY 4.0 | fetched on demand and cached (`data/raw/weather/`) |
| Pipeline validation | Synthetic junction series and scenes | `synthetic_junction` | project-generated | generated in memory |
| Final demonstration | Own camera recordings | `own_recordings` | private | future; never required by the public-data experiments |
| *(stored, not used by E1-E6)* Vehicle re-identification | AI City Challenge 2021 Track 2 | `aicity21_reid` | request + data licence agreement | **present**; kept for future multi-camera work |
| *(stored, not used by E1-E6)* Traffic-light state | LISA Traffic Light Dataset | `lisa_traffic_lights` | registration; non-commercial (*verify*) | **present**; kept for a possible signal-phase module |
| *(stored, not used by E1-E6)* Weather | Szeged (Hungary) hourly weather 2006-2016 | `weather_history_szeged` | Kaggle (*verify*) | **present**; deliberately **not joined** to METR-LA / PEMS-BAY (different place and years) |

**Nothing is downloaded automatically.** Large or licence-restricted data is placed by hand under
`data/raw/<dataset>/`. The provider's terms always prevail over the summaries here. Items marked
*verify* in the manifest are details that could not be confirmed from the provider while writing
this: the exact label-file name of AIC21 Track 4, the homography direction of CityFlow
calibration files, and UA-DETRAC's current licence wording. The adapters take these as parameters
instead of hard-coding them.

---

## 2. Dataset cards

### 2.1 UA-DETRAC (`ua_detrac`): perception

| | |
|---|---|
| **Purpose** | Vehicle detection and class mapping (E1), single-camera tracking (E2) |
| **Size** | about 10 h of video, 100 sequences (60 train / 40 test), 25 fps, 960×540; about 140k frames, 1.21M boxes |
| **Content** | Per-frame boxes with track ids and `vehicle_type` ∈ {car, van, bus, others}. Ignored regions per sequence. Weather and camera attributes. |
| **Class mapping** | car → car, van → car, bus → bus, others → truck (an approximation, documented). **No motorcycle, bicycle or pedestrian labels.** The mapping goes to the project's `ObjectClass`, not to YOLO/COCO names, and lives in the manifest (`classes.class_map`). Unknown labels are counted, never guessed. |
| **Canonical output** | `AnnotatedFrame` = project `Tracks` (id, xyxy, ObjectClass id) + ignored regions. `prepare` writes `data/annotations/ua_detrac/<seq>.csv` and a precomputed-detector table `data/detections/ua_detrac/<seq>_gt.csv`, so ground-truth boxes can run through the real tracker, trajectory store and feature engine. |
| **Train / val / test** | Official train/test sequences. Validation = held-out *train* sequences. Always split by sequence, never by frame. |
| **Limitations** | Chinese urban roads, 2015 cameras, mostly elevated views. No two-wheelers. "others" is heterogeneous. No metric calibration, so no speed ground truth. |
| **Licence / access** | Research use, non-commercial; reported as CC BY-NC-SA 3.0 (*verify*). Registration. No redistribution. |
| **Citation** | Wen et al., *UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking*, CVIU 193 (2020) 102907; arXiv:1511.04136. |

### 2.2 AI City Challenge, CityFlowV2 (`aicity_mtmc`): multi-camera, intersections, calibration

| | |
|---|---|
| **Purpose** | Tracking on real traffic-camera viewpoints (E2), multi-camera metadata, geo-referencing and homography (E3) |
| **Size** | 46 cameras in scenarios S01–S06 (intersections and road segments), about 3.6 h, 10 fps |
| **Content** | `gt/gt.txt` (MOTChallenge: frame, id, box). `calibration.txt` (homography between image and GPS). Per-scenario camera time offsets. ROI masks. |
| **Class mapping** | The ground truth has vehicle identities without classes → `vehicle → car` (configurable placeholder) |
| **Canonical output** | `AnnotatedFrame` per camera. `calibration_config()` converts the GPS homography into the project's `CalibrationConfig` (pixel ↔ metres east/north of a GPS origin). The existing `HomographyProjector` therefore gives metric positions, speeds and lon/lat for real AI City cameras. A test checks the projection against the raw homography (< 0.25 m) and the metric scale. |
| **Train / val / test** | Official train/validation scenarios; split by scenario/camera |
| **Limitations** | No vehicle classes. The homography direction and GPS order must be checked per release (`homography_direction`, `gps_order`). Lens distortion is ignored for fisheye cameras. |
| **Licence / access** | AI City data licence agreement, by request. Non-commercial research. No redistribution. |
| **Citation** | Tang et al., *CityFlow*, CVPR 2019; Naphade et al., *The 5th AI City Challenge*, CVPRW 2021. |

### 2.3 AI City Challenge 2021 Track 4 (`aicity21_anomaly`): incidents / anomalies

| | |
|---|---|
| **Purpose** | Timing evaluation of stopped-vehicle and possible-incident detection (E4) |
| **Size** | 100 labelled training videos and 150 test videos (test labels not public); about 15 min each, 800×410, 30 fps; freeway cameras |
| **Labels** | Anomaly intervals `video_id start_s end_s`. Stalled vehicles **and** crashes share one label. |
| **Terminology** | The canonical record keeps the source label verbatim (`anomaly`); it is **never renamed "accident"**. The system keeps its own taxonomy (NORMAL, SUDDEN_SLOWDOWN, STOPPED_VEHICLE, WRONG_WAY, ABNORMAL_QUEUE, POSSIBLE_INCIDENT) and its incident probability. For evaluation, any non-normal system output at or after the labelled start counts as a detection (`labels.positive_system_types`). |
| **Canonical output** | `IncidentAnnotation(dataset_id, video_id, start_s, end_s, source_label)` at **interval level**; frame indices only on demand (`frame_interval(fps)`). `prepare` writes `data/annotations/aicity21_anomaly/incidents.csv`. |
| **Metric** | `ai/evaluation/incidents.py`: a re-implementation of the published challenge score. TP = within 10 s of the start; F1; RMSE of the detection time; NRMSE capped at 300 s; S4 = F1 × (1 − NRMSE). Not the official evaluation server. |
| **Train / val / test** | Only the 100 training videos are labelled, so split **by video** (for example 60/20/20). Never split by time within a video. |
| **Limitations** | Freeway only (no junctions). Low resolution. Labels do not separate crash from stall. Our camera calibration for these videos is unknown, so speeds are pixel-based unless calibrated. |
| **Licence / access** | AI City data licence agreement, by request. Non-commercial research. No redistribution. |
| **Citation** | Naphade et al., *The 5th AI City Challenge*, CVPRW 2021. |

### 2.4 METR-LA (`metr_la`) and PEMS-BAY (`pems_bay`): forecasting

| | METR-LA | PEMS-BAY |
|---|---|---|
| **Purpose** | +5/+15/+30 min forecasting benchmark (E5) | same |
| **Size** | 207 loop detectors, LA County highways, 2012-03-01 … 2012-06-30, 5-min, 34,272 steps | 325 sensors, SF Bay Area, 2017-01-01 … 2017-05-31, 5-min, 52,116 steps |
| **Features** | speed only (mph; 0 = missing) | speed only (mph) |
| **Canonical output** | `series_id, timestamp (UTC), avg_speed_kmh`: mph → km/h, 0 → NaN, local wall-clock → UTC (the non-existent DST hour is dropped and counted), sensor subset selectable. Validated against `feature_schema`. | same (`pems_bay_<sensor>`) |
| **Not available** | flow, occupancy, density, queue, counts. Models on these datasets forecast **speed only**. | same |
| **Train / val / test** | chronological 70/10/20 (DCRNN convention), fit-on-train scaling, purge gap between splits, identical windows for every model, **never shuffled** | same |
| **Limitations** | Freeway speeds, US, 2012. No incidents labelled. No weather in the file (optional external context). | Freeway, 2017, very regular traffic (high baselines) |
| **Licence / access** | Released with DCRNN (code MIT). No explicit data licence. Research use with citation; do not re-host. | Caltrans PeMS data via DCRNN; research use with citation |
| **Citation** | Li et al., *DCRNN*, ICLR 2018; Jagadish et al., CACM 2014 | Li et al., ICLR 2018; Chen et al., TRR 1748 (2001) |

### 2.5 Context: weather and calendar

- **Calendar features** (hour, day of week, weekend, holiday, rush hour) come from timestamps in
  **local time**, so inference needs nothing but a clock.
- **Weather** (temperature, humidity, rain, visibility, condition) comes from the Open-Meteo archive
  (CC BY 4.0) for the dataset's location and period, and is cached in `data/raw/weather/`.
  Visibility is not provided by the archive, so the column is empty and dropped at fit.
- **Weather is never mandatory.** Each forecasting benchmark has two configs:
  - `forecasting_<dataset>_traffic_only.yaml`: no calendar, no weather;
  - `forecasting_<dataset>_context.yaml`: calendar + weather.

  They use the same sensors, split, windows and models, so the comparison isolates the effect of
  context. An integration test runs both variants on a METR-LA-format fixture. It checks that the
  saved preprocessing differs only in the context columns.

---

## 3. Which dataset for which experiment (never mixed)

| Experiment | Question | Dataset(s) | Metrics |
|---|---|---|---|
| **E1 Perception / detection** | How well does the detector find and classify vehicles (after class mapping)? | UA-DETRAC | AP50 per canonical class, precision, recall (`ai/evaluation/detection.py`) |
| **E2 Tracking** | How stable are track identities? | UA-DETRAC, AI City CityFlowV2 | MOTA, MOTP, ID switches (`MOTEvaluator`) |
| **E3 Calibration / geo** | Do real calibrations give plausible metric positions and speeds? | AI City CityFlowV2 | projection consistency, speed plausibility |
| **E4 Incident / anomaly** | Are stopped vehicles and possible incidents reported in time, with few false alarms? | AIC21 Track 4 (+ synthetic scenarios, Prompt 4) | F1 (±10 s), detection-time RMSE, S4, false alarms |
| **E5 Forecasting** | Do LR / RF / XGBoost / LSTM / GRU / Transformer beat the baselines at +5/+15/+30 min? | METR-LA, PEMS-BAY, PEMS08 (+ synthetic) | MAE, RMSE, R² (chronological test period) |
| **E6 Integrated demonstration** | Does the whole chain work on real footage? | own recordings | qualitative + component metrics |

Rules:

1. A model is evaluated only on held-out data **of the dataset it was trained on**.
2. Cross-dataset use, for example a detector trained on UA-DETRAC running on AI City video, is
   reported as a **transfer** result, never as in-domain.
3. The integrated demo (E6) reports, for every component, which dataset its model came from.
4. The anomaly benchmark is never used to tune forecasting, and vice versa.

---

## 4. Canonical representation (one schema)

Every adapter produces one of three canonical forms. All three are built from types that already
exist in the pipeline.

| Canonical form | Built from | Consumed by |
|---|---|---|
| `AnnotatedFrame` (ground truth per frame) | `core.types.Tracks` / `Detections`, `core.enums.ObjectClass` ids | detector and tracker evaluation; `ground_truth_tracked_frames()` → `TrackedFrame` → trajectory store / traffic feature engine; precomputed-detector CSV |
| `TimeSeriesDataset` (long frame) | `series_id, timestamp (UTC)` + columns of `core/feature_schema.py` (`TRAFFIC_COLUMNS`, `CALENDAR_COLUMNS`, `WEATHER_COLUMNS`) | forecasting preprocessor, windows, models, `ForecastingService` |
| `IncidentAnnotation` (interval) | dataset, video, start/end seconds, source label | incident timing evaluation |

About the feature schema:

- `core/feature_schema.py` is the **single** feature definition. This work removed a drift that
  existed before it: the forecasting layer had its own weather and calendar names. It now imports
  them from the core schema.
- `validate_frame()` rejects any column that is not in the schema. For example, a raw `speed_mph`
  that an adapter forgot to convert is an error.
- The training `FeatureSpec` rejects dynamic features that are not `TRAFFIC_COLUMNS`.
- The camera pipeline's own 1-min windows use the same names. A model trained on a public
  dataset can therefore be served on our own data *if* that data has the same columns and
  resolution. The forecasting service refuses a model whose fitted preprocessing does not match.

---

## 5. Own data (future) - no public experiment depends on it

```
our video → OpenCV → YOLO → ByteTrack → trajectories → traffic features → 1-min windows
          → runs/<camera>/<run>/traffic_states.jsonl  (+ anomalies.jsonl, Parquet in data/interim/)
          → own_recordings adapter → the same TimeSeriesDataset as METR-LA / PEMS-BAY
```

Raw video goes to `data/recordings/<camera>/`. Faces and number plates are personal data: keep
raw video private and share only aggregated windows.

---

## 6. Data directory

```
data/
  raw/          (placed by hand / cached)
    uadetrac/        DETRAC-Images/MVI_*/img*.jpg, DETRAC-Train-Annotations-XML/, DETRAC-Test-Annotations-XML/,
                     DETRAC-MOT-toolkit/
    anomaly/aic21_track4/   aic21-track4-train-data/*.mp4 (labelled), aic21-track4-test-data/*.mp4 (unlabelled),
                            train-anomaly-results.txt, ReadMe.txt, license.txt, DataLicenseAgreement_*.pdf
    aicity/          CityFlowV2 Track 1 (missing)
    aicity_reid/aic21_track2/   image_train|query|test/, *_label.xml, *_track.txt, tools/
    metr_la/         METR-LA.h5, adj_METR-LA.pkl
    pems_bay/        pems-bay.h5, pems-bay-meta.h5, adj_mx_bay.pkl
    traffic_lights/lisa/        Annotations/, dayTrain/, nightTrain/, day|nightSequence1-2/, sample-*/
    weather/         Open-Meteo cache; kaggle_szeged/weatherHistory.csv
    public/          short public demo videos (configs/datasets/public_videos.yaml)
  interim/      per-camera Parquet from the pipeline (detections, tracks)
  processed/    canonical time series (prepare): <dataset>/series.parquet + meta.json
  annotations/  canonical ground truth (prepare): per-sequence tracks, incident intervals
  features/     feature tables for learned models
  predictions/  model outputs kept for evaluation
  detections/   precomputed-detector tables (ground truth or detector output)
  tracks/       tracker outputs for evaluation
  recordings/   own camera video
  external/     small scripted downloads (PEMS08), legacy weather cache
```

- Nothing under `data/` is committed except folder READMEs and `.gitkeep`.
- Machine-specific dataset locations go in `configs/datasets/local.yaml` (git-ignored), for example
  `paths: {ua_detrac: D:/datasets/UA-DETRAC}`.
- The data root itself can be moved with `TI_DATA_DIR`.

---

## 7. Commands

```bash
cd backend
pip install -e ".[datasets]"                                # PyTables for METR-LA / PEMS-BAY HDF5
python -m traffic_intel.datasets status                      # all datasets: present / missing
python -m traffic_intel.datasets status metr_la              # what is missing + where to get it + licence
python -m traffic_intel.datasets info aicity21_anomaly       # full card
python -m traffic_intel.datasets experiments                 # dataset -> experiment matrix
python -m traffic_intel.datasets layout                      # create data/ folders
python -m traffic_intel.datasets prepare metr_la --every-k 20
python -m traffic_intel.datasets prepare ua_detrac --max-sequences 5
python -m traffic_intel.ai.training train forecasting --config ../configs/training/forecasting_metr_la_traffic_only.yaml
python -m traffic_intel.ai.training train forecasting --config ../configs/training/forecasting_metr_la_context.yaml
```

A missing dataset fails with a message that names the missing files, the expected location, the
source, the licence/access terms and the download steps. No partial run starts.
