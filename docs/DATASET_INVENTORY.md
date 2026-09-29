# Dataset inventory

Every dataset on disk, what it contains, and which ML task each can legitimately support. The data
was **found by scanning the disk, not taken from a list**. Every number was measured on the files.

```bash
cd backend
python -m traffic_intel.datasets inventory   # -> reports/dataset_inventory.json (format-driven discovery)
python -m traffic_intel.datasets quality     # -> reports/data_quality_report.{json,md}
python -m traffic_intel.datasets status      # manifest entries: present / missing
```

**How discovery works.** `datasets/inventory.py` walks `data/` and `runs/` and groups the files
into units. A unit is either a manifest `local_path` or, for any other folder with data, an
"unregistered" unit. Each unit is inspected by content:

- videos: resolution, fps and duration from each file header;
- images: sampled sizes;
- XML annotations: structure, objects, classes and ids;
- CSV: columns, label values, ranges and duplicates;
- HDF5: tables, time index and step;
- NumPy archives and Parquet metadata;
- pickles: read only with a restricted unpickler that can load NumPy arrays and nothing else,
  so no code from a downloaded file can run;
- SQLite tables and JSON(L).

Scan date: 2026-09-24. Nothing new appeared in Downloads after the datasets were moved into
`data/raw/`.

---

## 1. Summary

| Dataset | Type | Samples | Labels | Features | Temporal | Spatial | Potential task |
|---|---|--:|---|---|---|---|---|
| **UA-DETRAC** `raw/uadetrac` | traffic-camera image sequences + XML | 100 sequences, 140,131 frames (960×540); 1,274,055 boxes, 8,289 track ids | vehicle_type {car, van, bus, others} per box, track id, occlusion / truncation, ignored regions, weather per sequence | pixels | 25 fps sequences; no clock time | image coordinates only (no calibration) | vehicle **detection**; single-camera **tracking** |
| **AI City 2021 Track 4** `raw/anomaly/aic21_track4` | freeway surveillance video | 100 train videos (24.5 h) + 150 test videos (37.1 h); 800×410 | 18 anomaly intervals `video start_s end_s` in 17 train videos; test labels not public | pixels | ~30 fps (2 train videos 14.3 / 23.56 fps); seconds from video start | none (uncalibrated, no GPS) | **incident / anomaly evaluation** on real video |
| **METR-LA** `raw/metr_la` | loop-detector time series | 207 sensors × 34,272 steps (7.09 M readings) | none | speed (mph) only | 5 min, 2012-03-01 .. 2012-06-27, local LA time | sensor graph (207×207 adjacency, 1,722 edges); no coordinates on disk | speed **forecasting**; congestion-level **forecast classification** |
| **PEMS-BAY** `raw/pems_bay` | loop-detector time series | 325 sensors × 52,116 steps (16.9 M readings) | none | speed (mph) only | 5 min, 2017-01-01 .. 2017-06-30 | sensor lat/lon, freeway, direction, lanes (meta); 325×325 adjacency | speed **forecasting**; congestion-level **forecast classification** |
| **PEMS08** `external/pems08` | detector-station time series | 170 stations × 17,856 steps | none | flow, occupancy, speed | 5 min, 2016-07-01 .. 2016-08-31 | none | multi-target **forecasting** |
| **AI City 2021 Track 2 (CityFlowV2-ReID)** `raw/aicity_reid/aic21_track2` | vehicle crops | 85,058 images (52,717 train / 31,238 test / 1,103 query); 440 train identities, 40 cameras | vehicleID + cameraID (train); cameraID only (test / query) | pixels (crops, variable size) | none | camera id only | vehicle **re-identification** |
| **LISA Traffic Light** `raw/traffic_lights/lisa` | in-vehicle (dashcam) image sequences | 44,075 frames (1280×960); 112,370 box annotations (BOX files) | go, stop, stopLeft, warning, goLeft, warningLeft, goForward | pixels | frame numbers of day / night clips | none | traffic-light detection / state classification |
| **Szeged weather** `raw/weather/kaggle_szeged` | hourly weather table | 96,453 rows | none | temperature, apparent temperature, humidity, wind, visibility, pressure, summary | hourly, 2006-01-01 .. 2016-12-31 (Europe/Budapest) | one city (Szeged, Hungary) | weather context |
| **Open-Meteo cache** `external/weather` (unregistered path) | hourly weather (Parquet) | 2,544 rows in 2 files | none | temperature, humidity, rain, condition (visibility 100% missing) | 2016-07..09 (PEMS08 site), 2025-06..07 (demo site) | 2 lat/lon points | weather context features |
| **Synthetic junction series** (generated in memory) | simulated 1-min windows | 42 days × 4 approaches (default) | injected incidents (`gt_incident`) | the live feature set: count, speed, density, occupancy, flow, queue, stopped | 1 min | demo network | live-domain forecasting / anomaly |
| **Synthetic scenes** `synthetic/` | rendered video + ground truth | 1 scene (180 s, 1280×720, 10 fps), truth.csv 18,622 rows | per-object truth: id, class, metric position, speed | pixels + metric truth | 10 fps | metric ground plane | perception / speed **validation** |
| **Public demo clips** `raw/public` | short videos | 3 clips (30 s, 54 s, 21.5 s) | none | pixels | 12–25 fps | none | end-to-end **validation** only |
| **Own pipeline outputs** `runs/`, `interim/`, `db/` | 1-min windows, tracks, Parquet detections | 68 window records in 32 `traffic_states.jsonl`; 15 Parquet files (8,430 rows); database: network only, 0 windows | pipeline anomaly signals | canonical traffic features | 1 min | demo network | too small to train; validation only |
| **AI City CityFlowV2 Track 1** `raw/aicity` | — | **not on disk** | — | — | — | — | (multi-camera tracking, calibration) |

---

## 2. Dataset cards

### UA-DETRAC: detection and tracking
- **Path, format:**
  - `raw/uadetrac/DETRAC-Images/MVI_*/img*.jpg` (100 folders, 960×540);
  - `DETRAC-Train-Annotations-XML/` (60 files) and `DETRAC-Test-Annotations-XML/` (40 files);
  - `DETRAC-MOT-toolkit/` (MATLAB evaluation code, 808 files).
- **Samples:**

  | Split | Sequences | Annotated frames | Image files | Boxes | Track ids |
  |---|--:|--:|--:|--:|--:|
  | train | 60 | 82,085 | 83,791 | 598,281 | 5,952 |
  | test | 40 | 56,167 | 56,340 | 675,774 | 2,337 |

- **Labels:**
  - `vehicle_type`: car 1,052,408, van 95,570, bus 105,436, others 20,641;
  - occlusion / truncation flags;
  - 407 ignored-region boxes;
  - weather (sunny 23, cloudy 30, night 28, rainy 19 sequences);
  - camera state (stable 53, unstable 47).
- **Mapping to project classes** (manifest): van → car, others → truck. There are no
  pedestrian, bicycle or motorcycle labels.
- **Quality:**
  - class imbalance after mapping: train car 93.8%, bus 5.6%, truck 0.6%;
  - 1,706 train / 173 test images have no annotation entry, so they are excluded rather than
    assumed empty;
  - the same object has median IoU 0.85 (train) / 0.94 (test) between consecutive frames, so
    adjacent frames are near-duplicates;
  - 24,940 train boxes are centred inside ignored regions;
  - no invalid boxes, and no sequence appears in both splits.
- **Preprocessing:**
  - keep the official split;
  - make validation from whole train sequences;
  - subsample frames;
  - grey out ignored regions and drop labels inside them;
  - convert boxes to YOLO txt.
- **Limitations:** cameras in two Chinese cities (2015), mostly elevated views; "others" is a
  mixed class; no metric calibration, so no speed ground truth.

### AI City 2021 Track 4: incident / anomaly
- **Path, format:** `aic21-track4-train-data/*.mp4` (100) and `aic21-track4-test-data/*.mp4`
  (150), all 800×410; `train-anomaly-results.txt` (18 lines, `video start_s end_s`); `license.txt`
  (CC BY 4.0) plus the challenge licence agreement.
- **Labels:** one label type, "anomaly", covering stalled vehicles and crashes alike. It is never
  renamed "accident".
  - 17 of the 100 training videos are labelled; video 33 has two intervals.
  - 5 intervals start at 0 s, meaning the anomaly is already present when the video starts.
  - Anomalous time is 2.7 h of 24.5 h (10.9%).
- **Quality:**
  - every interval is valid against its video's duration, and no intervals overlap;
  - two training videos are not at 30 fps, so times are converted with each video's own fps;
  - the 150 test videos have no public labels.
- **Preprocessing:** split by video. The cameras are uncalibrated, so the perception pipeline runs
  in perception-only mode (detection + tracking in pixels) and window features are computed in
  scale-normalised pixel units.
- **Limitations:**
  - with only 18 events, metrics have wide confidence intervals;
  - no calibration, so the metric features and track detectors of the live system cannot be
    computed here;
  - freeway-only views.

### METR-LA: forecasting
- **Path, format:** `METR-LA.h5`, a pandas "fixed" table from pandas 0.15 / Python 2. It is read
  by the adapter's legacy reader; its pickled attributes are never unpickled. `adj_METR-LA.pkl`
  holds sensor ids, an id→index map and a 207×207 weighted adjacency.
- **Values:** speed 0–70 mph, where 0 means outage. 572,818 zeros are treated as missing, leaving
  8.08% missing overall; the worst sensor is missing 20.1%.
- **Time:** 5-min steps with no irregular steps. 12 local timestamps that don't exist (the
  2012-03-11 DST jump) are dropped. The longest outage per sensor has a median of 381 steps
  (≈32 h).
- **Spatial:** only the graph. The optional coordinates file is not on disk.
- **Limitations:** speed only (no flow or occupancy); LA freeways in 2012.

### PEMS-BAY: forecasting
- **Path, format:** `pems-bay.h5` (key `speed`, 325×52,116, legacy format), `pems-bay-meta.h5`
  (325 sensors: freeway, direction, lanes 1–6, lat 37.25–37.43, lon −122.08 – −121.84),
  `adj_mx_bay.pkl` (325×325, 2,694 edges).
- **Quality:**
  - 521 zeros (<0.01%), and the longest gap is 1 step;
  - 0.70% of readings sit in identical-value runs of 1 h or more;
  - the one-hour hole at the 2017 DST change is absent from the source.

### PEMS08: forecasting
- **Path, format:** `external/pems08/pems08.npz`, array `data` of shape (17,856, 170, 3):
  flow per 5 min 0–1,147, occupancy 0–0.90, speed 3–82 mph.
- **Quality:** 0.27% missing (the worst station 3.4%), all values in physical range. The canonical
  subset is in `processed/pems08`.

### CityFlowV2-ReID: vehicle re-identification
- **Labels:** 440 training identities; 19–682 images per identity (median 92), seen by a median
  of 4 cameras each. Test and query labels carry only the camera id; identities are withheld.
- **Why it is not used for training now:**
  - the pipeline has no re-identification or multi-camera component, so an embedding model would
    have no consumer;
  - BoT-SORT here uses motion and camera-motion compensation, not appearance.
  - It is kept for future cross-camera matching. A held-out evaluation would have to split the
    training identities.

### LISA Traffic Light: signal state
- **Labels:** 112,370 boxes in `frameAnnotationsBOX.csv` files (plus BULB files):
  - tag counts: go 48,892, stop 44,730, stopLeft 13,048, warning 2,669, goLeft 2,476,
    warningLeft 350, goForward 205;
  - all referenced frames exist, and no boxes are invalid;
  - 2,911 duplicate annotation rows.
- **Why it is not used:**
  - the camera is mounted in a car, not a fixed traffic camera;
  - the system has no traffic-light class and no signal-phase module that would consume a
    trained model.
  - It is kept for a possible signal-phase feature.

### Szeged weather: context
- **Quality:**
  - 24 duplicate rows;
  - 1,288 pressure readings outside 850–1090 mbar (mostly 0);
  - 22 humidity readings of 0;
  - `Loud Cover` is constant;
  - 517 missing `Precip Type`.
- **Why it is not used:** weather is only a valid feature when it matches the traffic data's
  place **and** time. No traffic dataset is in Hungary, and the METR-LA overlap is in time only.
  Using it would attach the wrong weather to every sample. The forecasting experiments instead
  fetch Open-Meteo archive weather for each dataset's own location and dates (CC BY 4.0, small,
  cached).

### Open-Meteo cache
- **Path:** `external/weather/*.parquet`, one file per location and date range, created by
  `forecasting/data/context.py` on first use.
- **Quality:** `visibility_km` is 100% missing, because the archive API doesn't provide it; the
  preprocessor drops that column at fit time.

### Synthetic data (project-generated)
- **Junction series:** 1-min windows with **the same feature set the live pipeline produces**
  (count, speed, density, occupancy, flow, queue, stopped) and injected incidents. It is the only
  labelled data in the live system's own feature space and resolution. It is labelled "synthetic"
  wherever it is used.
- **Scene (`synthetic/default`):** a rendered video with metric ground truth per object, used to
  validate tracking and speed estimation.
- **`synthetic/cache/*.pkl`:** a cache of generated runtime scenes (project classes). The
  inventory does not load it.

### Own pipeline outputs
- **`runs/`:** 68 window records in 32 `traffic_states.jsonl` files, plus tracks and anomaly
  logs.
- **`interim/public_highway_01`:** 15 Parquet files of detections / tracks / track summaries
  (8,430 rows).
- **`db/traffic.db`:** the network only (0 traffic windows at scan time).
- **Decision:** far too little to train on. Used for end-to-end validation only.

---

## 3. Dataset → task mapping

| Dataset | Task(s) it supports | Used? | Reason |
|---|---|---|---|
| UA-DETRAC | vehicle detection (car / bus / truck); single-camera tracking (track ids) | **yes**: detector fine-tuning and tracker comparison | only labelled traffic-camera detection / MOT data on disk; official train / test split by sequence |
| AI City 2021 Track 4 | anomaly / incident detection **evaluation** (real labels) | **yes**: anomaly evaluation on real video, split by video | only real labelled incident data. Uncalibrated cameras mean its features are pixel-space, so the models trained on it are evaluated, **not deployed** (the live detectors use metric features) |
| METR-LA | speed forecasting; congestion-level forecasting (future level from observed future speed) | **yes** | long real series; chronological split |
| PEMS-BAY | same as METR-LA | **yes** | same |
| PEMS08 | flow / occupancy / speed forecasting | **yes** | the only real set with flow and occupancy |
| Open-Meteo archive | weather context for the time-series sets | **yes**: context variants | matches each dataset's place and time |
| Synthetic junction series | forecasting, congestion level and anomaly in the **live** feature space | **yes**: models deployed in the live system | the live pipeline's features and 1-min resolution; injected incident labels |
| Synthetic scene | perception / speed validation | validation only | already used by tests |
| CityFlowV2-ReID | re-identification | **no** | no consumer in the system (no ReID or multi-camera module) |
| LISA Traffic Light | traffic-light state | **no** | dashcam view; no signal-phase module or class |
| Szeged weather | weather context | **no** | wrong place for every traffic dataset |
| METR-LA / PEMS-BAY adjacency | graph models | **no** (for now) | the model families compared are per-sensor; kept for future graph models |
| Public clips, own outputs | end-to-end validation | validation only | unlabelled / too small |
| AI City CityFlowV2 Track 1 | multi-camera tracking, calibration | **not on disk** | would be needed for metric calibration experiments |

### Tasks and what the data allows

| Task | Data | Training possible? |
|---|---|---|
| Object detection | UA-DETRAC | yes: fine-tune YOLO11 (transfer learning), evaluate on the official test split |
| Object tracking | UA-DETRAC track ids | trackers are not learned: compare ByteTrack vs BoT-SORT and tune thresholds on validation sequences |
| Traffic-state estimation | camera ground truth for counts / speed would be needed | no labelled real data. Metric states come from calibrated geometry; validated on the synthetic scene |
| Congestion (current state) | levels are *defined* by `configs/congestion_rules.yaml`; no independent labels exist | **no**: a classifier trained on rule labels would only imitate the rules |
| Congestion (future level) | METR-LA, PEMS-BAY, synthetic series | **yes**: label = rule level of the *observed* future speed, so the target is not a function of the inputs |
| Forecasting (+5/+15/+30 min) | METR-LA, PEMS-BAY, PEMS08, synthetic | **yes** |
| Anomaly detection | AIC21 Track 4 (real, evaluation); synthetic series (live features) | **yes** |
| Incident detection | AIC21 labels ("anomaly" only) | evaluated with the AIC21 timing metric; no incident sub-types exist in any label set |
| Risk | no dataset carries risk labels | **no**: the explainable risk engine is kept |
| Calibration of the incident score | AIC21 intervals (for the video detector's score only) | investigated. The live fused score stays **UNCALIBRATED** |
