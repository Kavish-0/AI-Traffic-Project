# AI Urban Traffic & Incident Intelligence: Project Report

**Scope of this report:** datasets, models, and model inputs and outputs of the
implemented system: architecture, perception and traffic-state estimation (phases 0–2),
forecasting (phase 5, summarised in §4.7) and incident intelligence (phase 6, summarised in §4.8).
**Status:** 24 September 2026.
**Companion documents:** [ARCHITECTURE.md](ARCHITECTURE.md) (system design),
[TRAFFIC_STATE_ESTIMATION.md](TRAFFIC_STATE_ESTIMATION.md) (metric definitions
and calibration), [PHASE1_PERCEPTION.md](PHASE1_PERCEPTION.md) (GPU
perception on public footage), [FORECASTING.md](FORECASTING.md) (forecasting models and
results), [INCIDENT_INTELLIGENCE.md](INCIDENT_INTELLIGENCE.md) (anomaly detection, incident
fusion, risk, what-if).

---

## Abstract

The project builds a system that turns traffic-camera video into structured,
explainable traffic information. It is designed to answer three questions:
*what is happening now*, *what will happen next*, and *is something
abnormal*. This report covers the implemented first stage, which answers
"what is happening now".

Video frames pass through the following stages:

1. **Detection:** a COCO-pretrained YOLO11 detector.
2. **Tracking:** a ByteTrack multi-object tracker, implemented natively.
3. **Projection:** a planar homography maps image pixels to road-plane metres.
4. **Traffic features:** a feature engine computes counts, space-mean speed,
   density, occupancy, flow, queue length and stopped vehicles over 60 s
   windows.
5. **Congestion level:** an explainable rule-based classifier assigns FREE,
   MODERATE, HIGH or SEVERE.

The system was evaluated two ways:

- **Synthetic dataset.** A seeded microsimulation provides exact ground truth.
  Presence counts and flow were exact in free and signalised traffic. Mean
  speed was within 0.7 km/h and queue length within 1.5 m in eight of nine
  60 s windows.
- **Public 4K motorway footage.** Processed on an NVIDIA RTX 4060 Laptop GPU at
  11.8–13.7 frames/s end to end. No ground-truth labels exist for this
  footage, so detection and tracking accuracy on real video remain to be
  measured.

The perception models are used pre-trained. Two later layers are now implemented:

- **Forecasting (§4.7).** Traffic is forecast +5/+15/+30 min with baselines, linear
  regression, Random Forest, XGBoost, LSTM/GRU and a Transformer, on chronological splits. On
  real PeMS08 detector data, XGBoost had the lowest error, and the deep models did not beat it.
- **Incident intelligence (§4.8).** Anomaly signals from several detectors are fused into
  *hedged* incident probabilities (never "accident detected"). An explainable risk score and a
  static what-if network model are added. Six synthetic incident scenarios, run through the real
  pipeline, are classified as intended.

---

## 1. Introduction

### 1.1 Objective

The system takes traffic camera feeds plus context data (weather, calendar)
and produces, for each monitored road approach:

1. **Current state:** vehicle counts by class, speed, density, occupancy,
   flow, queue and a congestion level. *(Implemented.)*
2. **Short-term forecast** at +5, +15 and +30 min. *(Designed; not implemented.)*
3. **Anomaly and incident likelihood.** *(Designed; not implemented.)*

### 1.2 What is reported here

| Phase | Content | Status |
|---|---|---|
| 0 | Architecture, interfaces, schemas, configuration, database schema, API skeleton | Complete |
| 1 | Perception: video input, detection, tracking, per-camera worker, Parquet recording, annotated video, GPU inference | Complete, and run on public footage |
| 2 | Traffic-state estimation: calibration, trajectories, traffic features, rule-based congestion classification | Complete, and validated on synthetic data |
| 3–9 | Persistence and live API, map, forecasting, anomaly/incident, risk, what-if simulation | Not started |

### 1.3 Design principles relevant to the models

- **Replaceable models.** Every model sits behind an interface (`Detector`,
  `Tracker`, `GroundProjector`, `CongestionClassifier`, …) and is selected in
  `configs/pipeline.yaml`. For example, YOLO can be replaced without changing
  downstream code.
- **Metric units, not pixels.** Speeds and distances are computed on the road
  plane in metres. Without a calibration, the system produces no metric values
  at all rather than guessing.
- **Baselines before learned models.** The congestion classifier is rule-based
  and explainable. Learned classifiers and forecasters will be compared
  against such baselines.

---

## 2. System overview

```
 video (file / webcam / RTSP)
   │   decode, resize to processing resolution, sample to process_fps
   ▼
 Preprocessing ── ROI mask (optional: CLAHE, lens undistortion)
   ▼
 DETECTOR  (YOLO11m, COCO-pretrained)           frame  → boxes, classes, confidences
   ▼
 TRACKER   (ByteTrack)                          boxes  → persistent track IDs
   ▼
 CAMERA MODEL (planar homography)               pixels → road-plane metres
   ▼
 TRAJECTORY KINEMATICS (least-squares velocity) positions → speed, direction, distance, stop time
   ▼
 FEATURE ENGINE (9 calculators, 60 s windows)   per-frame samples → traffic features
   ▼
 CONGESTION CLASSIFIER (rule-based)             features → FREE / MODERATE / HIGH / SEVERE + reasons
   ▼
 TrafficState  (per 60 s window; rolling 5 s live value every 1 s)
```

Each camera is handled by a **camera worker**. Frame-level data stays inside
the worker and is written only to Parquet and to an annotated video. The only
messages published to the rest of the system are the 1 s live values and the
60 s aggregated windows.

---

## 3. Datasets

> **Dataset strategy.** Each task uses a dataset suited to it: UA-DETRAC for detection and tracking,
> AI City CityFlowV2 for multi-camera tracking and calibration, AI City 2021 Track 4 for anomalies,
> and METR-LA / PEMS-BAY / PEMS08 for forecasting. All of them map into one canonical schema. See
> [DATASETS.md](DATASETS.md) for licences, adapters, the dataset-to-experiment matrix and which
> datasets still have to be obtained manually. The sections below describe the datasets used for
> the results reported so far.

The project uses three kinds of data:

- **(a)** a public pre-training dataset, used indirectly through the
  pretrained detector
- **(b)** public traffic videos, used as development footage
- **(c)** a synthetic dataset with exact ground truth, used for validation and
  testing

No data was labelled by hand, and no own camera footage has been recorded yet.

### 3.1 COCO (detector pre-training; used indirectly)

The detector is used exactly as published by Ultralytics, pretrained on
**MS COCO** (Lin et al., 2014), which has 80 object categories. The project
performs no training on COCO. The six COCO categories relevant to traffic are
mapped to the project's classes (§4.1.3). All other categories are discarded.

This has consequences for the results: COCO contains general street scenes,
not fixed high-mounted traffic cameras. Viewpoint and object-size differences
are therefore expected to reduce accuracy, particularly for distant vehicles,
two-wheelers and head-on truck cabs (§6.3).

### 3.2 Public development footage

Public videos stand in for own camera footage during development. A catalog
file, `configs/datasets/public_videos.yaml`, records each video's URL, licence
and SHA-256 hash. Downloads are verified against that hash and logged in
`data/external/SOURCES.md`.

| Name | Resolution | fps | Duration (frames) | Content | Licence | Used |
|---|---|---|---|---|---|---|
| `roboflow_vehicles` | 3840×2160 | 25 | 21.5 s (538) | Divided motorway seen from an overpass; cars and trucks in both directions | **Not stated** by the publisher (Roboflow `supervision` example assets). Used for local development only; not redistributed. | **Yes**, camera `public_highway_01` |
| `intel_car_detection` | 768×432 | 12.5 | 30.2 s (377) | Cars on a road | CC BY 4.0 (Intel IoT DevKit sample-videos) | Downloaded, not yet evaluated |
| `intel_person_bicycle_car` | 768×432 | 12 | 53.9 s (647) | Street scene with pedestrians, bicycles and cars | CC BY 4.0 | Downloaded, not yet evaluated |

**How the motorway footage was prepared** (`configs/cameras/public_highway.yaml`):

- **Analysed region.** One carriageway, the one whose traffic approaches the
  camera (two lanes). It is processed at 1920×1080 and 25 fps.
- **Approximate calibration from road markings.** There was no survey, so the
  site's geometry was inferred from the markings:
  1. A vehicle-free background image was built as the median of 14 frames.
  2. The three lane lines were located on it and found to be straight in the
     image, which is consistent with a planar road.
  3. Lateral scale: standard 3.75 m EU motorway lanes.
  4. Longitudinal scale: the lane-divider dash starts form an exact projective
     progression (fit residual 0.22 px RMS; cross-ratio of four consecutive
     dashes 1.33–1.34 against 4/3 expected). The dash-to-gap ratio is 1:1.92.
     The absolute dash period cannot be recovered from the image alone.
  5. Choosing the period: with 9 m, the only heavy truck measured 45 km/h,
     which is implausible given the EU 90 km/h truck limiter. **18 m** (6 m
     dash + 12 m gap) gives 91 km/h.
- **Result:** a homography with 0.05 m reprojection error over 12
  correspondences. Metric outputs on this footage are **indicative**, with an
  estimated ±10–15% error.
- **Placeholders:** the recording time and site location are unknown and set
  to placeholder values.

### 3.3 Synthetic traffic dataset (ground truth)

Real footage with ground-truth speeds, queues and counts was not available.
Instead, a seeded **microsimulation** generates scenes with exact ground truth
(`backend/traffic_intel/synthetic/`).

**Traffic model:**

- A single signalised approach with 2 lanes of 3.5 m.
- Vehicles follow the **Intelligent Driver Model** (Treiber, Hennecke &
  Helbing, 2000). Parameters: maximum acceleration 1.5 m/s², comfortable
  deceleration 2.0 m/s². Minimum gap and time headway vary by class (for
  example cars 2.0 m and 1.2 s, motorcycles 1.0 m and 0.8 s).
- Poisson arrivals.
- A fixed-time signal. Vehicles stop at the line on red and on amber when able.
- Class mix: 60% car, 22% motorcycle, 12% truck, 6% bus. Pedestrians walk along
  the kerb at 60 per hour.
- Desired speeds and vehicle dimensions vary by class. Examples: car 45 km/h,
  4.5 m long; bus 38 km/h, 12 m long.

**Camera model:** vehicle footprints are projected through a known perspective
homography into 1280×720 image boxes. Box height is included, scaled by the
local pixel-per-metre ratio.

**Detector noise model:**

| Effect | Value |
|---|---|
| Gaussian box jitter | σ = 1.5 px |
| Missed detections | 4% |
| Car↔truck label flips | 3% |
| Low-confidence detections (0.2–0.5) | 8% |
| Calibration picking noise | 1 px |

**Scenarios** (180 s each, 10 fps, 60 s warm-up, seed 7):

| Scenario | Arrivals (veh/h/lane) | Signal (cycle / green / amber, s) | Purpose |
|---|---|---|---|
| `free_flow` | 250 | none | uncongested reference |
| `default` | 500 | 60 / 27 / 3 | signalised, moderate queues |
| `congested` | 950 | 90 / 22 / 3 | oversaturated, standstill |
| `small` | 500 | 60 / 27 / 3 | 20 s fixture for automated tests |

**Ground truth available per frame:** object identity, class, lateral
position, distance upstream of the stop line, speed and image box. Per window,
the generator derives: vehicles present, space-mean speed, stop-line and
entry-line crossings, queue length and number of stopped vehicles. These use
the same definitions as the system (§4.5).

**Committed fixture.** `tests/fixtures/synthetic_small/` contains
`detections.csv`, `truth.csv`, `camera.yaml` and `scene.json`: 200 frames,
about 150 KB. An automated test checks that the seeded generator still
reproduces it exactly.

**Validity.** The synthetic data validates geometry, tracking logic and
feature definitions. It does **not** reproduce real detector behaviour: real
occlusion, lighting, appearance and failure modes are not modelled. Results on
it are an upper bound on what the pipeline can achieve given a detector of the
modelled quality.

---

## 4. Models

This section describes each model's role, method, configuration, inputs and
outputs. "Model" includes learned models (the detector), classical estimation
models (the tracker, camera model and kinematics) and rule-based models (the
feature engine and congestion classifier).

### 4.1 Object detector: YOLO11m (Ultralytics)

#### 4.1.1 Method and configuration

A single-stage convolutional object detector (Ultralytics YOLO11, medium
variant, `yolo11m.pt`), used with its published COCO weights. It has not been
fine-tuned.

| Parameter (`configs/pipeline.yaml`) | Value |
|---|---|
| Inference size (`imgsz`) | 1280 px (long side) |
| Confidence threshold | 0.25; per class: pedestrian 0.35, bicycle 0.30 |
| NMS IoU threshold | 0.5 |
| Maximum detections per frame | 300 |
| Device | `auto`: CUDA if available, else CPU (with a logged warning); fp16 on CUDA |
| Run device | `cuda:0`, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), CUDA 13.0, fp16 |

#### 4.1.2 Inputs and outputs

| | Description |
|---|---|
| **Input** | One or more BGR frames (H×W×3, uint8) at the processing resolution (1920×1080 for the motorway camera), after preprocessing. The ROI mask blacks out pixels outside the dilated analysis region. |
| **Output** | Per frame, a `Detections` set of N objects, each with: `bbox` (x1, y1, x2, y2 in pixels), `confidence` (0–1), `class_id`/`class_name` in the project taxonomy, and the frame `timestamp`. |

#### 4.1.3 Class mapping

Model class IDs are **not** assumed to be universal. The mapping is resolved
at load time against the model's own label table, configured by label name or
by integer ID, so a custom-trained model with different numbering needs only a
configuration change:

| Model label (COCO ID in YOLO11) | Project class |
|---|---|
| `car` (2) | car |
| `motorcycle` (3) | motorcycle |
| `bus` (5) | bus |
| `truck` (7) | truck |
| `bicycle` (1) | bicycle |
| `person` (0) | pedestrian |

#### 4.1.4 Runtime

Measured on 100 frames at 1080p, after warm-up:

| Model @ imgsz | ms / frame | frames/s |
|---|---|---|
| yolo11s @ 1280 | 32.5 | 30.7 |
| **yolo11m @ 1280** (selected) | 34.0 | 29.4 |
| yolo11m @ 960 | 35.5 | 28.2 |
| yolo11l @ 1280 | 46.3 | 21.6 |

The small (s) and medium (m) models cost almost the same, because the limit is
CPU-side pre/post-processing rather than the GPU. The medium model was chosen
because its accuracy advantage costs about 1.5 ms per frame. On the CPU-only
build, `yolo11s` at 640 px took about 200 ms per frame.

### 4.2 Multi-object tracker: ByteTrack

#### 4.2.1 Method

ByteTrack (Zhang et al., 2022) is **implemented natively** in NumPy/SciPy
rather than wrapped from a library. This removes the torch dependency and
makes the track lifecycle explicit and unit-testable.

- **Motion model:** a constant-velocity Kalman filter over box centre, aspect
  ratio and height, as in SORT (Bewley et al., 2016).
- **Association:** Hungarian assignment on IoU cost, in three rounds:
  1. High-confidence detections (≥ 0.5) against confirmed and lost tracks,
     with cost 1 − IoU·score and a threshold of 0.8.
  2. Low-confidence detections (0.1–0.5) against the remaining tracked tracks,
     threshold 0.5. This is the "BYTE" step, which keeps tracks alive through
     blur and occlusion.
  3. Remaining detections against tentative tracks.
- **Lifecycle:** NEW → TRACKED after 2 hits → LOST → REMOVED after 3 s unseen.
  Track IDs are never reused.
- **Class:** matching ignores class. Each track's class is a
  confidence-weighted vote over its history, which absorbs detector flicker
  between car and truck.
- **Variant:** BoT-SORT-style global motion compensation (Aharon et al., 2022).
  Background corners are tracked with Lucas–Kanade optical flow, and a
  RANSAC-fitted affine transform warps all track states. It is selectable in
  configuration for unstable camera mounts. Appearance re-identification is
  not included.

#### 4.2.2 Inputs and outputs

| | Description |
|---|---|
| **Input** | The detections of the current frame (boxes, confidences, classes). The frame image is used only by the motion-compensated variant. |
| **Output** | Confirmed tracks in this frame: `track_id`, `bbox`, `confidence` and voted `class_id`. |

### 4.3 Camera model: planar homography

| | Description |
|---|---|
| **Method** | A 3×3 homography estimated with OpenCV from ≥ 4 correspondences between pixels and road-plane metres. It is rescaled automatically to the processing resolution, sign-normalised so that points above the horizon are rejected, and reports a reprojection error in metres. Road-plane metres are georeferenced with an origin (lon/lat) and a bearing. |
| **Input** | Pixel coordinates, specifically the bottom-centre of each tracked box (assumed to touch the road). |
| **Output** | Ground coordinates (x across, y along the road, metres), WGS84 lon/lat, and a flag for whether the point lies inside the calibrated area (outside it, no speed is computed). |
| **Quality** | Reprojection error: 0.05 m (motorway); about 0.1 m on the synthetic camera, which has 1 px of simulated picking noise. |

### 4.4 Trajectory kinematics

| | Description |
|---|---|
| **Method** | Each track keeps a 10 s history of ground positions. **Velocity** is the least-squares slope of position against time over the last 1 s (a first-order Savitzky–Golay estimate), smoothed with an exponential moving average (α = 0.5). Below 12 km/h a 3 s window is also fitted and the lower value used, which reduces the upward bias that box jitter causes at standstill. Speed is withheld when the track is younger than 0.8 s, the box is cut off at the image edge, the point is outside the calibrated area, or the value exceeds 160 km/h. Distance is the integral of the smoothed speed. A vehicle is stopped below 5 km/h. |
| **Input** | Timestamped ground positions of one track. |
| **Output** | Per track: `speed_kmh` (or none), direction and heading, distance travelled, stopped duration, age. When the track ends, a trajectory summary with pixel and ground paths, mean speed, maximum stop time and line crossings. |

### 4.5 Traffic feature engine

A deterministic, modular model of standard traffic-engineering quantities.
Each processed frame is reduced to a sample of the objects inside the region
of interest. Nine independent calculators compute window metrics, and the
same calculators serve both the 60 s windows and the rolling 5 s live value.

| Feature | Definition (summary) |
|---|---|
| Counts | Unique tracks continuously present in the region for ≥ 1 s; counted per class; passenger-car-unit weighted total |
| Average / median speed | **Space-mean speed**: mean (median) of the instantaneous speeds of vehicles in the region, over all sampled instants |
| Density | Time-averaged vehicles (PCU) ÷ (region length in km × lanes); also expressed as a percentage of jam density (150 PCU/km/lane) |
| Occupancy | Share of time a virtual detection loop is covered by a vehicle footprint |
| Flow, inflow, outflow | Direction-checked crossings of the entry and exit lines, counted once per vehicle; flow in vehicles per hour |
| Queue length | Per lane, a chain of slow vehicles (< 7 km/h) starting within 15 m of the stop line, with bumper gaps ≤ 15 m. Length to the tail of the last vehicle. |
| Stopped vehicles | Time-averaged number of vehicles below 5 km/h for ≥ 2 s |
| Gaps, trajectory statistics | Bumper-to-bumper gaps in each lane; durations, distances and straightness of completed trajectories |

| | Description |
|---|---|
| **Input** | Tracked objects per frame (pixel boxes, ground positions, kinematics) and the camera's zone geometry (analysis region, lanes, entry/exit lines, queue zone, detection loop). |
| **Output** | `TrafficFeatures` per window: the quantities above plus a data-quality score (processed frames ÷ expected frames). |

Full definitions are in [TRAFFIC_STATE_ESTIMATION.md §3–4](TRAFFIC_STATE_ESTIMATION.md).

### 4.6 Congestion classifier (rule-based, explainable)

| | Description |
|---|---|
| **Method** | Four indicators are normalised to 0–1: speed deficit relative to free-flow speed, density ratio, occupancy and queue length. They are combined as a weighted mean (weights 0.40 / 0.30 / 0.15 / 0.15; missing speed is skipped and the weights renormalised). The score maps to a level: FREE ≤ 0.30 < MODERATE ≤ 0.55 < HIGH ≤ 0.75 < SEVERE. Override rules can only raise the level (for example, "≥ 5 stopped vehicles and speed < 5 km/h → SEVERE"). Hysteresis requires two consecutive lower windows before a downgrade. All values are configurable in `configs/congestion_rules.yaml`. |
| **Input** | One window's `TrafficFeatures` and the approach's free-flow speed (from the road-network configuration). |
| **Output** | Level (FREE / MODERATE / HIGH / SEVERE), a score from 0 to 1, and a human-readable explanation of every indicator, override and hysteresis decision. |

This classifier is the baseline that learned classifiers (Random Forest,
XGBoost) will later be compared against, using manually labelled windows.

### 4.7 Forecasting models (phase 5)

The forecasting layer is described in detail, with full results, in [FORECASTING.md](FORECASTING.md).

- **Inputs.** A window of recent 1-min (or 5-min) traffic states: speed, density, occupancy, queue,
  flow, counts. Plus calendar features (hour, weekday, holiday, rush hour) and weather.
- **Outputs.** The same targets at +5, +15 and +30 min, with predicted congestion levels.
- **Models compared on identical chronological splits and windows:**
  - baselines: persistence, seasonal-naive;
  - linear regression;
  - Random Forest;
  - XGBoost;
  - LSTM, GRU and a lightweight Transformer (three seeds each).
- **Result.** On real PeMS08 data, XGBoost had the lowest test RMSE at every horizon (flow RMSE
  242 veh/h vs 320 for persistence). The best deep model was 3.9 % worse. XGBoost is the
  production model.

### 4.8 Incident-intelligence models (phase 6)

The incident layer is described in detail in [INCIDENT_INTELLIGENCE.md](INCIDENT_INTELLIGENCE.md).

- **Track detectors (rules on trajectories).** Stopped vehicle (with duration, location and
  confidence), wrong-way (against the *configured* lane direction), and swerving or off-lane
  movement.
- **Series detectors (per 60 s window).**
  - Statistical: sudden slowdown (EWMA baseline, robust z-score, mean and p85 speed), queue growth,
    density z-score, inflow/outflow imbalance, EWMA control chart, rolling z-score.
  - Learned: Isolation Forest and an MLP autoencoder, fitted on normal windows only.
- **Incident fusion.** A logistic model over evidence families. Correlated congestion symptoms
  are combined by noisy-OR, and an interaction term requires symptoms *and* a physical cue for a
  high probability. A configurable taxonomy names the situation (normal, sudden_slowdown,
  stopped_vehicle, wrong_way, abnormal_queue, possible_incident). The weights are hand-set, so
  every output is marked `calibrated=false`.
- **Risk engine.** A documented weighted score:

  `risk = 100 × Σwᵢcᵢ/Σwᵢ × (0.7 + 0.3·importance)`

  over current severity, forecast severity, incident probability, queue, speed and density. It
  has safety floors and reason codes.
- **What-if.** Static traffic assignment on a NetworkX graph of the road network (BPR link
  times, method of successive averages). It is explicitly *not* a microscopic simulator.

---

## 5. System inputs and outputs

### 5.1 Inputs

| Input | Form | Source |
|---|---|---|
| Video | File (MP4 etc.), webcam (`webcam:N`) or stream (`rtsp://`, `http(s)://`) | `source:` in the camera configuration |
| Camera configuration | Resolution, processing fps, recording start time, calibration points, zones | `configs/cameras/*.yaml` |
| Road network | Junctions, road segments (lanes, speed limit, free-flow speed), approaches | `configs/network/*.yaml` |
| Model and pipeline settings | Detector weights, device, thresholds, class mapping, tracker, feature and rule parameters | `configs/pipeline.yaml`, `configs/congestion_rules.yaml` |

### 5.2 Outputs

**Primary output: `TrafficState`**, one per 60 s window per approach, plus a
rolling 5 s live value about every second. Below is the actual output for the
public motorway clip (21.5 s, so a single partial window; explanation
abbreviated):

```json
{
  "window_start": "2026-09-01T06:00:00Z", "window_s": 21.52, "is_live": false,
  "camera_id": "public_highway_01", "junction_id": "dev_highway", "road_id": "dev_highway_approach_seg",
  "vehicle_count": 7, "car_count": 6, "truck_count": 1, "bus_count": 0, "motorcycle_count": 0,
  "bicycle_count": 0, "pedestrian_count": 0,
  "average_speed_kmh": 109.93, "median_speed_kmh": 103.02,
  "density_veh_km": 4.94, "density_pct": 3.3, "occupancy_pct": 8.74,
  "flow_veh_h": 1171.0, "inflow": 4, "outflow": 7,
  "queue_length_m": 0.0, "stopped_vehicle_count": 0,
  "traffic_state": "free", "congestion_score": 0.0762,
  "explanation": ["speed 109.9 km/h = 92% of free-flow 120 km/h -> 0.10 x w0.40",
                  "density 4.9 veh/km/lane = 3% of jam density -> 0.04 x w0.30",
                  "occupancy 8.7% -> 0.15 x w0.15", "queue 0 m -> 0.00 x w0.15",
                  "score 0.08 -> FREE"],
  "quality": 1.0
}
```

**Recorded outputs.** Frame-level data is kept local and never published.

| Output | Content | Location |
|---|---|---|
| Detections (Parquet) | One row per detection: frame, timestamp, class, confidence, box | `data/interim/<camera>/<date>/<run>/detections-*.parquet` |
| Tracks (Parquet) | One row per tracked object per frame: track ID, class, box, anchor, ground x/y (m), speed, stopped time, age | `…/tracks-*.parquet` |
| Trajectory summaries (Parquet, JSONL) | One row per completed trajectory: paths in pixels and metres, mean speed, stops, line crossings | `…/track_summaries-*.parquet`, `runs/<camera>/<run>/tracks.jsonl` |
| Annotated video | Boxes, IDs, class, speed, trajectory tails, zones, live and window status | `runs/<camera>/<run>/annotated.mp4` |
| Run summary | Device, model, per-stage latency, calibration error, published message counts | `runs/<camera>/<run>/summary.json` |

**Published messages** (to the event bus, for the future API and dashboard):
`traffic.features.live` (1 s), `traffic.features.window` and `traffic.state`
(60 s). No other messages are permitted; this is enforced in code and tested.

---

## 6. Experiments and results

### 6.1 Accuracy on synthetic data (ground truth)

Estimates against ground truth, per 60 s window:

| Scenario | Window | Vehicles est / true | Speed km/h est / true | Outflow est / true | Queue m est / true | Level |
|---|---|---|---|---|---|---|
| free_flow | 1 | 11 / 11 | 42.3 / 41.6 | 9 / 9 | 0 / 0 | FREE |
| free_flow | 2 | 9 / 9 | 41.1 / 40.6 | 9 / 9 | 0 / 0 | FREE |
| free_flow | 3 | 13 / 13 | 41.4 / 40.9 | 13 / 13 | 0 / 0 | FREE |
| default | 1 | 31 / 31 | 11.4 / 11.1 | 19 / 19 | 20.5 / 21.0 | SEVERE |
| default | 2 | 26 / 26 | 12.9 / 13.5 | 19 / 19 | 11.4 / 10.1 | SEVERE (hysteresis hold) |
| default | 3 | 19 / 19 | 12.1 / 12.3 | 13 / 13 | 9.9 / 9.2 | HIGH |
| congested | 1 | 37 / 36 | 4.7 / 4.8 | 17 / 17 | 36.4 / 34.2 | SEVERE |
| congested | 2 | 30 / 30 | 4.0 / 4.5 | 19 / 19 | 39.5 / 34.9 | SEVERE |
| congested | 3 | 22 / 18 | 1.1 / 0.5 | 0 / 0 | 60.8 / 59.3 | SEVERE |

**Observations:**

- **Flow is exact in all nine windows.** Presence counts are exact in free and
  signalised traffic.
- **Mean speed error ≤ 0.7 km/h.** Queue error is ≤ 1.5 m in eight of nine
  windows (4.6 m in one congested window).
- **Counts in the congested standstill are +22%.** Diagnosis showed this is ID
  fragmentation of small, overlapping boxes 55–62 m from the camera. It is a
  known weakness of IoU-based association, and it does not affect flow.
- **Levels are ordered as expected**: every free-flow score is below every
  congested score.
- **The signalised `default` scenario is rated HIGH/SEVERE.** Red-phase stops
  lower space-mean speed even without real congestion, which shows that the
  rule thresholds need per-site tuning.

Several defects were found and corrected during this validation:

- a homography sign ambiguity
- an anchor-point bias of about 2 m in queue length
- a speed bias near standstill
- a class-sampling bias in the generator itself

### 6.2 Public motorway footage (no ground truth)

Configuration: yolo11m @ 1280 px, fp16, RTX 4060 Laptop; 538 frames at 1920×1080.

| Quantity | Value |
|---|---|
| End-to-end throughput | 11.8–13.7 frames/s (4K decoding overlapped with inference) |
| Mean latency per frame | detection 38.8–42.9 ms, preprocessing 5.7–6.1 ms, tracking 0.5 ms, features 1.4 ms |
| Raw detections in the analysis region | 894 (car 594, truck 198, pedestrian 93, bus 9); mean confidence 0.78 |
| Tracks / completed trajectories | 9 / 9 (7 counted vehicles: 6 cars, 1 truck) |
| Window result | FREE; space-mean speed 109.9 km/h; flow 1,171 veh/h; occupancy 8.7% |
| Per-vehicle mean speeds | 89–149 km/h for tracks labelled car, 91 km/h for the truck. The 89 km/h "car" is the misclassified DAF truck (§6.3). All indicative (approximate calibration). |
| Published messages | 22 live values, 1 window (features + state) |

### 6.3 Qualitative error analysis on real footage

Reviewing the annotated video showed:

1. **A truck labelled as a car.** A DAF truck cab seen head-on was classified
   as a car in most frames, so the per-track vote kept "car" for that track.
   (Another truck was recognised correctly, and the 198 truck detections come
   from both.) This is a COCO domain-shift effect.
2. **False pedestrians.** The 93 pedestrian detections are the *truck driver*
   seen through the windscreen. They formed a "pedestrian" track moving at
   122 km/h. The continuous-presence rule kept it out of the counts, but it
   remains in the raw data. A rule that suppresses pedestrian boxes inside
   vehicle boxes is an easy mitigation.

Both indicate that a fine-tuned or traffic-specific detector will be needed
before class-level counts on real footage can be trusted.

---

## 7. Limitations and threats to validity

- **Not validated on labelled real video.** Detection recall, tracking
  accuracy (MOTA/IDF1/HOTA), count accuracy and speed accuracy have not been
  measured on labelled real footage. The quantitative results are synthetic.
- **Approximate calibration** of the public footage (assumed lane width and
  dash period, one-truck sanity check). Its metric outputs are indicative.
- **Very little real data**: 21.5 s of one motorway view, one partial window.
  No statistical conclusions about traffic can be drawn from it.
- **Synthetic realism.** The simulated detector noise is independent and
  Gaussian; real errors are correlated and depend on appearance. The simulated
  traffic has one lane discipline, no lane changes and no turning.
- **Pretrained detector domain gap**: COCO is not traffic-camera data (§6.3).
- **Unknown licence** of the main public clip. It is used only locally; no
  frames or results derived from it should be published.
- **Rule thresholds are generic.** They were not calibrated to local traffic,
  and signalised approaches are over-rated.

---

## 8. Future work

1. **Record own footage** at a known site. Survey calibration points and
   validate speed against a known distance or a GPS probe vehicle.
2. **Label a real evaluation set** (a few minutes of annotated frames and
   tracks). Measure mAP, MOTA/IDF1 and count and speed error. Fine-tune the
   detector on local traffic, especially trucks, two-wheelers and the
   pedestrian-in-vehicle case.
3. **Dashboard (Prompt 5):** map, incident panel with evidence and uncertainty,
   operator acknowledge/dismiss, what-if panel, live updates (since implemented with REST
   polling; a WebSocket push channel remains an optional future interface).
4. **Forecasting on own data:** retrain on long recordings from the target site
   (the current models are trained on PeMS08 and on synthetic series).
5. **Incident calibration:** collect operator labels and real annotated incidents,
   fit and calibrate the fusion model, learn time-of-day baselines; learned
   congestion classifiers evaluated against the rule baseline (Phase 7).

---

## 9. Reproducibility

**Environment:**

- Windows 11, Python 3.13.1
- torch 2.14.0+cu130, Ultralytics 8.4.160
- OpenCV 5.0.0, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1
- NVIDIA RTX 4060 Laptop GPU, 8 GB

All synthetic data is seeded (seed 7). Public videos are pinned by SHA-256.

```bash
cd backend
pip install -e ".[dev,cv]"
pip install --force-reinstall --no-deps torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pytest                                                           # 108 tests
python -m traffic_intel.pipeline synthetic --scenario congested  # §6.1 tables (also: default, free_flow)
python -m traffic_intel.datasets download roboflow_vehicles
python -m traffic_intel.pipeline replay public_highway_01 --annotate --live   # §6.2
```

---

## References

- Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* arXiv:2206.14651.
- Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). *Simple Online and Realtime Tracking.* IEEE ICIP.
- Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11.* https://github.com/ultralytics/ultralytics
- Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV.
- Savitzky, A., & Golay, M. J. E. (1964). *Smoothing and Differentiation of Data by Simplified Least Squares Procedures.* Analytical Chemistry, 36(8).
- Treiber, M., Hennecke, A., & Helbing, D. (2000). *Congested traffic states in empirical observations and microscopic simulations.* Physical Review E, 62(2).
- Zhang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV.
