# Perception and traffic state estimation

This document covers the layer from **video to `TrafficState`** (Phases 1–2):
what each stage does, how every metric is defined, the assumptions behind the
numbers, their known limitations, how to calibrate a camera, and how accurate
the layer is on synthetic data with ground truth.

```
VideoSource ─► preprocess ─► Detector ─► Tracker ─► GroundProjector ─► TrajectoryStore
 (file/webcam/   (ROI mask,    (YOLO,      (ByteTrack /   (homography     (per-track history,
  RTSP; fps       CLAHE,        class       BoT-SORT)      px → metres)    speed, direction,
  sampling,       undistort)    mapping)                                   distance, stopped)
  resize)                                                                         │
                                                                                  ▼
TrafficState ◄── RuleBasedCongestionClassifier ◄── FeatureCalculators ◄── FrameSamples + line crossings
(per 60 s window,     (explainable,                 (counts, speed, density,   (every processed frame)
 and ~1 Hz live)       configurable)                  occupancy, flow, queue …)
```

Everything is composed from `configs/pipeline.yaml` through registries, so each
box can be replaced without touching the others (see `PerceptionPipeline.build`).

---

## 1. Running it

```bash
cd backend
pip install -e ".[dev,cv]"            # cv = ultralytics/torch (only needed for YOLO)

# No video or GPU needed: IDM traffic simulation + noisy detections, compared with ground truth
python -m traffic_intel.pipeline synthetic --scenario default     # free_flow | congested | small
python -m traffic_intel.pipeline synthetic --scenario congested --annotate   # + annotated.mp4

# Real footage: configure configs/cameras/cam_01.yaml, then
python -m traffic_intel.pipeline replay cam_01 --annotate --live
python -m traffic_intel.pipeline replay cam_01 --source path/to/video.mp4 --max-seconds 300
python -m traffic_intel.pipeline replay cam_01 --source webcam:0            # live webcam
python -m traffic_intel.pipeline replay cam_01 --source rtsp://user:pass@host/stream

# Any camera YAML + detections from a CSV (no detector): e.g. the exported synthetic dataset
python -m traffic_intel.synthetic export --scenario default --out data/synthetic/default --video
python -m traffic_intel.pipeline replay data/synthetic/default/camera.yaml \
       --detections data/synthetic/default/detections.csv --annotate

# Calibration check (writes frame / overlay / bird's-eye images)
python -m traffic_intel.computer_vision.calibration_tool cam_01 --frame 0
```

Outputs go to `runs/<camera>/<timestamp>/`:

| File | Content |
|---|---|
| `traffic_states.jsonl` | one `TrafficState` per closed 60 s window (the main output) |
| `live_states.jsonl` | rolling 5 s `TrafficState` about every 1 s (`--live`) |
| `tracks.jsonl` | one `TrackSummary` per finished trajectory (path in pixels and metres, speed, stops, crossings) |
| `annotated.mp4` | boxes, IDs, speeds, trajectory tails, zones, live/window HUD (`--annotate`) |
| `summary.json` | frames, processing fps, per-stage latency, calibration error, ROI length, all states |
| `data/interim/<camera>/<date>/<run_id>/{detections,tracks,track_summaries}-*.parquet` | frame-level data for offline datasets (`outputs:` in `configs/pipeline.yaml`) |

Example `TrafficState`: the first window of the synthetic `default` scenario,
with a few fields omitted. The median is far below the mean because most
samples at a red signal are stopped vehicles.

```json
{"timestamp": "2026-09-01T02:31:00Z", "window_s": 60.0, "is_live": false,
 "junction_id": "junction_a", "road_id": "road_a_sb", "approach_id": "junction_a_north",
 "vehicle_count": 31, "car_count": 20, "motorcycle_count": 6, "bus_count": 3, "truck_count": 2,
 "pedestrian_count": 4, "average_speed_kmh": 11.4, "median_speed_kmh": 1.53,
 "density_veh_km": 66.3, "density_pct": 44.2, "occupancy_pct": 63.83, "flow_veh_h": 1140.0,
 "inflow": 21, "outflow": 19, "queue_length_m": 20.5, "stopped_vehicle_count": 3,
 "traffic_state": "severe", "congestion_score": 0.753, "quality": 1.0,
 "explanation": ["speed 11.4 km/h = 23% of free-flow 50 km/h -> 0.96 x w0.40",
                 "density 66.3 veh/km/lane = 44% of jam density -> 0.55 x w0.30",
                 "occupancy 63.8% -> 1.00 x w0.15",
                 "queue 20 m -> 0.34 x w0.15",
                 "score 0.75 -> SEVERE"]}
```

---

## 2. Stages

### 2.1 Video input (`computer_vision/sources/opencv_source.py`)

| Source | Example `source` | Timestamps | Frame dropping |
|---|---|---|---|
| file | `data/raw/cam_01.mp4` | `replay_start + frame_index / source_fps`, deterministic | never (all sampled frames are processed) |
| webcam | `webcam:0` | UTC wall clock at capture | latest-frame buffer: a slow consumer skips frames instead of lagging |
| stream | `rtsp://…`, `http(s)://…` | UTC wall clock at capture | same as webcam, with automatic reconnect (exponential backoff) |

- **Sampling.** `process_fps` puts frames on a fixed time grid: the first
  decoded frame in each `1/process_fps` slot is used. A 30 fps file at
  `process_fps: 10` uses frames 0, 3, 6, …; a 25 fps file gets 0, 3, 5, 8, 10,
  … with the correct times.
- **Resolution.** `resolution: [w, h]` resizes right after decoding. Zones and
  calibration are given at their own reference size and are **rescaled
  automatically**, so changing the resolution never means redrawing zones.
- **Replay time.** For files, `replay_start` fixes when frame 0 happened, so
  hour and day-of-week features are right for recorded footage. `realtime`
  paces playback, and `loop` restarts the file with time continuing.
- **Preprocessing** (`preprocessing.py`, applied in order):
  - `roi_mask` blacks out everything outside the dilated ROI before detection.
    This is an optimisation; features filter by ROI independently.
  - `clahe` is a low-light contrast boost, applied only when the frame is dark.
  - `undistort` corrects lens distortion. Calibration points must then be
    picked on undistorted frames.

### 2.2 Detection (`computer_vision/detectors/`)

- The `Detector` interface is `detect(frames) -> list[Detections]`, batched.
  Implementations:
  - `yolo`: Ultralytics; any `.pt`, `.onnx` or `.engine` model.
  - `precomputed`: detections replayed from a CSV.
  - `synthetic`: the test dataset.
- **Class mapping** (`class_mapping.py`) is resolved against the model's own
  `names` table at load time. It maps by label (`car: car`,
  `person: pedestrian`) or by integer model id (`7: truck`). This lets COCO
  models and custom-trained models with different numbering both work.
  Unmapped classes are dropped inside NMS. `class_confidence` sets per-class
  minimum confidence (for example pedestrians 0.35).
- Output per object: `class_id`, `class_name`, `confidence`, `bbox`,
  `timestamp`, available through the `Detections.to_list()` view.
- Weights download automatically on first use (`auto_download`). On a CPU,
  `yolo11s` takes about 200 ms per frame at `imgsz: 640` on this machine, so
  use `yolo11n`, a lower `process_fps`, or a GPU for real time.

### 2.3 Tracking (`tracking/trackers/`)

- **ByteTrack** is implemented natively in NumPy/SciPy: a Kalman filter in
  (cx, cy, aspect, h) plus Hungarian assignment. Being native, it has no torch
  dependency and can be tested frame by frame. Each frame:
  1. High-score detections are matched to confirmed and lost tracks.
  2. **Low-score detections are matched to the remaining tracks.** This is the
     "BYTE" step, and it keeps tracks alive through blur and partial occlusion.
  3. The remaining detections are matched to tentative tracks.
  4. New tentative tracks are created.
- **Lifecycle:**

  ```
  NEW (tentative, not output) ─min_hits─► TRACKED ─miss─► LOST ─track_buffer_s─► REMOVED
                                                ▲──────── re-found ────┘
  ```

  IDs are never reused, even after `reset()`.
- Matching is class-agnostic, and each track's class is a
  **confidence-weighted vote** over its history. This matters because YOLO
  often flickers between car and truck.
- **BoT-SORT** is ByteTrack plus global motion compensation. Background
  corners (outside detection boxes) are tracked with pyramidal Lucas–Kanade
  optical flow, a partial affine is fitted with RANSAC, and all track states
  are warped by it. Use it for shaky poles or PTZ drift. ReID is not included.

### 2.4 Camera calibration (`computer_vision/homography.py`)

- The pixel-to-ground mapping is a planar homography fitted to at least 4
  correspondences (use 6–8, spread over the road). Ground metres are then
  placed on the map using `origin_lonlat` and `bearing_deg`.
- **Ground frame convention.** Metres on the road plane. `bearing_deg` is the
  compass direction of `+y`; `+x` is 90° clockwise from `+y`. The direction of
  travel is configured separately (lane `direction`, `travel_direction`, or
  derived from the entry line to the exit line).
- The sign of the homography is normalised so that points above the horizon
  map to NaN instead of to a mirrored road.
- **Reported quality:**
  - `reprojection_error_m`: mean error of the correspondences, in metres. A
    warning is logged above 1 m.
  - `in_calibrated_region()`: projections are only trusted inside the hull of
    the calibration points, plus a 15% margin.

### 2.5 Trajectories and speed (`tracking/trajectory.py`)

Each track keeps a bounded history (`history_s`) of anchor points, both in
pixels and in ground metres. The anchor is the **bottom-centre of the box**,
the point assumed to touch the road.

| Quantity | Method |
|---|---|
| velocity | Least-squares slope of ground position against time over the last `speed_window_s` (1 s). This is a first-order Savitzky–Golay estimate, robust to per-frame box jitter. It is then EMA-smoothed. |
| slow vehicles | Below `slow_speed_kmh` (12), velocity is also fitted over `slow_window_s` (3 s), and the **lower** of the two is used. This removes the upward bias jitter causes near standstill without delaying detection of a vehicle that has just stopped. |
| `vehicle_speed_kmh` | \|velocity\| × 3.6. It is **None** in any of these cases: the track is younger than 0.8 s; there are fewer than 3 reliable points or less than 0.5 s of span; the box is truncated at the frame edge; the anchor is outside the calibrated area; or the value exceeds `max_speed_kmh`. A stale speed expires after one window. |
| direction / heading | Unit velocity vector in ground coordinates, and compass heading via the bearing. |
| distance travelled | Integral of the smoothed speed over time. Integrating raw point-to-point jumps would turn jitter into phantom metres. Short unreliable gaps are bridged. |
| stopped duration | Continuous time with speed below `stopped_speed_kmh` (5). |
| class | Confidence-weighted majority over the track's detections. |

Finished trajectories (unseen for `max_age_s`) become `TrackSummary` records.
Each holds a 2 Hz polyline in pixels and metres, mean speed, maximum stop
time, line crossings and displacement along the travel axis.

### 2.6 Feature engine (`traffic/features/`)

1. **Per frame.** Tracks inside the ROI are reduced to a `FrameSample`: speed,
   stopped and queued flags, lane, distance upstream of the stop line, lateral
   offset, and loop occupancy. Line crossings become `CrossingEvent`s.
2. **Per window.** Independent `FeatureCalculator`s compute metrics from the
   samples and events, in the order listed in
   `feature_engine.params.calculators`. The same calculators serve fixed 60 s
   windows and the rolling 5 s live window.

Windows are aligned to multiples of `window_s` in *source* time. A window with
no frames is still emitted, with `quality = 0`, so gaps are explicit. At end
of stream, `flush()` emits the final partial window.

**Adding a feature.** Register a calculator; its outputs land in
`TrafficFeatures.extra` until they are promoted to a column:

```python
@FEATURE_CALCULATORS.register("heavy_share")
class HeavyShare(FeatureCalculator):
    def compute(self, w):
        c = w.results["count_by_class"]
        return {"heavy_share": (c["bus"] + c["truck"]) / max(1, w.results["vehicle_count"])}
```

Then add `heavy_share` to `feature_engine.params.calculators`.

---

## 3. Metric definitions

Notation:

- **ROI**: the configured analysis polygon. Presence and counts use it.
- **Measurement region**: the ROI intersected with the calibrated area. Every
  metric quantity uses it.
- **L**: the length of the measurement region along the travel axis.
- **lanes**: the number of lane zones, or `lanes` from the config.

| Metric | Definition |
|---|---|
| `vehicle_count`, `car_count`, … `bicycle_count` | Unique vehicle tracks present in the ROI during the window. A track needs ≥ `min_track_points` samples **and** a continuous run of ≥ `min_presence_s` inside the ROI. Class is the track's majority vote. A track spanning two windows is present in both; throughput is `flow`. |
| `pedestrian_count` | Same rule, class pedestrian. Pedestrians are excluded from every vehicle metric. |
| `pcu_count` | Sum of passenger-car units of the counted vehicles (car 1, motorcycle 0.5, bus/truck 3, bicycle 0.4; overridable). |
| `average_speed_kmh` | **Space-mean speed**: arithmetic mean of the instantaneous speeds of all vehicles in the measurement region, over every sampled instant. This is the definition of space-mean speed for spatial snapshots; the harmonic mean applies to point detectors such as loops. Stopped vehicles are included (about 0 km/h). It is None with fewer than 3 speed samples. |
| `median_speed_kmh`, `speed_p85_kmh` | Median and 85th percentile of the same samples. |
| `speed_distribution` | Histogram of **per-vehicle** mean speeds (bins 0-5, 5-10, 10-20, … 80+). |
| `density_veh_km` | Time-averaged vehicles in the measurement region (PCU-weighted if `use_pcu`) divided by (L km × lanes). |
| `density_pct` (`density_ratio` × 100) | Density divided by jam density (default 150 PCU/km/lane). This is the "Current density: 82%" figure. |
| `occupancy_pct` | With a virtual loop: % of sampled instants in which any vehicle footprint (the box bottom edge) overlaps the loop, like an inductive loop. Without a loop: space occupancy, the sum of typical vehicle lengths in the region divided by (L × lanes). The method used is in `extra.occupancy_method`. |
| `inflow` / `outflow` | Vehicles whose anchor crosses the entry or exit (stop) line **in the travel direction**. Each track counts once per line and direction. Crossings the other way go to `extra.reverse_crossings`. |
| `flow_veh_h` | outflow × 3600 / window. Uses inflow if there is no exit line. With no lines at all it uses q = k·v·lanes (`extra.flow_source` = 0, 1 or 2 for exit, entry, fundamental). |
| `stopped_vehicle_count` | Time-averaged number of vehicles that have been below 5 km/h for ≥ `stopped_min_s` (2 s). `extra` also has the maximum and the unique count. |
| `queue_length_m` | See §4. Time-averaged over the window; `extra.queue_max_m` has the maximum. |
| `avg_gap_m` | Mean bumper-to-bumper gap between consecutive vehicles in the same lane: tail of the vehicle ahead to nose of the one behind, using typical class lengths. |
| `trajectory_stats` | For trajectories that ended in the window: count, mean duration, mean distance, straightness (displacement ÷ path length), mean speed, number that completed entry-to-exit, number of reverse-direction tracks, longest stop. |
| `quality` | Processed frames divided by expected frames for the window span. |

---

## 4. Queue estimation

- **Stopped vehicle:** speed below `stopped_speed_kmh` (5 km/h) for at least
  `stopped_min_s` (2 s).
- **Queued vehicle:** speed below `queue_speed_kmh` (7 km/h), so stopped or
  crawling, with its anchor inside the **queue zone** (the measurement region
  if no queue zone is configured).
- **Queue:** per lane, the queued vehicles sorted by distance upstream of the
  stop line. The queue starts if the first one is within `queue_head_max_m`
  (15 m) of the stop line. It continues while the bumper gap to the next
  queued vehicle is at most `queue_max_gap_m` (15 m).
- **Queue length:** distance from the stop line to the **tail** of the last
  vehicle in the chain. The approach value is the longest lane queue, averaged
  over the window.
- **Anchor reference.** The box bottom is the part of the vehicle nearest the
  camera. For traffic approaching the camera that is the front bumper, so the
  tail is anchor + length; for receding traffic it is the rear bumper. This is
  derived automatically from the geometry (`anchor_reference: auto`), or can
  be set to `front`, `rear` or `center`. Using the correct reference removed a
  systematic under-estimate of about 2 m.
- **Visibility.** A queue cannot be measured beyond the camera's view.
  `extra.queue_exceeds_view_fraction` reports how often the tail reached 90%
  of the ROI length. The congestion rules treat "≥ 50% of the window" as at
  least HIGH.

Precision is at best about one vehicle length; the method does **not** claim
centimetre accuracy.

---

## 5. Congestion classification

The rule-based classifier (`traffic/congestion/rules.py`) is configured in
`configs/congestion_rules.yaml`.

1. **Indicators**, each normalised to 0..1 where 1 is worst:
   - speed: `(1 − avg_speed / free_flow_speed) / saturation`. Free-flow speed
     comes from the road segment in the network config.
   - density: `density_ratio / saturation`.
   - occupancy: `occupancy_pct / saturation`.
   - queue: `queue_length_m / min(saturation, ROI length)`.
2. **Score** = weighted mean of the available indicators. A missing speed is
   skipped and the remaining weights are renormalised.
3. **Level** from the score: FREE ≤ 0.30 < MODERATE ≤ 0.55 < HIGH ≤ 0.75 < SEVERE.
4. **Overrides** are boolean expressions over any feature, such as
   `"stopped_vehicles >= 5 and avg_speed_kmh < 5"`. They are evaluated with a
   whitelisted AST (no `eval`) and can only *raise* the level.
5. **Hysteresis:** a lower level is accepted only after 2 consecutive lower
   windows. The live classifier is stateless.
6. **Explanation.** Every indicator value, the score, any override that fired
   and any hysteresis hold are written to `explanation`, and low data quality
   is flagged.

ML classifiers (Random Forest, XGBoost) plug into the same
`CongestionClassifier` interface. They receive `free_flow_speeds` and
`stateful` if their constructor accepts them.

---

## 6. Assumptions

1. **Flat road inside the calibrated area.** A homography maps one plane.
   Slopes, crowned roads and flyovers produce distance errors that grow with
   the departure from planarity.
2. **The box bottom-centre lies on the road** and marks the nearest end of the
   vehicle. For tall vehicles seen obliquely, the visible bottom edge can sit
   slightly above or beside the true contact point.
3. **Lens distortion is negligible,** or `undistort` is enabled with
   calibration done on undistorted frames.
4. **The camera does not move after calibration.** BoT-SORT compensates for
   small shake in *tracking*, but the calibration itself does not follow
   camera motion. Recalibrate after any PTZ move.
5. **Typical vehicle lengths per class** are used for gaps, queue tails and
   space occupancy. Real lengths vary; a bus can be 9–18 m.
6. **One approach per camera, with one dominant travel direction.** Turning
   movements inside the view are tracked, but approach metrics assume a main
   axis.
7. **Clock.** File timestamps come from `replay_start` plus the container's
   fps. Variable-frame-rate files make timestamps approximate; set
   `source_fps` if the container reports a wrong rate.

---

## 7. Limitations (measured or known)

| Limitation | Effect | Mitigation |
|---|---|---|
| Perspective: 1 px in the far field can be several metres | Speed and position noise grow roughly with distance²; anchors near a far ROI edge flicker in and out | Keep the ROI inside the calibrated, well-resolved area. Speeds are only computed inside the calibrated hull. Presence needs a continuous run (`min_presence_s`). |
| Small, overlapping far-field boxes in dense standstill | IoU association fragments or swaps IDs. In the synthetic `congested` standstill window, presence counts were +4 of 18 (+22%). All extra tracks belonged to real vehicles 55–62 m away. | Use flow (line crossings) for throughput: it was exact in every test. Use a higher `imgsz` or a GPU model for small objects, and BoT-SORT for shaky cameras. |
| Speed near standstill | Residual positive bias of about 0.5–1 km/h from jitter, because a vector magnitude is always positive | Slow-vehicle long window. Stopped detection uses a 5 km/h threshold with a 2 s minimum. |
| Detection misses (night, rain, occlusion by buses) | Undercounts, and gaps in tracks | `clahe` for low light; the ByteTrack low-score round; `quality` reports frame coverage but *not* detection recall. Validate on labelled clips. |
| Class confusion (car/truck, motorcycle/bicycle) | Class counts and PCU errors | Majority vote per track; per-class confidence; fine-tune the detector on local traffic. |
| Signalised approaches | Red-phase stops lower space-mean speed without real congestion, so rules can over-rate a normal cycle (the synthetic `default` scenario scores 0.59–0.75) | Tune the weights, saturations and free-flow speed per approach. Windows of 60 s or more average over a cycle. The ML classifier can learn this. |
| Queue beyond the camera view | Queue length is capped at the visible length | Reported through `queue_exceeds_view_fraction`. An upstream camera is the real fix. |
| Calibration quality | Every metric quantity scales with calibration error | Reprojection error is stored and reported. Use the bird's-eye view check. Validate speed against a known distance or a GPS probe vehicle. |

Pixel movement is **never** reported as speed. Without a valid calibration the
pipeline refuses to start (`PerceptionPipeline.build` raises), and outside the
calibrated area speed is None, not a guess.

---

## 8. Calibrating a camera

1. Put your video at `data/raw/cam_01.mp4` (or set `source`), then write out a
   frame:
   `python -m traffic_intel.computer_vision.calibration_tool cam_01 --frame 100`.
2. Open `runs/calibration/cam_01/frame.png` in any image viewer and read the
   pixel coordinates of **6–8 road-surface points with known positions**:
   - stop-line ends
   - lane-marking corners (measure the dash length and gap on site, or on an
     aerial image with a scale bar)
   - kerb corners

   Spread them over the whole region where you want speeds, including the far
   end.
3. Choose a ground frame, for example origin at the left end of the stop line
   with `+y` pointing upstream. Enter `image_points_px`, `ground_points_m`,
   `image_size`, `origin_lonlat` and `bearing_deg` in the camera YAML.
4. Draw the zones in the same pixel coordinates:
   - `roi`
   - `lane` polygons, with `direction` in ground coordinates
   - `entry_line`, and `exit_line` at the stop line
   - `queue_zone`
   - `virtual_loop`, a strip just behind the stop line
5. Run the tool again and check:
   - `reprojection_error_m` is below about 0.3 m
   - in `overlay.png`, the 5 m grid lines match the lane markings
   - in `birdseye.png`, lanes are straight and parallel with the right width
   - `anchor_reference` matches the traffic direction
6. **Validate speed**: drive or observe a vehicle over a measured distance, or
   use a GPS probe, and compare with the `tracks.jsonl` speeds.

---

## 9. Accuracy on the synthetic dataset

The synthetic scene (`traffic_intel/synthetic/`) is built as follows:

- **Simulation:** Intelligent Driver Model car-following in 2 lanes, with a
  fixed-time signal, Poisson arrivals, and a class mix of 60% car, 22%
  motorcycle, 12% truck and 6% bus, plus pedestrians.
- **Camera:** vehicle footprints (with height) are projected through a
  perspective camera.
- **Detector noise:** 1.5 px box jitter, 4% missed detections, 3% car/truck
  flips, 8% low-confidence frames, and 1 px calibration-picking noise.

Estimates against ground truth per 60 s window, computed by
`python -m traffic_intel.pipeline synthetic --scenario …`:

| Scenario | Window | Vehicles est/true | Space-mean speed km/h | Outflow | Queue m | State |
|---|---|---|---|---|---|---|
| free_flow | 1 | 11 / 11 | 42.3 / 41.6 | 9 / 9 | 0 / 0 | FREE |
| free_flow | 2 | 9 / 9 | 41.1 / 40.6 | 9 / 9 | 0 / 0 | FREE |
| free_flow | 3 | 13 / 13 | 41.4 / 40.9 | 13 / 13 | 0 / 0 | FREE |
| default (signalised) | 1 | 31 / 31 | 11.4 / 11.1 | 19 / 19 | 20.5 / 21.0 | SEVERE (0.75) |
| default | 2 | 26 / 26 | 12.9 / 13.5 | 19 / 19 | 11.4 / 10.1 | SEVERE (held by hysteresis) |
| default | 3 | 19 / 19 | 12.1 / 12.3 | 13 / 13 | 9.9 / 9.2 | HIGH |
| congested | 1 | 37 / 36 | 4.7 / 4.8 | 17 / 17 | 36.4 / 34.2 | SEVERE |
| congested | 2 | 30 / 30 | 4.0 / 4.5 | 19 / 19 | 39.5 / 34.9 | SEVERE |
| congested | 3 (standstill) | 22 / 18 | 1.1 / 0.5 | 0 / 0 | 60.8 / 59.3 | SEVERE |

In summary:

- **Counts** are exact except in a far-field standstill (see §7).
- **Flow** is exact in every window.
- **Speed** is within 0.7 km/h.
- **Queue** is within 1.5 m, except 4.6 m in one congested window.

These numbers validate the geometry, tracking and feature logic. They do
**not** measure detector recall on real video; that needs labelled clips of
real footage (Architecture §14).

The tests (`tests/backend/unit/test_feature_engine.py`) enforce, on the
`default` and `free_flow` scenarios:

- counts within ±1
- outflow within ±1
- speed within ±1.5 km/h
- queue within ±3 m

They also enforce that `free_flow` is FREE, `congested` is SEVERE, and every
free-flow score is below every congested score.

**Committed test dataset:** `tests/fixtures/synthetic_small/` contains
`camera.yaml`, `detections.csv`, `truth.csv` and `scene.json`: 20 s at 10 fps,
about 150 KB. A test checks that the seeded generator still reproduces it
exactly.
