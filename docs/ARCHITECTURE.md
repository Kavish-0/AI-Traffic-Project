# AI Urban Traffic & Incident Intelligence: Architecture

Status: **v0.1 design, skeleton only**. No models are implemented yet. The
Phase column in §16 says when each part gets built.

The system answers three questions for every monitored road approach:

| Question | Output | Module |
|---|---|---|
| What is happening now? | density, speed, flow, queue, congestion level | 1-3 |
| What will happen next? | +5 / +15 / +30 min forecasts | 4 |
| Is something abnormal? | anomaly signals, then a fused incident probability | 5 |

These are combined per approach (6) into a risk score, shown on a map (7), and
used by a what-if simulator (8).

---

## 1. Design principles

1. **Interfaces before implementations.** Each replaceable stage (detector,
   tracker, projector, feature engine, classifier, forecaster, anomaly
   detector, fuser, risk engine, simulator) is an abstract class with a
   `Registry`. `configs/pipeline.yaml` chooses the implementation, so
   replacing YOLO means adding one adapter module and changing one config line.
2. **Frame rate and window rate are separated.** Per-frame data (30 fps × N
   cameras) stays inside the camera worker. Only per-window aggregates (1 s
   live, 60 s persisted) cross process boundaries. This one decision decides
   whether the system scales.
3. **Metric units, not pixels.** Every speed, distance and queue length comes
   from ground-plane metres via a calibrated homography, and the calibration
   error is stored with it.
4. **Honest outputs.** "Incident probability" is a *likelihood estimate from
   combined evidence*, flagged `calibrated=false` until it has been fitted on
   labelled events. The system never states that an accident happened.
5. **Baselines first.** Rule-based congestion, persistence/seasonal-naive
   forecasting, and threshold/EWMA anomaly detection are built before any
   learned model. Every learned model is reported *relative to* these
   baselines.
6. **Training and inference share code, not processes.** Both use one
   feature schema (`core/feature_schema.py`). Training runs offline via a CLI.
   The online pipeline only loads versioned artifacts.
7. **Reproducible offline replay.** A recorded video processed in replay mode
   produces the same output as the same video "live". Time comes from the
   source, not the wall clock.
8. **Small, then distributed.** One process with an in-memory bus is enough
   for development and demos. Redis and separate workers are added only when
   several cameras need several processes or GPUs.

---

## 2. System context and deployment modes

```
                ┌────────────── external ──────────────┐
  Cameras ─────►│ RTSP / HTTP streams, or video files  │
  Open-Meteo ──►│ weather (live + historical archive)  │
  holidays ────►│ calendar                             │
                └──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│ backend (Python package: traffic_intel)                                  │
│                                                                          │
│  Camera worker(s)  ──bus──►  Analytics worker  ──bus──►  API (FastAPI)   │
│  (CV, GPU)                   (window-rate ML)            REST + WS       │
│        │                            │                        │           │
│        ▼                            ▼                        │           │
│  data/interim (Parquet)       PostgreSQL + PostGIS ◄─────────┘           │
│        │                            │                                    │
│        └──────► ai.training (offline CLI) ──► models/ (versioned)        │
└──────────────────────────────────────────────────────────────────────────┘
                                   │ HTTP (REST polling)
                          React dashboard (map, charts)
```

| Mode | Processes | Bus | Use |
|---|---|---|---|
| **local (implemented)** | 2: pipeline runtime (camera threads + analytics, `python -m traffic_intel.pipeline run`) and API (`uvicorn`, no torch) | in-memory asyncio inside the runtime; the **database** hands data to the API; the dashboard **polls the REST API** (no WebSocket) | development, demos, tests - current scope |
| **replay** | 1 (`pipeline replay`), no pacing | in-memory | dataset generation from recorded video |
| **distributed (future)** | 1 API, 1 analytics, 1 per camera or GPU | Redis Streams | several live cameras - see [DEPLOYMENT_FUTURE.md](DEPLOYMENT_FUTURE.md) |

Everything is one Python package with several entry points
(`uvicorn traffic_intel.api.main:app`, `python -m traffic_intel.pipeline …`,
`python -m traffic_intel.ai.training …`). It is a modular monolith, not
microservices. For an academic project with one team, microservices would
cost operational complexity and give no benefit.

---

## 3. Components

### 3.1 Perception (Module 1): `computer_vision/`, `tracking/`

Implemented in Phases 1-2. Full details, assumptions and limitations are in
[TRAFFIC_STATE_ESTIMATION.md](TRAFFIC_STATE_ESTIMATION.md).

| Stage | Interface | Implementation | Notes |
|---|---|---|---|
| Video input | `VideoSource` | `OpenCVVideoSource`: file, `webcam:N`, RTSP/HTTP; `SyntheticVideoSource` | Resizing to `resolution` and sampling to `process_fps` on a fixed time grid happen here. Live sources use a latest-frame buffer and reconnect; files use `replay_start + index/fps` and never drop frames. |
| Image processing | `FramePreprocessor` (chain) | `roi_mask`, `clahe`, `undistort` | Must preserve calibrated geometry. Zones and calibration rescale automatically to the processing resolution. |
| Detection | `Detector` | `yolo` (Ultralytics), `precomputed` (CSV replay), `synthetic` | Configurable `ClassMapper` works by label or model id, with per-class confidence. Batch interface. |
| Tracking | `Tracker` | Native ByteTrack; BoT-SORT = ByteTrack + optical-flow camera-motion compensation | One stateful instance per camera. Explicit NEW → TRACKED → LOST → REMOVED lifecycle; IDs never reused; class decided by majority vote. |
| Projection | `GroundProjector` | Planar homography, sign-normalised, with reprojection error and a calibrated-region test | Anchor = bottom-centre of the box. Local metres map to WGS84 via origin and bearing. |
| Trajectories | `TrajectoryStore` | Per-track bounded history | Least-squares velocity (plus a long window near standstill), direction, distance, stopped duration. `TrackedObject` view and `TrackSummary` on finish. |

`PerceptionPipeline` (`pipeline/perception.py`) wires the stages from
`configs/pipeline.yaml`. Any stage can be injected. It emits `TrackedFrame`s,
window `TrafficFeatures` and `TrafficState`s, a live `TrafficState` about every
1 s, and finished `TrackSummary`s. `CameraWorker` (`pipeline/camera_worker.py`)
runs the pipeline for one camera. It writes the annotated video and Parquet
(detections, tracks, trajectories) locally and publishes **only** the 1 s live
values and 60 s windows, enforced by `publish.WorkerPublisher`. GPU setup and
the public development footage are covered in
[PHASE1_PERCEPTION.md](PHASE1_PERCEPTION.md).

### 3.2 Traffic feature engine (Module 2): `traffic/features/`

Stateful for each camera. Every processed frame is reduced to a `FrameSample`
(objects in the ROI with speed, stopped and queued flags, lane, distance
upstream of the stop line) plus line-crossing events. Pluggable
**`FeatureCalculator`s** turn samples into metrics:

- counts
- speed
- density
- occupancy
- flow
- stopped vehicles
- queue
- gaps
- trajectory statistics

New features are new registered calculators; their outputs go to
`TrafficFeatures.extra` until promoted to columns. The engine emits:

- `live_features()`: the same calculators over a rolling 5 s window, about
  1 Hz, for the dashboard.
- `close_windows()`: fixed 60 s `TrafficFeatures`, aligned to source time.
  Empty windows are emitted with `quality = 0`. These are persisted, feed every
  ML model, and are the system's canonical time series. `flush()` emits the
  final partial window.
- `finished_tracks()`: `TrackSummary` for each completed trajectory (used by
  trajectory anomaly models and turning-movement counts).

`traffic/state.py` merges the features with the congestion verdict into the
flat **`TrafficState`** output: timestamp, junction and road IDs, per-class
counts, speed, density, occupancy, flow, queue, stopped count, traffic state,
score and explanation.

### 3.3 Congestion classification (Module 3): `traffic/congestion/`

`CongestionClassifier.classify(features, context) -> CongestionAssessment`.

- **Rule-based baseline** (implemented; `configs/congestion_rules.yaml`):
  - speed ratio, density ratio, occupancy and queue ratio are normalised,
    weighted and thresholded into FREE, MODERATE, HIGH or SEVERE
  - override rules can only raise the level; they are boolean expressions
    evaluated with a whitelisted AST, never `eval`
  - hysteresis stops the level flickering
  - every indicator, override and hold is written to `explanation`
- **ML** (Random Forest, XGBoost): trained on *manually labelled* windows.
  Labels produced by the rules can be used only for pre-training, because
  testing against rule labels only measures how well the model copies the
  rules.

### 3.4 Forecasting (Module 4): `forecasting/`

`Forecaster.fit()` runs offline and `Forecaster.predict()` runs online. The
input is 60 min of 1-min features plus context. The output is
density ratio, speed and flow at +5, +15 and +30 min.

| Model | Family | Role |
|---|---|---|
| Persistence, seasonal-naive | non-learned | **mandatory baselines** |
| Ridge linear regression | tabular (lag features) | simplest learned model |
| Random Forest, XGBoost | tabular (lags, rolling stats, calendar, weather) | usually strong on small data |
| LSTM / GRU | sequence | test whether temporal modelling helps |
| Temporal Transformer | sequence | test whether attention helps at this data scale |

Tabular models use direct multi-horizon strategy: one model per horizon, or
multi-output. Sequence models output all horizons together. The evaluation
protocol is in §14.

### 3.5 Anomaly and incident (Module 5): `anomaly/`, `incidents/`

**Implemented (Phase 6).** Full description, formulas and results: [INCIDENT_INTELLIGENCE.md](INCIDENT_INTELLIGENCE.md).

These are layers of **evidence**, followed by **fusion**:

```
Track-level (camera worker, every frame)   Series-level (analytics, every 60 s window)
  stopped_vehicle (outside queue zone)       sudden_slowdown  EWMA baseline + robust z (mean AND p85)
  wrong_way (vs configured lane direction)   queue_growth     slope + rise vs median
  trajectory_rules (swerving, off-lane)      density_zscore   median/MAD, minus stopped vehicles
                                             flow_imbalance   inflow >> outflow
                                             ewma / zscore    control charts
                                             isolation_forest / autoencoder (learned, optional)
           │                                          │
           └────────── AnomalySignal (score 0..1) ────┘
                              │
                 WeightedLogisticFuser (per approach)
       S = noisy-OR of symptom families (slowdown, queue, density, flow)
       p = σ(−4 + 2.5·S + 2.5·stopped + 2.0·trajectory + 1.0·outlier + 3.0·S·max(stopped, trajectory))
       configurable taxonomy -> normal | sudden_slowdown | stopped_vehicle | wrong_way |
                                abnormal_queue | abnormal_trajectory | possible_incident
       lifecycle: same id while active, type escalates (never silently downgrades), resolves after 3 quiet windows
                              │
     IncidentAssessment {probability, calibrated=false, type, label, hedged message, evidence,
                         severity, risk_level, junction/road, location, contributing signals}
```

`incidents/engine.py::IntelligenceEngine` runs series detectors → fusion → (forecast) → risk
per closed window. It is driven by the bus (`pipeline/analytics_worker.py`), by the `ingest`
CLI, by the synthetic scenarios and by the tests. Results go to the database through `database/store.py` (`TrafficStore`, SQLite in dev / PostgreSQL + PostGIS in prod).

Signalised junctions produce a queue every red phase, and that is normal. Stopped-vehicle rules
ignore vehicles inside `QUEUE_ZONE` until a very long wait (`max_queue_wait_s`). Series detectors
compare against recent windows (a learned time-of-day profile is future work).

**Calibration path.** The weights start hand-set (`calibrated=false`, stated in every message).
Operator feedback (`PATCH /incidents/{id}` with `true_incident` or `false_alarm`) and annotated
clips provide labels. Logistic regression is then fitted to them, followed by Platt or isotonic
calibration. After that, `calibrated=true`.

### 3.6 Risk engine: `risk/`

**Implemented (Phase 6).** A transparent weighted score from 0 to 100 (`risk/weighted.py`,
`configs/risk.yaml`):

```
base = Σ wᵢ·cᵢ / Σ wᵢ over available components
       current_severity .20, forecast_severity .20, incident .30, queue .10, speed .10, density .10
risk = 100 · base · (0.7 + 0.3 · road_importance)
level: low < 25 ≤ medium < 50 ≤ high < 75 ≤ critical
```

Safety floors: wrong-way → at least HIGH; incident probability ≥ 0.85 → at least HIGH. Every result
carries `reason_codes` (e.g. `SEVERE_CONGESTION_FORECAST`, `STOPPED_VEHICLE`, `WRONG_WAY_VEHICLE`)
and the formula with its numbers, so the UI can show *why* a road is high risk.

### 3.7 Junction intelligence (Module 6)

The unit of analysis is the **approach**: one road entering one junction.

```
Junction ──1:N── Approach ──1:1── RoadSegment (directed graph edge, map polyline)
                     │
                     └──1:N── Camera ──1:N── CameraZone, CameraCalibration(versioned)
```

Each camera belongs to exactly one approach. An approach normally has one
camera; supporting several per approach is a later extension that merges
features. Junction-level views aggregate approaches, for example worst level,
total inflow, and "queue spill-back risk" when an approach's queue nears its
segment length.

### 3.8 Map (Module 7)

PostGIS stores the segments as `LINESTRING` (EPSG:4326). `GET /network`
returns a GeoJSON FeatureCollection with each segment's latest level, density,
speed and risk in `properties`. MapLibre draws the lines with a data-driven
colour (green, yellow, orange, red, and grey when there is no data). Clicking
a road calls `GET /approaches/{id}/snapshot` to fill the Current, Prediction
and Incident panels; the snapshot is then polled every few seconds for its live
values. `GET /network` is refetched when the dashboard's live poll of
`/traffic/current` detects a newly closed window, so road colours update
without a page reload.

### 3.9 What-if simulation (Module 8): `simulation/`

**Implemented (Phase 6)** in `simulation/static_assignment.py`. This is a static traffic assignment
on a small NetworkX `DiGraph`. It is **not** a microsimulator.

1. Baseline: observed flow on each edge (latest stored windows); `assumed_vc × capacity` where no
   camera exists. Capacity = `capacity_veh_h` or lanes × 900 veh/h.
2. Scenario: remove closed edges and scale capacities (`capacity_factors`, `demand_factor`).
3. Reassign the displaced flow between the closed/reduced edge's endpoints over the edge itself
   (if still open) and the k = 3 shortest alternative paths. Use BPR travel time
   `t = t0·(1 + 0.15(v/c)^4)` and iterate with the method of successive averages (50 iterations).
4. Map each edge's v/c to a congestion probability, `1/(1+e^(−10(v/c−0.85)))`. A junction's
   probability is the flow-weighted mean over the edges entering it.
5. Return the change in flow per segment, the congestion probability before and after (segments
   and junctions), the demand that could not be served, summary lines and the **explicit list of
   assumptions**.

The graph only covers what the network config contains: monitored approaches plus a few connecting
links. The core limitation is that flows are observed only where there are cameras. Other links
need assumed or estimated flows, and results state this.

---

## 4. Data flow

```
                         per frame (≈10 fps / camera)                     per window (60 s)
┌────────────┐  Frame   ┌──────────┐ Detections ┌─────────┐ Tracks ┌───────────┐
│VideoSource ├─────────►│Preprocess├──►Detector─►│ Tracker ├───────►│ Projector │
└────────────┘          └──────────┘            └─────────┘        └─────┬─────┘
                                                                         │ TrackedFrame (+ground_xy)
                           ┌─────────────────────────────────────────────┤
                           ▼                                             ▼
                 TrackAnomalyDetectors                          TrafficFeatureEngine
                           │ AnomalySignal                  live(1 Hz) │    │ TrafficFeatures (60 s)
═══════════════════════════╪══════════════ EVENT BUS ══════════════════╪════╪═══════════════════
                           ▼                                           │    ▼
                ┌──────────────────── Analytics worker (per approach) ─┼────────────────────┐
                │ persist ──► CongestionClassifier ──► SeriesAnomaly   │                    │
                │         ──► Forecaster (every minute)                │                    │
                │         ──► IncidentFuser(signals, neighbours) ──► RiskEngine              │
                └──────────────┬───────────────────────────────────────┼────────────────────┘
                               │ persist                               │ publish
                        PostgreSQL/PostGIS ──► REST API ◄── polled by the Dashboard
```

Bus topics (`pipeline/bus.py::Topics`): `traffic.features.live`,
`traffic.features.window`, `traffic.tracks.finished`, `anomaly.signal`,
`traffic.congestion`, `traffic.forecast`, `incident.update`, `risk.update` and
`camera.status`.

### Clock

Every message carries a **source timestamp**. For live streams this is the UTC
wall clock at capture. For files it is `replay_start + frame_pts`. Windows are
aligned to minute boundaries of source time. As a result, hour and day-of-week
features are correct for replayed footage, and replaying at 10× speed gives
the same windows as real time.

---

## 5. Folder structure

```
AI-Project-Traffic/
├── backend/
│   ├── pyproject.toml            # core deps + extras: cv, ml, distributed, dev
│   ├── Dockerfile, alembic.ini
│   └── traffic_intel/
│       ├── core/                 # NO deps on other packages: enums, types (hot path),
│       │                         # schemas (boundary), config, registry, feature_schema
│       ├── computer_vision/      # VideoSource, FramePreprocessor, Detector, GroundProjector,
│       │   │                     # homography, zones, class_mapping, annotate, calibration_tool
│       │   ├── sources/  detectors/
│       ├── tracking/             # Tracker, Kalman, trajectory (TrajectoryStore)  └── trackers/
│       ├── traffic/              # TrafficFeatureEngine, CongestionClassifier, state (TrafficState)
│       │   ├── features/  congestion/
│       ├── synthetic/            # IDM traffic scene + noisy detections + ground truth (tests, demos)
│       ├── forecasting/          # Forecaster └── models/
│       ├── anomaly/              # Track/Series detectors, IncidentFuser └── detectors/
│       ├── risk/                 # RiskEngine
│       ├── simulation/           # TrafficSimulator (NetworkX)
│       ├── context/              # WeatherProvider (open_meteo, none), CalendarProvider (holidays)
│       ├── incidents/            # IntelligenceEngine (per-window analytics), scenarios, CLI
│       ├── datasets/             # dataset manifest + adapters (UA-DETRAC, AI City, METR-LA, ...)
│       ├── ai/                   # model_store (TrainableModel, ModelStore),
│       │   ├── training/         # offline CLI + dataset builders
│       │   └── evaluation/       # metrics, walk-forward harness, reports
│       ├── pipeline/             # perception (per-camera pipeline), runner, sinks, event bus,
│       │                         # camera_worker, analytics_worker, runtime (local full system), CLI
│       ├── database/             # ORM models, portable types, session, store (TrafficStore), migrations/
│       ├── services/             # API use-cases: network, traffic/forecast, incidents, simulation, system
│       └── api/                  # FastAPI app, schemas (API contract), routes/; live hub + websocket (optional, off)
├── frontend/                     # React + TS + Tailwind + MapLibre + ECharts (Vite)
│   └── src/ api/ state/ hooks/ components/ pages/ theme/ lib/ test/
├── configs/                      # pipeline.yaml, rules, network/, cameras/, training/
├── data/                         # raw/ interim/ processed/ external/ labels/  (git-ignored)
├── models/                       # versioned artifacts (git-ignored)
├── infra/db/init/                # PostGIS extension bootstrap
├── tests/backend/ unit/ integration/
├── docs/ARCHITECTURE.md
└── docker-compose.yml
```

**Dependency rule.** `core` depends on nothing. Module packages depend on
`core`, and on `ai` if they have learned models. `pipeline` wires modules
together. `api` depends on `services`, which depends on `core`, `database`, the configuration and
the (metadata-only) model registry. It **never** depends on concrete CV or ML implementations,
so the API process never imports torch. A test enforces this.

---

## 6. Data models

Two families, chosen deliberately:

| Family | Where | Tech | Why |
|---|---|---|---|
| Hot-path types | `core/types.py` | `@dataclass(slots=True)` + NumPy (struct-of-arrays) | Created for every frame. Pydantic validation at 30 fps × N cameras would cost measurable CPU. |
| Boundary schemas | `core/schemas.py` | Pydantic v2 (frozen) | Validated, serialisable, used on the bus, in the API and in repositories. Mirrored in `frontend/src/api/types.ts`. |

Key models: `Frame`, `Detections`, `Tracks`, `TrackedFrame` (hot path).
`DetectionRecord`, `TrackSummary`, `TrafficFeatures`, `ContextFeatures`,
`CongestionAssessment`, `Forecast`/`ForecastPoint`, `AnomalySignal`,
`IncidentAssessment`, `RiskAssessment`, `ApproachSnapshot`,
`SimulationRequest`/`SimulationResult`, `BusMessage` (boundary).
`ModelMetadata` (artifacts). Config models in `core/config.py`
(`NetworkConfig`, `CameraConfig`, `CalibrationConfig`, `ZoneConfig`,
`PipelineConfig`).

Identifiers are readable slugs from config (`cam_01`, `junction_a_north`,
`road_b_wb`), and the same key is used in configs, DB, logs and UI.

---

## 7. Module interfaces

| Interface | Key method(s) | Rate | State |
|---|---|---|---|
| `VideoSource` | `frames() -> Iterator[Frame]` | frame | stream |
| `FramePreprocessor` | `process(Frame) -> Frame` | frame | none |
| `Detector` | `detect(Sequence[Frame]) -> list[Detections]` | frame (batched) | none |
| `Tracker` | `update(Detections, Frame) -> Tracks`, `reset()` | frame | per camera |
| `GroundProjector` | `image_to_ground`, `ground_to_lonlat` | frame | calibration |
| `TrafficFeatureEngine` | `update(TrackedFrame)`, `live_features()`, `close_windows()`, `finished_tracks()` | frame in → window out | per camera |
| `TrackAnomalyDetector` | `update(TrackedFrame, approach) -> [AnomalySignal]` | frame | per camera |
| `CongestionClassifier` | `classify(TrafficFeatures, Context) -> CongestionAssessment` | window | none/model |
| `Forecaster` (`TrainableModel`) | `fit(train, valid)` offline, `predict(approach, history, future_ctx, t)` online | minute | model |
| `SeriesAnomalyDetector` | `score(approach, history) -> [AnomalySignal]` | window | model |
| `IncidentFuser` | `fuse(approach, t, signals, features, neighbours, open) -> IncidentAssessment?` | window | open incidents |
| `RiskEngine` | `assess(features, congestion, forecast, incident) -> RiskAssessment` | window | none |
| `TrafficSimulator` | `load_network()`, `what_if(request, baseline) -> SimulationResult` | on request | graph |
| `WeatherProvider` / `CalendarProvider` | `observed`, `forecast` / `is_holiday` | minute (cached) | cache |
| `ModelStore` | `save`, `load_production`, `list`, `promote` | startup | filesystem |
| `EventBus` | `publish(BusMessage)`, `subscribe(*topics)` | window | backend |
| Repositories | `ObservationRepository`, `AnalyticsRepository`, `IncidentRepository` | window | DB |

Every interface has a `Registry`, and planned implementations are declared
with `register_lazy(name, module)`. Heavy dependencies therefore load only
when that implementation is selected.

---

## 8. Database schema (PostgreSQL + PostGIS in production, SQLite locally)

Defined in `backend/traffic_intel/database/models.py` and used through
`database/store.py` (`TrafficStore`, the application's only persistence API). The column types
are portable (`database/types.py`). Geometry is PostGIS `geometry(…, 4326)` when the connected
PostgreSQL has PostGIS, and WKT text otherwise. JSON becomes JSONB on PostgreSQL. The same
models therefore run on a zero-setup SQLite file (`data/db/traffic.db`, the default), on plain
PostgreSQL and on PostGIS (`TI_DATABASE_URL`). Tables are created on start-up
(`create_all`), and the network and cameras are synced from `configs/`. Versioned Alembic
migrations (`database/migrations/`) are the path once the schema is frozen for deployment.

**Static network and cameras** (geometry EPSG:4326)

| Table | Key columns |
|---|---|
| `junction` | id, name, signalized, geom POINT |
| `road_node` | id, junction_id?, geom POINT |
| `road_segment` | id, from_node_id, to_node_id, lanes, speed_limit, free_flow_speed, capacity_veh_h, length_m, geom LINESTRING |
| `approach` | id, junction_id, road_segment_id, direction |
| `camera` | id, approach_id, source_uri, process_fps, active, geom POINT |
| `camera_calibration` | camera_id, **version**, homography 3×3, point pairs, origin, bearing, reprojection_error_m, valid_from |
| `camera_zone` | id, camera_id, type (roi, lane, entry_line, exit_line, queue_zone, virtual_loop), points_px, direction, geom |

**Time series** (composite PK on entity + time; monthly range partitions from Phase 3)

| Table | PK | Content |
|---|---|---|
| `traffic_observation` | (approach_id, window_start) | every `TrafficFeatures` field + calibration_version |
| `context_observation` | (region, ts) | weather, temperature, precipitation, visibility, holiday |
| `congestion_state` | (approach_id, ts, classifier) | level, score, explanation; several classifiers can run side by side |
| `forecast` | (approach_id, issued_at, model_name, horizon_min, target) | value, interval, target_time. Storing forecasts lets them be scored later against what actually happened. |
| `risk_assessment` | (approach_id, ts) | score, level, components, reason_codes, per-window incident_score |
| `approach_live_state` | approach_id | latest ~1 s live value (upserted - the live stream is not stored as a series) |
| `camera_status` | camera_id | heartbeat from the runtime: state, frames, fps, last frame, detector/device, preview path, error |

**Events**

| Table | Content |
|---|---|
| `anomaly_event` | type, scope, detector, score, track_ids, details, geom, incident_id? |
| `incident` | status, probability (score), max_probability, calibrated, severity, type, label, message, evidence, contributing signals, risk_level, junction/road, geom, first_detected, last_updated, resolved_at, **operator_label** (the label source) |
| `model_registry` | task, name, version, framework, artifact_uri, feature_schema_hash, metrics, status |
| `simulation_scenario` | request, result |

**Deliberately not stored in Postgres:** frame-level detections. At
10 fps × 20 objects × N cameras that would be millions of rows per day with
almost no query value. They go to partitioned Parquet (`data/interim/`), which
suits pandas and training well. If time-series volume grows, TimescaleDB is a
drop-in extension on the same schema.

---

## 9. API boundaries

REST is served at `/api/v1`, with OpenAPI at `/docs`. The routes are thin; they call the
service layer (`backend/traffic_intel/services/`), which reads the store, the configuration and
the model registry. Responses are typed (`api/schemas.py`), separate from internal domain
objects. Unavailable data is explicit: a `status` or `message` field, or `null`. It is never a
placeholder value.

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | liveness + database connectivity |
| GET | `/system/status` | backend, database, pipeline, forecast model, cameras, weather (real signals), WS clients |
| GET | `/network` | GeoJSON: roads (latest **fresh** level, risk, incident), junctions, cameras, incident points |
| GET | `/junctions`, `/junctions/{id}` | approaches with level, risk, cameras, active incident |
| GET | `/roads`, `/roads/{road_id}` | road list; the road's full panel (snapshot of its approach) |
| GET | `/approaches/{id}/snapshot` | live + window state, forecast (with status), incident, incident score, risk, cameras, synthetic flag |
| GET | `/approaches/{id}/history?minutes=` | closed 60 s windows |
| GET | `/approaches/{id}/forecast` | forecast with explicit status (`ok`, `insufficient_history`, `no_model`, `stale`, `no_data`) |
| GET | `/approaches/{id}/forecast/history?horizon_min&target` | forecast vs actual |
| GET | `/traffic/current`, `/traffic/summary`, `/traffic/{id}/history` | latest states; KPIs from fresh live values only |
| GET | `/anomalies`, `/incidents/anomalies/recent` | anomaly signals (evidence) |
| GET | `/incidents`, `/incidents/{id}` | hedged incidents with `score_label`, evidence, uncertainty note |
| PATCH | `/incidents/{id}` | operator feedback (acknowledge / resolve / dismiss + TRUE INCIDENT / FALSE ALARM label) |
| GET | `/risk`, `/risk/{id}/history` | risk with reason codes and formula |
| GET | `/cameras`, `/cameras/{id}`, `/cameras/{id}/status` | config + heartbeat status |
| GET | `/cameras/{id}/frame.jpg` | latest annotated frame (≈1/s preview written by the camera worker; 404 = unavailable) |
| POST | `/simulation/what-if`, GET `/simulation/scenarios/{id}` | `SimulationResult` (NetworkX static assignment) |
| GET | `/models`, `/models/{task}/evaluation` | registry (metadata only); latest experiment metrics |
| PUT | `/cameras/{id}/calibration`, `/cameras/{id}/zones` | 501 - calibration is edited in `configs/cameras/*.yaml` (future) |

**How the dashboard stays current: REST polling** (frontend `src/state/live.tsx`, `src/config.ts`).

| Interval (default) | Endpoints | Why |
|---|---|---|
| live, 3 s (`VITE_POLL_LIVE_MS`) | `/traffic/summary`, `/approaches/{selected}/snapshot`, `/traffic/current` | live values change every second; `/traffic/current` (latest window per approach) is the change detector |
| on a detected new window, else 30 s (`VITE_POLL_SLOW_MS`) | `/network`, `/junctions`, `/incidents`, `/risk`, `/approaches/{id}/history`, `/risk/{id}/history`, `/approaches/{id}/forecast/history` | these only change when a 60 s window closes |
| status, 10 s (`VITE_POLL_STATUS_MS`) | `/cameras`, `/system/status` | heartbeats every 2 s, offline after 15 s |
| 5 min | `/roads` | near-static |

Change detection is per approach (a lagging camera's new window still counts). Polling pauses
while the tab is hidden, a tick is skipped while the previous request is still running, and a
failed poll shows DISCONNECTED / "not updating" with the last values clearly marked, never
invented ones. Frame-level data is never exposed over HTTP.

**Optional push channel (future interface, off by default).** `TI_WS_ENABLED=true` mounts
`/ws/traffic` (alias `/ws/live`) and starts the LiveHub, which polls the database and pushes
compact `traffic.update`, `incident.update` and `camera.status` messages
(`{"action":"subscribe","topics":[…],"approach_ids":[…]|null}`). The current frontend does not
use it. The intended future path is camera workers → Redis Streams → backend → WebSocket → React
([DEPLOYMENT_FUTURE.md](DEPLOYMENT_FUTURE.md)).

**Not exposed:** training and model promotion. These are offline CLI actions by design.

---

## 10. Real-time processing

**Camera worker** (thread in dev mode, process in distributed mode):

```
capture thread ──► 1-slot "latest frame" buffer ──► processing loop (sampled to process_fps)
                                                    detect → track → project → features → signals
                                                    ──► thread-safe queue ──► async bus publisher
```

- **Backpressure.** Live sources overwrite the single-slot buffer, so a slow
  model lowers the effective fps instead of building up latency. Dropped
  frames are counted in `camera.status`. Replay sources block instead, so
  every frame is processed.
- **Effective fps is known** for each frame (timestamps come from the source),
  so speeds stay correct even when frames are dropped.
- **GPU sharing (Phase 9).** One detector service batches frames from all
  cameras (`detect(Sequence[Frame])`).
- **Latency budget** (per camera, 10 fps): detection ≤ 60 ms (GPU, 960 px),
  tracking ≤ 5 ms, projection and features ≤ 5 ms. End-to-end
  capture-to-dashboard ≤ 2 s for live features and ≤ 5 s for incidents.
  Each stage's latency is measured into `TrackedFrame.processing_ms`.
- **Analytics worker** is a single async loop. Per-window work is small
  (milliseconds per approach). It keeps an in-memory ring buffer of the last
  ~2 h of windows per approach and backfills from the DB at startup, so
  forecasters never query the DB on the hot path.
- **Failure handling.** Reconnect on stream loss with backoff, and
  `tracker.reset()` after a reconnect. Windows with little valid video have
  low `quality`. Models skip or flag low-quality windows, and the UI greys
  them out instead of showing zeros.

---

## 11. Training vs inference

```
                OFFLINE (ai.training CLI)                              ONLINE (pipeline)
 data/interim Parquet ─┐
 traffic_observation ──┼─► build-dataset ─► data/processed/*.parquet
 context_observation ──┘   (feature_schema)          │
 data/labels/*.csv ────────────────────────┐         ▼
                                           ├─► train ─► evaluate (walk-forward, vs baselines)
                                           │               │
                                           │               ▼
                                           │    models/<task>/<name>/<ver>/ + metadata.json
                                           │               │  promote (manual, after review)
                                           │               ▼
                                           │    PRODUCTION pointer ───────────►  ModelStore.load_production()
                                           │                                     (hash check vs feature_schema)
 operator feedback (incident labels) ◄─────┴──────────────────────────────────── PATCH /incidents/{id}
```

- The same `feature_schema` code builds features for training and for
  inference. A model trained on a different schema hash **refuses to load**.
- Scalers and encoders are saved *inside* the model artifact, never
  recomputed online.
- Promotion is a human decision based on the evaluation report, never
  automatic.
- There is no online learning in scope. Retraining is a periodic offline
  job.

---

## 12. Offline vs real-time

| Component | Real-time | Offline |
|---|---|---|
| Video decode, detection, tracking, projection | ✅ | ✅ (replay to build datasets) |
| Feature engine (live + 60 s windows) | ✅ | ✅ (replay) |
| Rule-based congestion, EWMA/z-score, track rules | ✅ | ✅ (backtesting) |
| Forecaster / IF / AE / RF / XGB **inference** | ✅ | ✅ (evaluation) |
| Model **training**, hyper-parameter search, evaluation | ❌ | ✅ |
| Camera calibration, zone drawing | ❌ (API stores results) | ✅ (manual, tool-assisted) |
| Dataset labelling (congestion, incidents, speed ground truth) | ❌ | ✅ |
| Time-of-day baseline profiles (for z-score) | reads | ✅ computes nightly |
| What-if simulation | on demand (seconds) | ✅ scenario studies |
| Weather | cached live fetch | historical archive for training |

---

## 13. Traffic metric definitions

The authoritative definitions are in
[TRAFFIC_STATE_ESTIMATION.md §3–4](TRAFFIC_STATE_ESTIMATION.md#3-metric-definitions),
next to the implementation. In summary:

| Metric | Definition |
|---|---|
| **Spot speed** (per track) | Least-squares slope of ground position over 1 s (3 s near standstill) × 3.6. None when unreliable. |
| **avg_speed_kmh** | *Space-mean* speed: arithmetic mean of instantaneous speeds of the vehicles in the measurement region over all sampled instants. (The v0.1 draft said "harmonic mean". That applies to point detectors such as loops; for spatial snapshots the arithmetic mean *is* the space-mean.) Stopped vehicles are included. |
| **vehicle_count** / **pcu_count** | Unique tracks continuously present in the ROI for ≥ 1 s (and ≥ 3 samples); PCU-weighted sum. |
| **density_veh_km** / **density_ratio** | Time-averaged vehicles (PCU) in the measurement region / (L × lanes); ratio to jam density (150 PCU/km/lane). |
| **occupancy_pct** | Virtual-loop time occupancy; space occupancy if there is no loop. |
| **flow_veh_h**, **inflow**, **outflow** | Direction-checked line crossings, once per track and line; flow from the exit line. |
| **stopped_vehicles** | Time-averaged count of vehicles under 5 km/h for ≥ 2 s. |
| **queue_length_m** | Per lane, a chain of queued vehicles starting ≤ 15 m from the stop line with bumper gaps ≤ 15 m. Length to the tail of the last vehicle; the anchor reference (front or rear bumper) is derived from the geometry. |
| **avg_gap_m** | Bumper-to-bumper gap in the same lane using typical class lengths. |

---

## 14. Evaluation methodology

| Component | Ground truth | Metrics |
|---|---|---|
| Detection | labelled frames (subset of own footage; UA-DETRAC for reference) | mAP@0.5, per-class recall (motorcycles and pedestrians are the hard classes) |
| Tracking | labelled short sequences | MOTA, IDF1, HOTA, ID switches per 100 tracks |
| Counts | manual counts on 5-10 min clips | absolute and % error per class |
| Speed | known marking distances + timing, or a probe vehicle with GPS | MAE km/h, bias; per distance from the camera |
| Congestion classifier | manually labelled windows | macro-F1, confusion matrix vs the rule baseline |
| Forecasting | future observed windows | MAE, RMSE, R² **per horizon and per target**, vs persistence and seasonal-naive. Walk-forward chronological folds, no shuffling, scalers fitted on train only. Report mean ± std over seeds for deep models. Also report peak vs off-peak. |
| Anomaly and incident | annotated clips, operator labels, synthetic injections into real series | precision, recall, PR-AUC, **detection delay**, **false alarms per hour**, reliability diagram once calibrated |

Public datasets are used only where our own data is too small. METR-LA and
PEMS-BAY suit forecasting-model development. The AI City Challenge anomaly
track and CADP suit incident behaviour. Report results on these separately
from results on our own camera data.

---

## 15. Risks and difficult parts

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **Too little data for deep forecasting.** A few hours of video gives a few hundred 1-min windows, but LSTM and Transformer need weeks. | Deep models lose or overfit, and the comparison is meaningless. | Collect long recordings early (Phase 1 onwards, recording always on). Use METR-LA/PEMS-BAY for architecture development. Report honestly that XGBoost may win. |
| 2 | **Speed accuracy.** Homography assumes a flat road. The box bottom is not the ground contact point for tall vehicles. Jitter grows with distance. | Biased speeds, then wrong congestion and slowdown signals. | ≥ 6 well-spread calibration points, reprojection error stored, smoothing, speeds ignored beyond a max reliable distance, validation against ground truth (§14). |
| 3 | **Tracking ID switches and occlusion** in dense or heterogeneous traffic (two-wheelers weaving). | Double counts, spurious speed spikes, false wrong-way alerts. | Line-crossing counts with direction check, minimum track age, per-track speed outlier rejection, BoT-SORT where needed, wrong-way requires ≥ 8 m sustained. |
| 4 | **Incident labels are rare.** | "Incident probability" cannot be calibrated at first. | Publish `calibrated=false` and label it as a score in the UI. Use synthetic injection, annotated public clips and operator feedback to collect labels. |
| 5 | **Signal cycles look like anomalies.** | Periodic false alarms at every red phase. | Time-of-day baselines, windows ≥ one cycle, queue-zone exclusion, hysteresis. |
| 6 | **Compute.** YOLO on CPU runs about 3-8 fps at 640 px. Windows plus CUDA setup can be fragile. | Real-time multi-camera may not be feasible on a laptop. | `process_fps` sampling, smaller models (`yolo11n`), replay mode for development, Docker with CUDA for the GPU path. |
| 7 | **Geo-referencing public footage.** Most online videos have no known location or scale. | The map and speeds become fictional. | Prefer own recordings at a known site. Otherwise mark calibration and location as approximate in the UI and report it. |
| 8 | **What-if validity.** Only camera-covered links have observed flow. | Simulated redistribution depends on assumed demand. | Small graph, explicit assumptions in each result, sensitivity analysis (vary demand ±20%). |
| 9 | **Train/serve skew and leakage** (future weather, scaler fitted on all data, random splits). | Optimistic metrics that do not hold. | One `feature_schema`, schema hash checks, walk-forward splits, only *forecast* weather as a future covariate. |
| 10 | **Privacy.** Video of people and number plates. | Ethical and legal exposure. | Store no raw video beyond the needs of the dataset. No plate or face recognition. Blur in any published clips. Keep only aggregates long term. |
| 11 | **Licensing.** Ultralytics YOLO is AGPL-3.0. | Fine for academic work, and a constraint on commercial distribution. | Keep it behind the `Detector` interface. RT-DETR or an Apache-licensed detector can be dropped in. |
| 12 | **Scope creep.** Eight modules is a lot. | Nothing ends up finished. | Phases below: each ends with a working, demonstrable system. |

---

## 16. Phased development plan

Each phase ends with something that runs, is tested and can be demonstrated.

| Phase | Deliverable | Key work | Exit criteria |
|---|---|---|---|
| **0 Skeleton** ✅ | This repository | Interfaces, schemas, configs, ORM, API stubs, frontend shell, tests | `pytest` green, `npm run build` green |
| **1 Perception** ✅ | Video → tracked detections → annotated video + Parquet | OpenCV source (file/webcam/RTSP, read-ahead decoding), YOLO adapter (CUDA with CPU fallback) + class map, native ByteTrack + BoT-SORT, per-camera worker publishing only live values and windows, Parquet recorder, public-footage catalog | Run on public motorway footage on an RTX 4060 (yolo11m @1280, fp16, 12–14 frames/s end to end for 4K). **Pending:** count and speed error on a labelled real clip. |
| **2 Calibration & traffic state** ✅ | Metric speeds, all Module-2 metrics, 60 s windows, `TrafficState` | Homography projector, headless calibration check tool, zone geometry, pluggable feature engine, rule-based congestion, synthetic IDM dataset | Synthetic accuracy table in TRAFFIC_STATE_ESTIMATION.md §9. **Pending:** speed validated against a known distance on real footage. |
| **3 Persistence, API, live UI** ✅ | Live KPIs in the browser | Alembic migration, config→DB sync, repositories, analytics worker v1 (persist + congestion), WS gateway, dashboard approach panel + charts | Replay a clip and watch KPIs and level update live |
| **4 Junction & map** ✅ | Multi-camera junction on a PostGIS map | Multiple camera workers, `/network` GeoJSON, MapLibre coloured roads, click → snapshot, junction view | 4 approaches coloured correctly from 4 videos |
| **5 Forecasting** ✅ | Model comparison and live forecasts | Dataset builder, baselines, LR/RF/XGB, LSTM/GRU/Transformer, walk-forward harness, model store, evaluation report + UI chart | Report table per horizon vs baselines; best model promoted and serving +5/15/30 |
| **6 Anomaly & incidents** ✅ | Incident probability with evidence | Track rules, EWMA/z-score, Isolation Forest, autoencoder, fusion + hysteresis, incident store + APIs, operator feedback API (incident feed UI: Prompt 5) | Six synthetic scenarios behave logically; PR-AUC, recall, delay and false alarms/h on labelled synthetic series ([INCIDENT_INTELLIGENCE.md](INCIDENT_INTELLIGENCE.md) §9). **Pending:** real labelled incidents. |
| **7 Risk & ML congestion** (risk ✅) | Risk score; RF/XGB congestion vs rules | Risk engine, labelling tool/CSV, classifier training and comparison | Macro-F1 vs rule baseline reported |
| **8 What-if** ✅ (backend) | "Close Road B" → impacts | Graph from network config, BPR + MSA reassignment, UI scenario panel | Plausible, explained results on demo network with sensitivity analysis |
| **9 Hardening** (future scope) | Deployable system | Redis Streams bus, per-camera processes, batched GPU detector, Docker with CUDA, integration tests, load test, docs/report | N cameras at target fps; end-to-end latency within budget |

Phases 5-7 can overlap once Phase 3 is recording data. **Start recording
footage as early as possible**, because the ML phases depend on having hours
to weeks of observations.

---

## 17. Open decisions (to settle when the relevant phase starts)

- Video source: your own recordings at a known junction (preferred), or
  public footage.
- Hardware: whether a CUDA GPU is available. This decides the model size and
  the camera count.
- Country and holiday calendar: `configs/pipeline.yaml` currently says `IN`
  as a placeholder, matching the placeholder coordinates.
- Base map tiles: MapLibre demo tiles for now. Use OpenFreeMap or a
  self-hosted style for real street detail.
- TimescaleDB: adopt only if `traffic_observation` queries become slow.
