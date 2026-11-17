# AI Urban Traffic & Incident Intelligence

Traffic-camera video and context (weather, calendar) are turned into:

- **what is happening now**;
- **what happens next**;
- **whether something is abnormal**;
- **how risky it is**;
- **what happens if a road closes**.

These are shown on a live dashboard for operators.

```
PERCEPTION ──► UNDERSTANDING ──► PREDICTION ──► ANOMALY ──► INCIDENT INTELLIGENCE ──► RISK ──► DECISION SUPPORT
YOLO + tracker   traffic features   +5/+15/+30 min  statistical +   hedged incident          documented    what-if
                 congestion level   forecasting     learned det.    score (uncalibrated)     risk score    simulation
```

| Scope | What |
|---|---|
| **Current scope** | Local development + full AI system + complete UI. Everything runs on one machine; no Docker required. |
| **Future scope** | Deployment, distributed processing, cloud scaling: [docs/DEPLOYMENT_FUTURE.md](docs/DEPLOYMENT_FUTURE.md). Not implemented. |

---

## 1. Architecture and data flow

```
video file / webcam / RTSP / synthetic scene
  │  camera worker (one thread per camera, pipeline runtime process)
  ▼
OpenCV frames → YOLO → ByteTrack/BoT-SORT → trajectories (metric, homography) → traffic feature engine
  │                                   per-frame data stays HERE (Parquet optional; ~1/s annotated preview JPEG)
  ├─ ~1 s live values ─┐
  └─ 60 s windows ─────┤  in-memory event bus (same process)
                       ▼
            analytics worker: series anomaly detectors + track signals → incident fusion (score, type)
                              → forecasting (PRODUCTION model, +5/+15/+30) → risk engine
                       │
                       ▼
            database (SQLite locally / PostgreSQL + PostGIS)    ← network + cameras synced from configs/
                       │
  API process (FastAPI, NO torch): services → REST API
                       │   (the dashboard polls REST - no WebSocket)
                       ▼
  React dashboard: map, road panel, forecast, trends, incidents, risk, camera view, what-if, system status
```

- **How the dashboard gets live data: REST polling** (no WebSocket, no Redis). Every 3 s it polls
  `/traffic/summary` (KPIs), the selected road's `/approaches/{id}/snapshot` (live values) and
  `/traffic/current` (latest closed window per road). When that last one changes, a 60 s window has
  closed, and only then are the per-window panels refetched (map colours, charts, incidents, risk);
  a 30 s safety interval covers the rest. Camera and system status are polled every 10 s. Intervals
  are set in `frontend/.env.local` (`VITE_POLL_LIVE_MS`, `VITE_POLL_STATUS_MS`, `VITE_POLL_SLOW_MS`).
  Polling pauses while the browser tab is hidden. If a poll fails the header shows DISCONNECTED and
  panels say they are not updating; no values are invented.

- **Only aggregates move between components.** Frame rate is not window rate: per-frame
  detections and tracks never reach the database, the API or the browser.
- **Every AI model sits behind an interface**, and `configs/pipeline.yaml` selects the
  implementations.
- **One feature definition** (`core/feature_schema.py`) serves training and live inference. A
  model whose preprocessing hash does not match refuses to load.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 2. Technology stack

| Layer | Technology |
|---|---|
| Perception | OpenCV, Ultralytics YOLO11 (CUDA if available, CPU fallback), native ByteTrack / BoT-SORT, homography |
| Analytics | NumPy / pandas, scikit-learn (Isolation Forest, autoencoder), XGBoost, PyTorch (LSTM / GRU / Transformer; training only), NetworkX (what-if) |
| Backend | Python 3.11+, FastAPI, pydantic v2, SQLAlchemy 2.0 (SQLite / PostgreSQL + PostGIS), in-memory asyncio bus |
| Frontend | React 19, TypeScript, Tailwind CSS 4, MapLibre GL, ECharts, Vite, Vitest + Testing Library |

## 3. Datasets

Each task uses its own dataset; they are never mixed. What is on disk, what each dataset can
legitimately be used for, and why some are not used: [docs/DATASET_INVENTORY.md](docs/DATASET_INVENTORY.md)
(measured quality issues: `reports/data_quality_report.md`). Sources, licences and layout:
[docs/DATASETS.md](docs/DATASETS.md) and `configs/datasets/manifest.yaml`.

| Task | Dataset | Used for |
|---|---|---|
| Detection / tracking | UA-DETRAC | YOLO11 fine-tuning and evaluation; ByteTrack vs BoT-SORT |
| Incidents | AI City 2021 Track 4 | real-video incident evaluation (not deployed: uncalibrated cameras) |
| Forecasting, congestion level | METR-LA, PEMS-BAY, PEMS08 | model-family benchmarks (+ Open-Meteo context variants) |
| Live-domain models | synthetic junction series | the forecaster the pipeline serves; live anomaly evaluation |
| Stored, not used | CityFlowV2-ReID, LISA traffic lights, Szeged weather | no consumer in the system / wrong place for the traffic data |
| Multi-camera / calibration | AI City CityFlowV2 Track 1 | not on disk |
| Demo footage | public highway video | end-to-end validation |

```bash
python -m traffic_intel.datasets inventory   # discover every dataset on disk (content-based)
python -m traffic_intel.datasets quality     # data-quality report
python -m traffic_intel.datasets status      # manifest: present / missing
```

Nothing large is downloaded automatically.

## 4. Run everything locally

**Prerequisites:** Python 3.11+ and Node.js 20+. PostgreSQL is optional. An NVIDIA GPU is
optional; it is used by YOLO if present.

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate                  # Windows;  source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,cv,ml,datasets]"  # cv = YOLO/torch, ml = sklearn/xgboost, datasets = HDF5 readers
# NVIDIA GPU (optional): install the CUDA torch wheel (docs/PHASE1_PERCEPTION.md)
cp ../.env.example ../.env              # optional - every setting has a local default
```

Two processes, in two terminals (both from `backend/`):

```bash
# 1) pipeline runtime: cameras + analytics -> database   (profiles in configs/runtime.yaml)
python -m traffic_intel.pipeline run                     # "demo": 4 synthetic approaches + real public footage (YOLO)
python -m traffic_intel.pipeline run --profile synthetic # CPU only, no video needed
python -m traffic_intel.pipeline run --profile public    # real footage only (needs the downloaded video)

# 2) API (REST; never loads torch)
uvicorn traffic_intel.api.main:app --reload              # http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev                             # http://localhost:5173  (proxies /api to :8000)
```

### Database

- **Default: none needed.** A SQLite file is created at `data/db/traffic.db`, and the network
  and cameras are synced from `configs/`.
- **PostgreSQL / PostGIS (optional).** Create a database and set
  `TI_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/traffic` in `.env`.
  Tables are created on start; PostGIS geometry is used automatically when the extension is
  installed, and WKT text otherwise.
- **Docker (optional).** `docker compose up -d db` starts PostGIS with the credentials from
  `docker-compose.yml`.

### What to expect on first start

- **Synthetic scenes are generated once**, then cached in `data/synthetic/cache`.
- **The first 60 s window appears after about a minute.** Until then the dashboard says
  "Waiting for traffic data…".
- **Forecasts need 30 consecutive 1-min windows**, because the production model's look-back is
  30 minutes. Before that the dashboard shows "Forecast unavailable - insufficient historical
  data (n/30)".
- **Synthetic approaches are badged "synthetic test data" everywhere.** They demonstrate the
  pipeline and are not real-world measurements.

## 5. Environment variables

All are optional. See [.env.example](.env.example) and [frontend/.env.example](frontend/.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `TI_DATABASE_URL` | SQLite `data/db/traffic.db` | SQLAlchemy URL (PostgreSQL / PostGIS in production) |
| `TI_LOG_LEVEL`, `TI_LOG_FORMAT` | `INFO`, `text` | `json` = one JSON object per line |
| `TI_CONFIG_DIR`, `TI_DATA_DIR`, `TI_MODELS_DIR` | repo `configs/`, `data/`, `models/` | locations |
| `TI_LIVE_STALE_S`, `TI_WINDOW_STALE_S`, `TI_CAMERA_OFFLINE_S` | 10, 180, 15 | when data is shown as stale / a camera as offline |
| `TI_WS_ENABLED` | `false` | optional future push channel `/ws/traffic`; the dashboard does not use it |
| `TI_API_CORS_ORIGINS` | `["http://localhost:5173"]` | allowed browser origins |
| `VITE_BACKEND_URL` | `http://localhost:8000` | where the dev server proxies `/api` |
| `VITE_POLL_LIVE_MS`, `VITE_POLL_STATUS_MS`, `VITE_POLL_SLOW_MS` | 3000, 10000, 30000 | dashboard REST polling: live values, camera/system status, per-window safety refresh |
| `VITE_MAP_STYLE_URL` | unset → offline schematic map | any MapLibre basemap style URL |

Cameras, road network, detector, thresholds, runtime profiles and model choices live in
`configs/*.yaml`. Nothing is hard-coded, and no secrets are in Git.

## 6. Models: setup, training, evaluation

All training is offline (`cd backend; python -m traffic_intel.ai.training ...`). Every run
writes an experiment record, and models are registered as CANDIDATE. `configs/pipeline.yaml` names
what serves. Procedures: [docs/ML_TRAINING.md](docs/ML_TRAINING.md). Measured results and the
reasons for each choice: [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md).

| Model | Serving now | Trained / evaluated on | Command (`train <task> --config ../configs/training/<file>`) |
|---|---|---|---|
| Detector | YOLO11m COCO-pretrained (downloaded on first use) | UA-DETRAC: zero-shot vs fine-tuned YOLO11s/m | `detection detection_ua_detrac.yaml` (+ `detection_ua_detrac_rfs.yaml`) |
| Tracker | ByteTrack, thresholds tuned on UA-DETRAC validation | UA-DETRAC: ByteTrack vs BoT-SORT | `tracking tracking_ua_detrac.yaml` |
| Forecaster | `xgboost_d8` trained on the live feature set (synthetic junction) | METR-LA, PEMS-BAY (± calendar & weather), PEMS08, synthetic junction; 8 model types + baselines | `forecasting forecasting*.yaml` |
| Congestion level | explainable rules; future levels = forecast → rules | future-level classifiers on METR-LA, PEMS-BAY, synthetic junction | `congestion congestion_level_*.yaml` |
| Anomaly | statistical detectors, fused (learned ones stay off) | synthetic junction with injected incidents | `anomaly anomaly.yaml` |
| Incident probability | fused evidence, **UNCALIBRATED** | AI City 2021 Track 4 real video (evaluation only) | `incident incident_aic21.yaml` |
| Risk | explainable weighted engine (`configs/risk.yaml`) | no labels exist | - |

```bash
python -m traffic_intel.ai.training summary   # reports/model_evaluation_tables.md, from the records
python -m traffic_intel.ai.training figures   # reports/figures/ (index: reports/figures/INDEX.md)
python -m traffic_intel.ai.training promote forecasting <model> <version>   # or retire
```

More detail: [docs/FORECASTING.md](docs/FORECASTING.md) and
[docs/INCIDENT_INTELLIGENCE.md](docs/INCIDENT_INTELLIGENCE.md).

## 7. REST API

OpenAPI is at `http://localhost:8000/docs`. The main REST routes (prefix `/api/v1`):

- `/health` and `/system/status`
- `/network` (GeoJSON), `/junctions[/{id}]`, `/roads[/{id}]`, `/cameras[/{id}]`,
  `/cameras/{id}/frame.jpg`
- `/approaches/{id}/snapshot|history|forecast|forecast/history`
- `/traffic/current|summary`, `/anomalies`, `/incidents[/{id}]` (PATCH = operator feedback),
  `/risk`
- `POST /simulation/what-if`

The dashboard uses only these REST routes (polling, see §2). Full list:
[docs/ARCHITECTURE.md §9](docs/ARCHITECTURE.md).

A WebSocket push channel (`/ws/traffic`, LiveHub) exists in the backend as an **optional future
interface**. It is off by default (`TI_WS_ENABLED=false`: route not mounted, nothing polls for it) and
the current frontend does not use it. The future path is camera workers → Redis Streams → backend
→ WebSocket → React ([docs/DEPLOYMENT_FUTURE.md](docs/DEPLOYMENT_FUTURE.md)).

## 8. Dashboard

| Page | What it shows |
|---|---|
| **Overview** | Three headline numbers; the map (roads coloured Flowing / Slow / Heavy / Jammed by the backend, grey = no data); **Needs attention** (roads with an open incident or congestion, worst first); the selected road (speed, next-30-minute forecast, incident score marked UNCALIBRATED, risk) beside its camera view or speed trend; the riskiest roads |
| **Incidents** | Possible incidents with time, road, type, score, risk and evidence; detail view with **TRUE INCIDENT / FALSE ALARM** feedback, stored for calibration |
| **Simulate** | What-if: close a road, close a lane or raise demand (pick the road on the map). The map shows the closure and where traffic moves; the panel lists affected roads, congestion before → after and the assumptions. Labelled decision support |
| **AI models** | What each stage serves now and its measured result on real data, read from the experiment records |
| **System** | Backend, database, pipeline, forecast model, cameras and weather status from real signals; production models |

The header shows LIVE only when the last REST poll succeeded and the API reports fresh live values
(otherwise WAITING FOR DATA or DISCONNECTED). A site without a known location (the public video clip)
is drawn on a plain background, without a city map or weather.

Presenting the project: [docs/EVALUATION_BRIEF.md](docs/EVALUATION_BRIEF.md).

## 9. Testing

```bash
cd backend && pytest                      # unit + integration (no GPU / video / PostgreSQL needed)
ruff check traffic_intel ../tests         # lint
mypy traffic_intel                        # strict type checking
cd ../frontend && npm test                # component, API-client and UI integration tests (Vitest)
npm run lint && npm run typecheck && npm run build
```

- **End-to-end test:** `tests/backend/integration/test_local_system.py` runs the real runtime
  (synthetic camera → tracker → features → analytics → database), then the API on the same
  database.
- **Synthetic data in tests:** synthetic scenes and fixtures are used only as test data and are
  labelled as such in the UI.

## 10. Known limitations

- **No incident labels from this site yet.** Incident scores are uncalibrated and the fusion
  weights are hand-set. On real video (AI City 2021 Track 4), the stopped-vehicle signal ranks
  incident windows well (ROC-AUC 0.987), but with 3 incident videos per split, alarm rates are
  uncertain and a calibration does not transfer
  ([docs/MODEL_EVALUATION.md §4](docs/MODEL_EVALUATION.md)).
- **Synthetic demo data.** The demo profile's junction approaches are simulated test scenes.
  Their numbers demonstrate the pipeline, not a real site. Only the public highway camera is
  real footage, and its calibration is approximate.
- **Forecasting model.** The production forecaster was trained on synthetic junction data. The
  real-data evidence (METR-LA, PEMS-BAY, PEMS08) chooses the model family, not the weights. A
  model for a real site needs weeks of recordings from that site.
- **What-if** is a static macroscopic assignment on a small graph, with assumed flows on roads
  without cameras. It is not a microscopic simulator.
- **Camera view** is a ≈1 fps annotated preview, not a video stream.
- **No authentication.** The dashboard is intended for local use.
- **Real-time throughput** depends on the GPU. On CPU, lower `process_fps` in the camera
  configuration.

More: [docs/INCIDENT_INTELLIGENCE.md](docs/INCIDENT_INTELLIGENCE.md) §10,
[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §7.

## 11. Future deployment and scaling

Docker images, a managed PostgreSQL/PostGIS, Redis Streams with separate camera processes,
GPU inference servers, object storage, load balancing and monitoring are documented, but not
implemented, in [docs/DEPLOYMENT_FUTURE.md](docs/DEPLOYMENT_FUTURE.md).

## Repository layout

| Path | What |
|---|---|
| `backend/traffic_intel/` | `core`, `computer_vision`, `tracking`, `traffic`, `forecasting`, `anomaly`, `incidents`, `risk`, `simulation`, `context`, `datasets`, `ai`, `pipeline`, `database`, `services`, `api` |
| `frontend/src/` | `api/` (typed REST client), `config.ts` (polling intervals), `state/` (live polling, selection), `hooks/`, `components/`, `pages/`, `test/` |
| `configs/` | `pipeline.yaml`, `runtime.yaml`, `anomaly.yaml`, `risk.yaml`, `network/`, `cameras/`, `training/`, `datasets/` |
| `data/`, `models/`, `experiments/`, `runs/` | data, model registry, experiment records, runtime outputs (git-ignored) |
| `docs/` | ARCHITECTURE, DATASETS, FORECASTING, INCIDENT_INTELLIGENCE, PROJECT_REPORT, PHASE1_PERCEPTION, TRAFFIC_STATE_ESTIMATION, DEPLOYMENT_FUTURE |
| `tests/backend/` | `unit/`, `integration/` |
