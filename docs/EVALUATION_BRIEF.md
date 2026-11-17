# Evaluation brief: AI Urban Traffic & Incident Intelligence

A one-stop summary for presenting the project: what it is, how it works, what was measured, what it
cannot do yet, how to demo it, and answers to likely questions. Every number below comes from a
saved experiment record (`experiments/`) or `docs/MODEL_EVALUATION.md`.

---

## 1. The problem in one paragraph

Traffic operators watch many cameras and sensors but cannot see everything at once. They need to know
**what is happening now**, **what will happen in the next 30 minutes**, **whether something abnormal
(a possible incident) is going on**, **which roads are riskiest**, and **what would happen if a road
were closed**. This project turns raw traffic video and road-sensor data into those five answers and
shows them on one live dashboard.

## 2. What the system does (the pipeline)

```
Video / road sensors
  -> 1. Perception      YOLO11m finds vehicles, ByteTrack follows each one, homography turns pixels into metres
  -> 2. Understanding   per-road features every window: count, speed, density, flow, queue, stopped vehicles
                        -> congestion level (Flowing / Slow / Heavy / Jammed) from explainable rules
  -> 3. Prediction      speed forecast at +5 / +15 / +30 min (XGBoost)
  -> 4. Anomaly         statistical detectors (sudden slowdown, queue growth, z-scores, stopped vehicles)
  -> 5. Incident        detectors fused into an incident score, honestly labelled UNCALIBRATED
  -> 6. Risk            documented weighted score 0-100 with reason codes
  -> 7. Decision        what-if simulation: close a road / a lane / raise demand, see where traffic goes
  -> Dashboard          React + map, polling a FastAPI REST API
```

## 3. Architecture (how the parts fit)

| Part | Technology | Job |
|---|---|---|
| Pipeline process | Python, OpenCV, Ultralytics YOLO11, native ByteTrack, NumPy/pandas, XGBoost | Reads video or sensors, runs all AI stages, writes results to the database |
| Database | SQLite locally (PostgreSQL + PostGIS supported) | Stores per-window traffic states, forecasts, incidents, risk, network |
| API process | FastAPI, pydantic, SQLAlchemy (no PyTorch, so it stays light) | REST endpoints the dashboard reads |
| Dashboard | React 19, TypeScript, Tailwind, MapLibre (map), ECharts (charts) | Overview, Incidents, Simulate, AI models, System pages |
| Configs | YAML in `configs/` | Which models serve, network, cameras, rules, risk weights |

Design points worth saying out loud:

- **Only aggregates move.** Per-frame detections stay in the pipeline; the database and browser see
  per-window summaries. That keeps it fast.
- **Every model sits behind an interface**, chosen in `configs/pipeline.yaml`; swapping a model is a
  config change.
- **One feature definition** (`core/feature_schema.py`) is used for both training and live inference,
  and a model whose preprocessing hash does not match refuses to load.
- **Nothing is invented.** When data is missing or stale the dashboard says so instead of showing
  numbers.

## 4. Three run modes

| Mode | Data | Ports | Use it to show |
|---|---|---|---|
| **Real data (recommended for the demo)** | PEMS-BAY: 325 real freeway speed sensors in San Jose, recorded 2017, replayed at today's time of day | API 8000, dashboard 5173 | Real map, live colours, forecasts, incidents, risk, what-if |
| **Video (GPU)** | Public highway clip through YOLO11m + ByteTrack on the RTX 4060 | API 8001, dashboard 5174 | Live camera view with boxes, vehicle counts and speeds |
| Demo (synthetic) | Simulated junction scenes | API 8000, dashboard 5173 | Pipeline testing only |

## 5. The modules and their measured results

### 5.1 Vehicle detection: YOLO11m (COCO-pretrained)
- **Test result (UA-DETRAC, 5,625 images, 63,377 vehicles):** mAP50-95 **0.634**, mAP50 0.874,
  precision 0.86, recall 0.85, about 12 ms per image at 960 px.
- **We fine-tuned too, and rejected it.** YOLO11m fine-tuned on UA-DETRAC scored mAP50-95 0.488 on
  the same test set (YOLO11s fine-tunes: 0.40 to 0.43). Cars improved a little (0.66 vs 0.62) but
  trucks collapsed (0.20 vs 0.67), and the model forgot the pedestrian, bicycle and motorcycle classes,
  so the pretrained model stays in service. The fine-tuned weights are kept for reference.

### 5.2 Tracking: ByteTrack
- **Test MOTA 0.749** on UA-DETRAC (precision 0.936, recall 0.804), after tuning one threshold
  (`new_track_thresh` 0.6 -> 0.4, chosen on validation, improved test MOTA from 0.737).
- BoT-SORT got the same MOTA but was **33x slower** (35 ms vs 1 ms per frame), so ByteTrack was kept.

### 5.3 Speed forecasting: XGBoost (real-data mode)
- Trained on PEMS-BAY (Jan to May 2017), tested on unseen weeks (late May to June 2017), 109 sensors,
  8 inputs (current speed + time of day, day of week, weekend, holiday, rush hour).
- **Test RMSE 4.41 km/h vs 5.26 for "no change" persistence: 15% better.** MAE 2.16 vs 2.44.
- By horizon (RMSE): +5 min 2.40, +15 min 4.54, +30 min 6.28 km/h.
- Across METR-LA, PEMS-BAY and PEMS08, **tree models (XGBoost, Random Forest) matched or beat the deep
  models** (LSTM, GRU, Transformer) at a fraction of the cost, which is why XGBoost serves.

### 5.4 Congestion level
- Explainable rules (`configs/congestion_rules.yaml`) map speed/density/queue to Flowing, Slow,
  Heavy, Jammed. No independent labels exist, so a trained classifier would only copy the rules.
  Future levels = forecast -> the same rules.

### 5.5 Anomaly and incident detection
- Statistical detectors are fused into one incident score per road.
- **Real-video evaluation (AI City 2021 Track 4):** the stopped-vehicle signal separates incident from
  normal windows with **ROC-AUC 0.987** (PR-AUC 0.891), better than the learned detectors (autoencoder
  0.925, isolation forest 0.846).
- **Why the score stays UNCALIBRATED:** the test split has only **3 incident videos**. It found 2 of 3,
  and calibrating on so few made probabilities worse on new videos. Saying "uncalibrated" is the
  honest choice, and every screen says so.

### 5.6 Risk score
- Documented weighted formula (`configs/risk.yaml`), 0 to 100, with reason codes such as
  "Severe congestion now" or "Congestion worsening". No dataset has risk labels, so it is transparent
  instead of trained.

### 5.7 What-if simulation
- Static macroscopic re-assignment on the road graph (NetworkX). Close a road or a lane, or raise
  demand 20%, and the map shows where traffic moves (orange = more, teal = less, dashed = closed),
  with congestion probability before -> after. It is labelled decision support, not a prediction.

## 6. Datasets used (each for its own task, never mixed)

| Dataset | What it is | Used for |
|---|---|---|
| UA-DETRAC | 100 traffic-camera sequences, 140k frames, 1.27M boxes | Detector and tracker evaluation, fine-tune experiment |
| PEMS-BAY | 325 freeway sensors, 6 months, 5-min speeds | Forecaster training; live real-data mode |
| METR-LA | 207 LA sensors, 4 months | Forecasting model comparison |
| PEMS08 | 170 stations, flow/occupancy/speed | Forecasting model comparison |
| AI City 2021 Track 4 | Real freeway videos with labelled anomalies | Incident detection evaluation |
| Public highway clip | Real motorway video | Live video demo |
| Synthetic junction | Simulated with exact ground truth | Pipeline tests and speed validation |

## 7. Honest limitations (say these before they are asked)

1. **Incident score is uncalibrated.** Only 3 labelled incident videos per split; a real site needs
   its own labelled incidents. Operators mark each alert TRUE INCIDENT / FALSE ALARM in the Incidents page,
   and that feedback is stored for future calibration.
2. **Real-data mode replays recorded 2017 sensors** at today's time of day; it is not a live feed.
3. **The video clip has no known location**, so the video dashboard shows a schematic view instead of a
   city map, and weather is switched off for it.
4. **What-if is static**, not a microscopic simulator; roads without measurements use assumed flows
   (marked with *).
5. **Camera view is about one annotated frame per second**, not a video stream.
6. **No login or deployment.** It runs on one machine; the cloud design is documented in
   `docs/DEPLOYMENT_FUTURE.md` but not built.

## 8. Five-minute demo script

### Before the evaluation: start the app (3 terminals, PowerShell)

Terminal 1, pipeline (from `backend`):
```powershell
cd "C:\Users\Kavish Gupta\AI-Project-Traffic\backend"
$env:TI_PIPELINE_FILE="pipeline_pems_bay.yaml"; .\.venv\Scripts\python.exe -m traffic_intel.pipeline run --profile pems_bay
```

Terminal 2, API (from `backend`):
```powershell
cd "C:\Users\Kavish Gupta\AI-Project-Traffic\backend"
$env:TI_PIPELINE_FILE="pipeline_pems_bay.yaml"; .\.venv\Scripts\python.exe -m uvicorn traffic_intel.api.main:app --port 8000
```

Terminal 3, dashboard (from `frontend`):
```powershell
cd "C:\Users\Kavish Gupta\AI-Project-Traffic\frontend"
npm run dev
```

Open http://localhost:5173 and wait about a minute until the header says **LIVE**.

Optional video mode (3 more terminals, same pattern): pipeline with `pipeline_video.yaml` and
`--profile video`, API with `--port 8001`, and the dashboard with
`$env:VITE_BACKEND_URL="http://localhost:8001"; npx vite --port 5174`. Open http://localhost:5174.

To stop: press Ctrl+C in each terminal.

### The script

| Time | Show | Say |
|---|---|---|
| 0:00 | Overview: 3 numbers, map | "325 real freeway sensors in San Jose. Colours come from our congestion rules. LIVE means fresh data." |
| 0:45 | Click a red/orange road in **Needs attention** | "The road card shows current speed, our XGBoost forecast for the next 30 minutes, the incident score marked uncalibrated, and the risk score." |
| 1:30 | Traffic trends | "Speed over time, forecast vs actual so you can see how accurate the forecast is, and the incident score over time." |
| 2:00 | **Incidents** page, click one, show evidence | "Each alert is a possible incident with its evidence. The operator confirms or rejects it, and that feedback will calibrate the model later." |
| 2:45 | **Simulate**: click a road on the map, Close road, Run | "Where would traffic go? Orange roads take more, the closed road is dashed. It is decision support, not a guarantee." |
| 3:30 | **AI models** page | "Every model with the number it scored on real data: detector mAP 0.634, tracker MOTA 0.749, forecast 15% better than persistence, incident ROC-AUC 0.987." |
| 4:15 | Video dashboard (5174) if running | "The same pipeline on real video: YOLO11m on the GPU detects and tracks vehicles and measures their speed." |
| 4:45 | **System** page | "Health of each part from real signals. That's the system; happy to go deeper on any model." |

## 9. Likely questions and answers

**Why YOLO11m and not a fine-tuned model?** We fine-tuned on UA-DETRAC and tested both on the same
5,625 test images. Pretrained scored 0.634 mAP50-95, fine-tuned 0.488: cars got slightly better but
trucks dropped from 0.67 to 0.20, and the extra COCO classes were lost. We let the test set decide.

**Why XGBoost and not an LSTM or Transformer?** We trained 8 model types on three public datasets with
chronological splits. Tree models matched or beat the deep ones and are much faster and easier to
explain.

**What does "15% better than persistence" mean?** Persistence predicts "speed stays the same", the
standard baseline. Our forecast error (RMSE) is 4.41 km/h vs 5.26, so 15% lower, measured on weeks the
model never saw.

**Why is the incident probability uncalibrated?** A probability must match real frequencies. With
only 3 incident videos per test split we cannot calibrate honestly, so we show the score as a ranking and
label it. The ranking itself is strong (ROC-AUC 0.987).

**How do you avoid data leakage?** Chronological train/validation/test splits, scaling fitted on train
only, model selection on validation, and the test set used once. For video, splits are by video, not
frame.

**How is speed measured from video?** Tracks are projected from pixels to metres with a homography
calibrated per camera, and speed is distance over time. On synthetic scenes with exact ground truth,
mean speed was within 0.7 km/h.

**Is it real-time?** Yes. On the RTX 4060 the full video pipeline runs about 12 to 14 frames per second;
the dashboard polls every 3 seconds.

**What is the risk score based on?** A documented weighted formula of current congestion, forecast
congestion, speed drop and incident evidence, with reason codes. No dataset has risk labels, so it is
transparent instead of learned.

**How is the what-if computed?** Demand is re-assigned over the road graph with the closed road removed
or its capacity halved, and congestion probability is recomputed per road. It is static, so it shows
direction and scale, not exact minutes.

**How would you deploy it?** Docker images, PostgreSQL + PostGIS, Redis Streams between camera
workers, GPU inference servers, and a WebSocket push channel (already in the code, switched off). This
design is in `docs/DEPLOYMENT_FUTURE.md`.

**How is it tested?** 236 backend tests (unit and an end-to-end run of the real pipeline) and 28
dashboard tests, plus type checking and linting on both sides.

## 10. Where to find more

`README.md` (overview), `docs/ARCHITECTURE.md`, `docs/MODEL_EVALUATION.md` (all numbers and reasons),
`docs/FORECASTING.md`, `docs/INCIDENT_INTELLIGENCE.md`, `docs/DATASET_INVENTORY.md`,
`docs/PROJECT_REPORT.md`.
