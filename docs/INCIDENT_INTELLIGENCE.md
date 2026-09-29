# Incident Intelligence: anomaly detection, incident fusion, risk and what-if

**Phase 6 of the system (Prompt 4).** This layer turns the traffic state produced by the
perception layer (and the forecasts of the ML layer) into three things:

- **evidence:** anomaly signals;
- **hedged incident hypotheses:** each with an incident probability and an uncertainty note;
- **explainable risk:** a score with reason codes.

It also answers "what if this road closes?" with a deliberately simple network model.

> **This is decision support, not an autonomous emergency-response system.**
> The system never states "accident detected". It reports, for example, *"Possible incident -
> incident probability 87% (uncalibrated estimate from combined signals; not a confirmed accident -
> verify before acting)"*. Every incident carries its probability, the evidence it rests on,
> `calibrated: false` (until the fusion weights are fitted on labelled incidents) and an
> `uncertainty_note`. Operators confirm, dismiss or resolve incidents. The engine never overrides
> an operator decision.

---

## 1. Architecture

```
 camera worker (per camera, every processed frame)                analytics worker (per 60 s window)
 ─────────────────────────────────────────────────                ───────────────────────────────────
 tracks ─► trajectories (TrajectoryStore: ground xy, speed,        TrafficFeatures + TrafficState
           heading, stopped duration)                                   │  (history ring buffer / approach)
      │                                                                 ▼
      ▼                                                     SERIES detectors  (anomaly/detectors/)
 TRACK detectors  (anomaly/detectors/track_rules.py)          sudden_slowdown   EWMA baseline + robust z
   stopped_vehicle   track, location, duration, conf.         queue_growth      slope + rise vs median
   wrong_way         vs configured lane direction             density_zscore    median/MAD z-score
   trajectory_rules  swerving / off-lane                      flow_imbalance    inflow >> outflow
      │                                                       ewma, zscore      control charts, 4 features
      │  AnomalySignal (bus: anomaly.signal)                  isolation_forest  learned (optional)
      └──────────────────────────────┐                        autoencoder       learned (optional)
                                     ▼                                 │ AnomalySignal
                          IncidentFuser (anomaly/fusion.py) ◄──────────┘
                    evidence families -> incident probability, type (taxonomy), severity,
                    lifecycle (open / update / escalate / resolve), hedged message
                                     │ IncidentAssessment          Forecast (+5/+15/+30, Prompt 3)
                                     ▼                                 │
                          RiskEngine (risk/weighted.py) ◄──────────────┘
                    risk_score 0..100, risk_level, reason_codes, formula breakdown
                                     │
             TrafficStore (database/store.py, ORM) ──► services ──► REST API (api/) ◄── polled by the dashboard
                                                              what-if: StaticAssignmentSimulator
                                                              (simulation/static_assignment.py, NetworkX)
```

- **One engine, several drivers.** `incidents/engine.py::IntelligenceEngine.process_window` is the
  only place where the chain runs. It is used by the bus-driven `pipeline/analytics_worker.py`, the
  offline `ingest` CLI, the synthetic scenarios and the tests.
- **Where each part runs.** Track detectors run inside the camera worker, where the frame-rate
  trajectories live. Their signals are rate-limited (re-emitted at most every 10–30 s per track)
  and are the only new message type a camera worker may publish (`anomaly.signal`).
- **The API stays light.** The API process loads neither torch nor sklearn. It reads the store and
  runs the NetworkX what-if. A test enforces this.
- **Everything is config-driven.** Configuration lives in:
  - `configs/pipeline.yaml`: which detectors run, with which parameters;
  - `configs/anomaly.yaml`: fusion weights and the incident taxonomy;
  - `configs/risk.yaml`: risk weights, levels and reason thresholds;
  - camera YAML lane zones: the expected direction of travel;
  - `configs/network/*.yaml`: road importance and capacities.

---

## 2. Anomaly detection: several layers, never one signal

### 2.1 Track level (per camera, per frame)

| Detector | Fires when | Output (in `AnomalySignal.details`) |
|---|---|---|
| `stopped_vehicle` | A vehicle is nearly stationary (≤ 3 km/h) for ≥ `min_stopped_s` (30 s, configurable). Vehicles waiting inside a configured **queue zone** (at a signal) only count after `max_queue_wait_s` (180 s). | `track_ids`, `location` {lon, lat, ground x/y, distance upstream of the stop line}, `stopped_duration_s`, `road_id`, `junction_id`, `confidence`, `in_queue_zone` |
| `wrong_way` | Net displacement **against the lane's configured direction** ≥ 8 m within 4 s, heading cosine ≤ −0.5, speed ≥ 5 km/h. | track, location, cosine, distance |
| `trajectory_rules` | Heading rate > 30°/s at ≥ 15 km/h (swerving), or ≥ 2 s on the monitored road area but outside every lane zone. | track, reason, speed |

**Stopped-vehicle confidence:**

`confidence = ramp(duration; min_stopped_s → 2·min_stopped_s) × (0.5 + 0.5·trajectory reliability) × (1.0 if other traffic is still moving, else 0.7)`

A lone stopped car in moving traffic is more suspicious than a car standing in a jam.

**Road direction is never hard-coded.** The expected direction comes from each lane zone's
`direction` in the camera YAML, falling back to the camera's travel direction. A unit test flips
the configured direction and checks that the other vehicle becomes the wrong-way one.

### 2.2 Series level (per approach, per 60 s window)

All series detectors compare the **newest window** with the windows just before it. This is a
local, adaptive baseline, so one global threshold does not have to fit every site and hour. Each
detector returns a score in [0, 1] plus the numbers that produced it.

| Detector | Signal type | Method |
|---|---|---|
| `sudden_slowdown` | SUDDEN_SLOWDOWN | Baseline = EWMA of recent speeds. `drop = (baseline − current)/baseline`. `z = (baseline − current)/max(1.4826·MAD, 0.1·baseline)`. `score = ramp(drop; 0.25 → 0.6) × min(1, z/3)`. The same score is computed for **p85 speed** and the smaller one is used: a real stream slowdown slows the fast vehicles too, while one stopped car only drags the mean down (that is the stopped-vehicle detector's evidence). |
| `queue_growth` | SUDDEN_QUEUE | Least-squares slope of queue length over the last 4 windows × rise above the recent median queue. Boosted when the queue runs out of the camera view. |
| `density_zscore` | UNUSUAL_DENSITY | Robust z-score (median/MAD) of density, upward only. The density explained by the window's stopped vehicles is removed first, so one breakdown is not counted twice. |
| `flow_imbalance` | UNUSUAL_FLOW | Vehicles accumulating: Σ(inflow − outflow)/Σ inflow over 3 windows, plus outflow collapse vs its median. |
| `ewma` | STATISTICAL_OUTLIER | EWMA control chart (λ = 0.3, L = 3) on speed (down), density, occupancy and queue (up). |
| `zscore` | STATISTICAL_OUTLIER | Moving-average ± rolling-std z-score on the same features. |
| `isolation_forest` | STATISTICAL_OUTLIER | sklearn Isolation Forest on site-independent deviation features (z-scores and one-window changes vs a rolling 15-window baseline). Fitted **offline on normal windows only**. |
| `autoencoder` | STATISTICAL_OUTLIER | Bottleneck MLP (8→16→4→16→8) trained to reproduce normal windows; reconstruction error. |

Learned outputs are mapped to [0, 1] using the training distribution: 0 up to the 99th percentile
of normal windows, 1 at 2× the 99.9th percentile. It is an outlier score, not an incident
probability. The learned detectors are optional pipeline inputs (commented in
`configs/pipeline.yaml`) and are loaded from the model registry `models/anomaly/<kind>/<version>/`.

**Worked example (from the specification):** speeds 45, 43, 42, 40, 18, 7 km/h.

| Window | Baseline | Drop | Score | Reading |
|---|---|---|---|---|
| 18 km/h | ≈ 42 km/h (EWMA) | 57 % | 0.91 | abnormal slowdown |
| 7 km/h | ≈ 35 km/h (EWMA) | 80 % | 1.0 | abnormal slowdown |

For the 18 km/h window, z = 24/4.2 ≈ 5.7 ≥ 3. The 7 km/h window keeps its full score because the
robust spread (MAD) is not inflated by the 18 km/h window. Ordinary ±2.5 km/h fluctuation gives 0.
(Test: `tests/backend/unit/test_anomaly_detectors.py`.)

---

## 3. Incident fusion: many weak signals → one hedged hypothesis

`anomaly/fusion.py::WeightedLogisticFuser`, configured in `configs/anomaly.yaml`.

**Evidence families.** Each family's evidence `e_f` is the strongest signal of that family in the
last 300 s, fading linearly with age:

| Family | Signals |
|---|---|
| speed_drop | sudden_slowdown |
| queue_growth | sudden_queue |
| density_change | unusual_density |
| flow_imbalance | unusual_flow |
| stopped_vehicle | stopped_vehicle |
| trajectory_anomaly | wrong_way, abnormal_trajectory |
| anomaly_score | statistical_outlier (EWMA, z-score, Isolation Forest, autoencoder) |

```
S = 1 − Π (1 − e_f)       over the SYMPTOM families (speed_drop, queue_growth, density_change, flow_imbalance)
z = −4 + 2.5·S + 2.5·e_stopped + 2.0·e_trajectory + 1.0·e_anomaly + 3.0·S·max(e_stopped, e_trajectory)
incident_probability = 1 / (1 + e^−z)
```

**Why this form:**

- **Symptoms are correlated.** A slowdown, a growing queue and rising density are usually three
  views of one cause: congestion. Adding them would count congestion three times, so they are
  combined with a noisy-OR into one strength S.
- **Evidence must agree to be strong.** The interaction term makes the probability high only when
  traffic symptoms coincide with a physical cue (a stopped vehicle, or abnormal movement in the
  jam). "Do not rely on one signal" is built into the functional form:

| Evidence | Incident probability |
|---|---|
| nothing | 0.02 |
| heavy congestion alone (S = 1) | 0.18 |
| any single cue alone | ≤ 0.38 |
| congestion + stopped vehicle | 0.94 – 0.99 |

The weights are **hand-set** (`calibrated = false`). The calibration path:

1. Operators label incidents (`PATCH /incidents/{id}` with true_incident / false_alarm).
2. A logistic regression is fitted on the family evidences.
3. Platt or isotonic calibration is applied.
4. `calibrated = true`.

### 3.1 Incident taxonomy (configurable)

`configs/anomaly.yaml → incident_types` is evaluated top to bottom; the first matching rule names
the situation. New types can be added in YAML (tested).

| Type | Label | Rule (default) |
|---|---|---|
| `possible_incident` | Possible incident | probability ≥ 0.6 **and** ≥ 2 families ≥ 0.3 |
| `wrong_way` | Wrong-way vehicle | wrong-way signal present (severity ≥ HIGH) |
| `stopped_vehicle` | Stopped vehicle | stopped-vehicle evidence ≥ 0.4 |
| `abnormal_queue` | Abnormal queue | queue-growth evidence ≥ 0.4 |
| `sudden_slowdown` | Sudden slowdown | speed-drop evidence ≥ 0.4 |
| `abnormal_trajectory` | Abnormal trajectory | trajectory evidence ≥ 0.5 |
| `normal` | Normal | otherwise |

**Lifecycle per approach:**

- An incident opens when the type is not `normal`.
- While it persists it keeps its `incident_id` and updates its probability and evidence.
- Its type may **escalate** (for example stopped_vehicle → possible_incident) but is never silently
  downgraded.
- It resolves after 3 consecutive normal windows.

**Severity:**

- critical: probability ≥ 0.85 with SEVERE congestion;
- high: probability ≥ 0.7;
- medium: probability ≥ 0.35;
- the type can impose a minimum (wrong-way ≥ high).

---

## 4. Risk engine (documented, not opaque)

`risk/weighted.py`, configured in `configs/risk.yaml`. Every component is in [0, 1]:

| Component | Definition | Weight |
|---|---|---|
| current_severity | congestion score of the latest window (rule classifier) | 0.20 |
| forecast_severity | worst forecast level over +5/+15/+30 min: free 0, moderate 0.35, high 0.65, severe 1 | 0.20 |
| incident | fused incident probability (0 if no open/acknowledged incident) | 0.30 |
| queue | queue_length / min(60 m, ROI length) | 0.10 |
| speed | (1 − speed / free-flow speed) / 0.8 | 0.10 |
| density | density_ratio / 0.8 | 0.10 |

```
base = Σ wᵢ·cᵢ / Σ wᵢ             (over AVAILABLE components - e.g. no forecast -> weights renormalised)
risk_score = 100 · base · (0.7 + 0.3 · road_importance)       road_importance ∈ [0,1] from the network config
risk_level:  low < 25 ≤ medium < 50 ≤ high < 75 ≤ critical
```

**Safety floors.** These only ever raise a level, and each is stated in the explanation:

- a wrong-way vehicle → at least HIGH;
- incident probability ≥ 0.85 → at least HIGH.

**Reason codes.** Each component or signal above its threshold adds a code and a sentence:
SEVERE_CONGESTION_NOW, HIGH_CONGESTION_NOW, SEVERE_CONGESTION_FORECAST, CONGESTION_WORSENING,
POSSIBLE_INCIDENT, ELEVATED_INCIDENT_PROBABILITY, ABNORMAL_SLOWDOWN, STOPPED_VEHICLE,
WRONG_WAY_VEHICLE, QUEUE_INCREASING, VEHICLES_ACCUMULATING, ABNORMAL_TRAJECTORY, LONG_QUEUE,
LOW_SPEED, HIGH_DENSITY, HIGH_IMPORTANCE_ROAD, LOW_DATA_QUALITY.

The first line of `explanation` is always the formula with the actual numbers, for example
`risk = 100 x 0.93 x (1 - 0.3 + 0.3 x 0.80) = 88 [current_severity 1.00x0.2, incident 1.00x0.3, ...]`.

---

## 5. What-if simulation (static assignment, NOT a microscopic simulator)

`simulation/static_assignment.py` builds a NetworkX `DiGraph` from `configs/network/*.yaml`:

- **Nodes** are junctions and road nodes.
- **Edges** are directed road segments. Each has a length (from its geometry), free-flow time,
  capacity (`capacity_veh_h`, or lanes × 900 veh/h), importance and status.

The method:

1. **Baseline flow per edge.** Use the latest *observed* flow where a camera exists; otherwise
   `assumed_vc × capacity` (0.5). Multiply by `demand_factor`.
2. **Treat each closed or reduced edge's flow as demand between its end nodes.** Re-assign it over
   {the edge itself, if still open} ∪ {k = 3 shortest alternative paths}, using the Method of
   Successive Averages. In each of 50 iterations, every OD pair is loaded all-or-nothing onto its
   currently cheapest path, and the result is averaged with step 1/n. Link times follow BPR:
   `t = t0·(1 + 0.15·(v/c)^4)`. The flow on other edges is background load.
3. **Congestion probability per edge** = `1/(1 + e^(−10·(v/c − 0.85)))`, which is 50 % at
   v/c = 0.85. A junction's probability is the flow-weighted mean over the edges entering it.
4. **Return** per-segment flow change and congestion probability before/after, per-junction
   probability, unserved demand (no alternative path), human-readable summary lines and the full
   list of assumptions.

Example (`python -m traffic_intel.incidents what-if --close road_b_wb` on the demo network):

```
Road B (westbound): CLOSED (900 veh/h displaced)
Road F (northbound): +44% flow, congestion probability 3% -> 21%
Road G (south-west): +156% flow, congestion probability 3% -> 99%
Road A (southbound): +22% flow, congestion probability 3% -> 8%
Road C (northbound): +78% flow, congestion probability 3% -> 60%
Road E (westbound): +44% flow, congestion probability 3% -> 21%
Junction A: congestion probability 3% -> 34%
```

**Limitations, stated with every result:**

- no individual vehicles, signal timing, turning restrictions, queue spill-back or behavioural
  change (for example trips being cancelled);
- true origins and destinations are unknown, so only the displaced flow is re-routed;
- unobserved links use assumed volumes;
- the probability mapping is not calibrated.

Use it for relative "where will traffic go" answers, not predictions.

---

## 6. Storage

Results are stored through `database/store.py::TrafficStore`, on the project's ORM models
(`database/models.py`). The column types are portable: SQLite locally (`data/db/traffic.db`),
PostgreSQL + PostGIS in production (`TI_DATABASE_URL`).

| Table | Content |
|---|---|
| `incident` | id, last update, first_detected, resolved_at, approach, **junction_id, road_id**, **incident_type**, label, **probability (score)**, max_probability, calibrated, severity, **risk_level**, **status**, message, **evidence**, contributing signals, geom, operator_label / operator_note |
| `anomaly_event` | every signal: type, detector, score, track_ids, location (geom), details, incident_id |
| `risk_assessment` | risk_score, level, reason_codes, components, explanation, per-window incident score |
| `traffic_observation` + `congestion_state` | every closed 60 s window (features + level) |
| `forecast` | issued forecasts per horizon and target (for forecast vs actual) |
| `simulation_scenario` | what-if requests and results |

Operator decisions win: an acknowledged or dismissed incident keeps that status when the engine
updates its score. Only `resolved` (evidence gone) is set automatically.

---

## 7. APIs

Base path `/api/v1` (OpenAPI at `/docs`).

| Method & path | Returns |
|---|---|
| `GET /traffic/current?junction_id=` | latest TrafficState per approach |
| `GET /traffic/{approach_id}/history?since=&limit=` | recent windows |
| `GET /anomalies?approach_id=&junction_id=&type=&since=&min_score=&limit=` | anomaly signals (evidence) |
| `GET /incidents?status=&approach_id=&junction_id=&active_only=&since=` | incidents with probability, evidence, contributing signals, uncertainty note |
| `GET /incidents/{id}` | one incident |
| `PATCH /incidents/{id}` `{status: acknowledged\|resolved\|dismissed, label, note}` | operator feedback (future calibration labels) |
| `GET /risk?approach_id=&junction_id=` | latest risk per approach (score, level, reason codes, components, formula) |
| `GET /risk/{approach_id}/history` | risk time series |
| `POST /simulation/what-if` `{closed_segment_ids, capacity_factors, demand_factor}` | SimulationResult (impacts, junctions, summary, assumptions) |
| `GET /simulation/scenarios/{id}` | a stored what-if result |
| `WS /ws/live` | optional future push channel, **off by default** (`TI_WS_ENABLED`); the dashboard polls the REST routes above |

---

## 8. Commands

```bash
cd backend
# the six synthetic scenarios through the real pipeline (add --store to fill the dev DB for the API)
python -m traffic_intel.incidents scenarios [--name stopped_vehicle] [--store]
# replay a recorded camera run (runs/<camera>/<run>/traffic_states.jsonl + anomalies.jsonl)
python -m traffic_intel.incidents ingest ../runs/<camera>/<run_id>
# what-if on the configured network (baseline = latest stored observations)
python -m traffic_intel.incidents what-if --close road_b_wb [--capacity road_a_sb=0.5] [--demand 1.1]
# train + evaluate the learned anomaly detectors, promote to the registry
python -m traffic_intel.ai.training train anomaly --config ../configs/training/anomaly.yaml --promote best-val
# API
uvicorn traffic_intel.api.main:app --reload
```

---

## 9. Test scenarios and results

`incidents/scenarios.py` simulates a 2-lane mid-block approach using IDM car-following with
injected events. Vehicle boxes are projected through a known homography. Everything **from
tracking onwards is the real pipeline**:

1. trajectories;
2. the feature engine;
3. track detectors;
4. the congestion classifier;
5. the series detectors, fusion and risk.

Only YOLO and the tracker are replaced by ground truth. Each run is 15 min (5 fps): 10 normal
minutes, then the event.

| # | Scenario | Injected event | Reported type | Max. incident probability (before → after) | Peak risk |
|---|---|---|---|---|---|
| 1 | normal | none | normal | 0 % → 0 % | 9, low |
| 2 | sudden_slowdown | all traffic slows to ~10 km/h | sudden_slowdown | 0 % → 38 % | 46, medium |
| 3 | stopped_vehicle | a car stops in lane 2 at 37 m; others pass in lane 1 | stopped_vehicle | 0 % → 35 % | 33, medium |
| 4 | wrong_way | one car drives against the lane direction | wrong_way | 0 % → 5 % | HIGH (safety floor) |
| 5 | queue_formation | demand > downstream bottleneck; queue grows, traffic keeps discharging | abnormal_queue | 0 % → 38 % | 68, high |
| 6 | multiple_signals | a car blocks lane 2 near the exit + lane 1 slows | possible_incident | 0 % → 99 % | 90, critical |

**Reading the table:**

- Each single abnormal pattern is **named** (type) but stays **below 0.6**. Only the combination
  of independent evidence (stopped vehicle + queue growth + slowdown + density) is called a
  *possible incident*.
- A wrong-way vehicle has a low *incident* probability (nothing has happened to traffic yet). The
  **risk** is nevertheless forced to HIGH by the safety floor, and the reason code says why.
- The integration test checks these logical properties, not exact numbers
  (`tests/backend/integration/test_incident_scenarios.py`):
  - quiet before the event;
  - the right type after it;
  - stopped-vehicle output fields (track id, location ≈ 37 m, duration ≥ 30 s, road, confidence);
  - the wrong-way risk floor;
  - no stopped-vehicle false positive in a creeping queue;
  - ≥ 3 evidence families for the possible incident;
  - hedged wording everywhere;
  - the bus-driven analytics worker reproduces the direct engine.

Three detector refinements came directly out of these scenarios. Each is documented in the code:

- **Slowdown uses p85 speed too.** Otherwise one stopped car in light traffic looked like a stream
  slowdown.
- **Density discounts stopped vehicles.** Same reason.
- **Off-lane only counts inside the ROI.** Vehicles leaving past the exit line were flagged as
  "off lane".

### 9.1 Learned detectors on labelled synthetic series

Experiment `experiments/anomaly/20260924-020138_synthetic_anomaly/` (`configs/training/anomaly.yaml`).

**Setup:**

- **Data.** 35 days of synthetic 1-min series for 4 approaches, with injected capacity drops as
  labels. Learned detectors were fitted on normal windows of days 1–14. All detectors were
  evaluated on days 15–35 (40 incidents).
- **Impactful incidents.** Only 12 of the 40 incidents actually hurt traffic (demand ≥ 0.9 ×
  reduced capacity). The rest happened at low demand and are invisible in traffic data.
- **Alarm thresholds.** 0.5 for the detectors; 0.3 ("elevated") for the fused probability, which
  cannot exceed about 0.38 from series evidence alone.

| Detector | PR-AUC | Event recall | Recall (impactful) | False alarms / h | Median delay (min) |
|---|---|---|---|---|---|
| flow_imbalance | 0.062 | 0.13 | 0.42 | 0.013 | 2 |
| **fused probability** | 0.015 | 0.38 | **0.83** | **0.31** | 2 |
| density_zscore | 0.011 | 0.38 | 0.83 | 0.41 | 2 |
| queue_growth | 0.011 | 0.28 | 0.92 | 0.25 | 2 |
| sudden_slowdown | 0.010 | 0.40 | 0.83 | 0.49 | 3 |
| autoencoder | 0.009 | 0.05 | 0.17 | 0.03 | 7.5 |
| zscore | 0.009 | 0.55 | 0.83 | 0.65 | 4 |
| isolation_forest | 0.009 | 0.00 | 0.00 | 0.00 | – |
| ewma | 0.009 | 0.50 | 0.83 | 0.68 | 4 |

The base rate of incident windows is 0.0085 (the PR-AUC of a random score).

**How to read these numbers:**

1. **Window-level PR-AUC is low for every detector** (at most 7× the base rate). Most
   incident-labelled minutes change nothing measurable: a capacity drop at night is invisible.
   Event recall on *impactful* incidents is the meaningful number.
2. **No single detector is both sensitive and quiet.**
   - Single statistical detectors catch 83–92 % of impactful incidents within 2–4 min, but at
     0.25–0.68 false alarms per approach-hour.
   - The fused probability keeps the same recall (0.83) with fewer false alarms (0.31/h). Combining
     evidence helps, which supports the "do not rely on one signal" design.
   - 0.3 false alarms/h per approach is still too many for operators. That is why the series-only
     probability stays below "possible incident", and a camera-level physical cue (stopped vehicle,
     abnormal trajectory) is needed to reach it.
3. **The learned detectors add little on this data.** Isolation Forest never reaches the 0.5
   alarm level. The autoencoder finds 17 % of impactful incidents. Both were fitted on normal
   windows that already contain rush-hour dynamics, so capacity drops resemble ordinary peaks in
   their feature space. They stay **optional** pipeline inputs (commented in `pipeline.yaml`),
   registered and promoted so they can be switched on and re-evaluated on real recordings. This is
   reported as a negative result, not hidden.

---

## 10. Limitations

- **Uncalibrated probabilities.** The fusion weights are hand-set and the scores are not
  probabilities in the frequentist sense until they are fitted on labelled incidents. The output
  says so every time (`calibrated: false`).
- **Synthetic validation only.** The scenarios and the anomaly benchmark use simulated traffic.
  They validate logic and machinery, not real-world detection rates. There is no labelled
  real-incident dataset yet.
- **Camera-bound evidence.** Track detectors see only the calibrated view (about 60 m). A stopped
  vehicle just outside the view shows up only indirectly (queue growth, flow imbalance). Occlusion
  and ID switches can break stopped-duration counting. The detectors re-emit per track and use
  trajectory reliability, but a real tracker will make mistakes.
- **Signals.** The queue zone and `max_queue_wait_s` protect against red-phase queues. Without
  signal-phase data, a very long red or a blocked exit can still look like a stopped vehicle.
- **Time-of-day baselines.** The series detectors use a recent-window baseline, not a learned
  weekly profile, so a normal rush-hour onset produces some statistical-outlier signals (weight 1.0
  → p ≤ 0.27 alone).
- **Neighbour context is not used yet.** `neighbour_features` (upstream/downstream approaches) is in
  the fuser interface but not in the default model.
- **What-if is a static macroscopic model** on a small graph with assumed volumes on unobserved
  links (see §5).
- **The risk weights are a transparent policy, not a learned model.** They should be agreed with
  the traffic authority.

## 11. What remains (Prompt 5 and later)

- **Dashboard (React + MapLibre + ECharts).** Map colouring from `/traffic/current` and `/risk`, an
  incident panel with evidence and uncertainty, acknowledge/dismiss buttons (`PATCH /incidents`),
  a what-if panel, and live updates (implemented with REST polling; see ARCHITECTURE.md §9).
- **Live wiring.** Run the analytics worker next to the camera workers in the dev runner so the API
  shows live data without `ingest`. Redis bus for multi-process deployments. `GET /network` GeoJSON
  for the map.
- **Calibration.** Collect operator labels, then fit the fusion model and calibrate it (Platt or
  isotonic). Learn a time-of-day baseline for the series detectors. Use neighbour-approach
  consistency in the fuser.
- **Real-footage validation.** Annotated real incidents and stopped vehicles, measuring
  precision/recall, false alarms per camera-hour and detection delay. Tune `min_stopped_s` and
  queue zones per site.
