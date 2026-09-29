# Traffic forecasting (Phase 5)

This layer predicts future traffic states **+5, +15 and +30 minutes** ahead
from recent history, weather and the calendar. It then turns the predictions
into the same congestion levels (FREE / MODERATE / HIGH / SEVERE) that the
live pipeline reports.

```
PAST TRAFFIC (TrafficState windows)   weather (Open-Meteo)   calendar (local time, holidays)
            │                                 │                        │
            └────────────── canonical time series (series_id, timestamp, features) ────┘
                                              │
                 regularise grid ─ short gaps interpolated ─ long gaps excluded
                                              │
                chronological split  train │ val │ test   (fit scaling on train only)
                                              │
                 identical windows for every model (inputs t-L+1..t, targets t+5/15/30 min)
                                              │
   persistence  seasonal-naive  linear  random forest  XGBoost  LSTM  GRU  Transformer
                                              │
              MAE / RMSE / R² / skill-vs-persistence per horizon  →  comparison + plots
                                              │
          registry (models/forecasting/<model>/<version>/)  →  ForecastingService
                                              │
                    {"current_state": ..., "forecast": {"5_min": ..., ...}, "predictions": ...}
```

---

## 1. Why these models are compared

The research question is **whether added model complexity buys forecast
accuracy on traffic data, and at which horizons.** The candidates therefore
form a ladder of increasing complexity, and every rung has to justify itself
against the rungs below it.

| Rung | Model | What it can represent | Why it is included |
|---|---|---|---|
| 0 | **Persistence** ("no change") | nothing: y(t+h) = y(t) | The naive reference. At 5 min it is often hard to beat. A model with **skill ≤ 0** against it is useless. |
| 0 | **Seasonal naive** (same time yesterday) | the daily cycle only | Tests whether "typical behaviour at this time of day" beats "current state". |
| 1 | **Linear regression (OLS)** — *the learned baseline* | a weighted sum of recent lags, rolling means, calendar and weather | The simplest learned model. Anything more complex must beat it to be worth its cost. |
| 2 | **Random Forest** | non-linear, threshold effects, interactions (e.g. rain × rush hour) | Robust and nearly tuning-free; shows what non-linearity alone adds. |
| 2 | **XGBoost** | the same, boosted and regularised, with early stopping | Usually the strongest tabular model; the benchmark the deep models must beat. |
| 3 | **LSTM** | learns its own temporal features from raw sequences (gated memory) | Tests whether learned sequence representations beat hand-crafted lags. |
| 3 | **GRU** | the same with fewer parameters (no separate cell state) | Tests whether the LSTM's extra gate is worth it at this data scale. |
| 3 | **Temporal Transformer** | self-attention over the whole input window | Tests whether attention beats recurrence. Kept small (about 70k parameters, the LSTM's size class). |

The tabular models (rungs 1–2) receive the **same engineered features**:
lagged values, rolling means, first differences, calendar and weather. The
deep models (rung 3) receive the **raw window** of the same input columns.
Differences between rungs therefore come from the model class, not from extra
information.

### Advantages and disadvantages

| Model | Advantages | Disadvantages |
|---|---|---|
| Persistence | zero cost; excellent at very short horizons; always available | ignores trends and daily patterns; its error grows fast with horizon |
| Seasonal naive | captures the daily rhythm; zero cost | blind to today's deviations (incidents, weather, weekday vs weekend) |
| Linear | fast (seconds), interpretable coefficients, stable, little data needed | cannot represent regime changes (free flow ↔ queueing) or interactions |
| Random Forest | non-linear, robust to outliers, minimal tuning | large on disk; cannot extrapolate beyond the training range; slower inference |
| XGBoost | usually the most accurate tabular model; early stopping; fast on GPU | more hyper-parameters; engineered lags limit how far back it "sees" |
| LSTM / GRU | learns temporal features itself; natural for sequences | needs more data and GPU time; sensitive to seeds and learning rate; less interpretable |
| Transformer | attends to any past step directly; parallel training | the most data-hungry; small datasets rarely justify it; least interpretable |

---

## 2. Datasets

| Dataset | Real? | Resolution | Series × length | Features | Role |
|---|---|---|---|---|---|
| **PeMS08** (Caltrans District 8, San Bernardino CA, Jul–Aug 2016) + **Open-Meteo** weather | **real** | 5 min | 12 detector stations × 62 days | flow, speed, occupancy + temperature, humidity, rain, weather condition | **The model comparison.** Answers "does complexity help?" on real traffic. |
| **Synthetic junction** (4 approaches) + **real** Bengaluru weather (Jun–Jul 2025) | synthetic traffic | 1 min | 4 approaches × 42 days | the full pipeline schema: speed, density, occupancy, queue, flow, counts, inflow, outflow, stopped vehicles | Trains the model **served by the forecasting service**, because it matches the camera pipeline's `TrafficState` output. Not evidence about real traffic. |
| **Own observations** (`runs/*/traffic_states.jsonl`) | real | 1 min | — | full schema | Supported by the loader (`source: observations`), but not enough recorded yet. |

Notes on PeMS08:

- Downloaded from the ASTGCN authors' mirror, hash-pinned
  (`configs/datasets/public_timeseries.yaml`).
- Units are converted to the pipeline's (veh/h, %, km/h).
- Rows with flow = 0 **and** occupancy = 0 are treated as detector outages
  (0.35%).
- Visibility is not available in the weather archive, so it is dropped
  automatically.

The synthetic generator (`forecasting/data/synthetic_series.py`) models:

- weekday double-peak, weekend and holiday demand profiles
- signal capacity and point queues
- rain-dependent capacity and speed, driven by real weather
- random incidents, measurement noise and sensor outages

---

## 3. Pipeline details

| Step | Rule | Where |
|---|---|---|
| Missing values | Each series is re-indexed to a regular grid. Traffic gaps of ≤ `max_gap_fill_steps` are linearly interpolated; **longer gaps stay missing** and every window touching them is excluded (nothing is invented). Weather is time-interpolated or forward-filled. Remaining context gaps get the training mean. | `preprocess.py` |
| Temporal features | Time of day and day of week as sin/cos, weekend, **holiday** (the `holidays` package, per country), **rush hour** (configurable local-time windows, weekdays only). All in local time. | `context.py` |
| Weather | temperature, humidity, rain, visibility (dropped if absent), weather condition (WMO code → clear / cloudy / fog / drizzle / rain / snow / thunderstorm, one-hot) | `context.py` |
| Scaling | z-score per traffic and numeric weather column, **with statistics from the training period only**; stored as JSON next to every model | `preprocess.py` |
| **Chronological split** | train / validation / test are consecutive time periods, **never shuffled**. A sample belongs to a period only if its furthest target is also inside it, so targets never leak across the boundary. Input windows may reach back into the previous period: that is legitimate past data. | `windows.py` |
| Identical samples | Windows are built once, with the longest lookback any model needs. **Every model is scored on exactly the same test origins.** | `windows.py` |
| Sequence generation | `[N, L, F]` windows gathered on the GPU from one array; `L` (`lookback_steps`) is set per model in the config, e.g. `lstm` 12 vs `lstm_L24` 24 steps | `_torch.py` |
| Multi-horizon | *Direct* strategy: one model outputs all horizons × targets at once, with an explicit horizon axis `[N, H, K]` | `interfaces.py` |
| Targets | Learned models predict the **change** from the current value (`target_transform: delta`) and return levels | `interfaces.py` |
| Model selection | by **validation** RMSE on the primary target; the test period is only used for reporting | `experiment.py` |
| Seeds | Deep models are trained with 3 seeds; the best-validation seed is registered, and mean ± std over seeds is reported | `experiment.py` |

Mini-batches of *training* windows are shuffled for SGD. This is standard and
does not leak anything: each window only contains data from before its own
targets, and validation and test periods lie strictly after the training
period.

---

## 4. Metrics

All metrics are in original units, after inverse scaling:

- **MAE** — average absolute error.
- **RMSE** — penalises large misses, such as a missed congestion onset.
- **R²** — variance explained on the test period.
- **Skill** — 1 − RMSE / RMSE(persistence). This is the key number for the
  research question: above 0 means better than assuming "no change".

---

## 5. Commands

```bash
cd backend
pip install -e ".[dev,ml]"

# data
python -m traffic_intel.datasets --catalog timeseries download pems08

# train + evaluate (every run: new experiment id + new model versions; nothing overwritten)
python -m traffic_intel.ai.training train forecasting --config ../configs/training/forecasting_pems08.yaml
python -m traffic_intel.ai.training train forecasting --config ../configs/training/forecasting.yaml --promote best-val
python -m traffic_intel.ai.training train forecasting --config ... --only persistence,linear,xgboost   # subset

# evaluate / compare
python -m traffic_intel.ai.training evaluate forecasting --experiment <experiment_id>   # tables
python -m traffic_intel.ai.training list forecasting                                   # all runs (index.csv)
python -m traffic_intel.ai.training promote forecasting <model> <version|latest>

# inference
python -m traffic_intel.forecasting example --at 2025-07-08T08:00:00+05:30 --approach junction_a_north
python -m traffic_intel.forecasting predict --history <traffic_states.jsonl>
```

```python
from traffic_intel.forecasting.service import ForecastingService
svc = ForecastingService.from_registry()        # PRODUCTION model
result = svc.predict(history)                   # list of TrafficState (dicts or models), >= required_history_steps
result.model_dump()                             # current_state, forecast{5_min,15_min,30_min}, predictions, explanations
result.to_forecast()                            # system-wide Forecast schema (bus / API / DB)
```

### Outputs

| What | Where |
|---|---|
| Experiment record | `experiments/forecasting/<id>/`: `config.yaml`, `dataset.json` (periods, features, samples), `models.json` (hyper-parameters, fit time, epochs, parameters), `metrics_test.csv`, `metrics_val.csv`, `seed_runs.csv`, `comparison.md`, `plots/` |
| Run index (append-only) | `experiments/forecasting/index.csv` |
| Model registry | `models/forecasting/<model>/<version>/` containing `params.json`, `model.*`, `preprocessor.json`, `metadata.json`, plus `models/forecasting/<model>/PRODUCTION` |

---

## 6. Results

Full records: `experiments/forecasting/20260924-011513_pems08_benchmark/comparison.md` (real data) and
`experiments/forecasting/20260924-012321_synthetic_junction/comparison.md` (synthetic data, the run that
registered the PRODUCTION model). All models use the same chronological split and the same test origins.

### 6.1 PeMS08 (real freeway detectors), target `flow_veh_h`, test RMSE (veh/h)

| model | +5 min | +15 min | +30 min | mean | skill vs persistence |
|---|---|---|---|---|---|
| **xgboost** | **212.9** | **242.1** | **272.5** | **242.5** | 0.231 |
| transformer | 218.4 | 253.2 | 284.5 | 252.0 | 0.202 |
| gru | 219.0 | 254.6 | 286.1 | 253.3 | 0.199 |
| lstm_L24 (24-step input) | 222.1 | 257.5 | 289.1 | 256.3 | 0.189 |
| lstm (12-step input) | 225.7 | 260.1 | 292.6 | 259.5 | 0.178 |
| random_forest | 228.2 | 264.8 | 296.0 | 263.0 | 0.167 |
| linear | 228.0 | 279.9 | 342.4 | 283.4 | 0.112 |
| persistence | 252.9 | 312.1 | 396.0 | 320.3 | 0.000 |
| seasonal_naive | 670.9 | 670.8 | 670.7 | 670.8 | -1.165 |

Seed spread of the deep models (3 seeds, mean test RMSE ± std): GRU 256.9 ± 2.1, LSTM 262.8 ± 1.6,
LSTM_L24 259.2 ± 1.5, Transformer 255.8 ± 3.0. For speed and occupancy, random forest and XGBoost are
level with each other and ahead of the deep models (speed RMSE 3.48 / 3.50 km/h, occupancy 1.29 / 1.32 %).

### 6.2 Synthetic junction (schema-compatible, 1-minute), target `avg_speed_kmh`, test RMSE averaged over horizons

| model | MAE | RMSE | R² |
|---|---|---|---|
| **xgboost** (PRODUCTION) | 1.887 | **3.354** | 0.844 |
| random_forest | 1.982 | 3.449 | 0.835 |
| gru | 1.909 | 3.457 | 0.834 |
| lstm | 2.166 | 3.638 | 0.816 |
| transformer | 2.037 | 3.680 | 0.811 |
| linear | 2.194 | 3.822 | 0.794 |
| persistence | 2.612 | 5.100 | 0.628 |
| seasonal_naive | 3.509 | 7.334 | 0.256 |

### 6.3 Interpretation (what the numbers do and do not show)

- Every learned model beats persistence, and the gap grows with the horizon. On PeMS08, XGBoost is 16 %
  better than persistence at +5 min and 31 % better at +30 min.
- **The deep models do not beat XGBoost here.** The best deep model (Transformer) is 3.9 % worse than
  XGBoost on mean flow RMSE. Even allowing one seed standard deviation, no deep model's mean over seeds
  gets below XGBoost. This result holds for this dataset size (about 138 k training windows, 12 stations,
  5-minute data) and these hyper-parameters. It is not a general ranking.
- Doubling the LSTM input from 12 to 24 steps gives a small gain (259.5 → 256.3). That gain is within
  about two seed standard deviations.
- Linear regression is a strong baseline at +5 min but falls behind at +30 min, where non-linear effects
  such as peak onset matter more.
- Synthetic results only validate the pipeline and give the service a model it can load. They are not
  evidence about real traffic.
- Cost: XGBoost trained in 39 s. GRU and Transformer took 84–91 s on the GPU. All models predict in well
  under 0.1 ms per sample.
