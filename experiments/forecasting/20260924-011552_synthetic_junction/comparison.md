# Forecasting experiment `20260924-011552_synthetic_junction`

6 weeks x 4 approaches of synthetic 1-minute traffic states (demand profiles, signal capacity, point queues, incidents, sensor gaps) driven by REAL Bengaluru weather (Open-Meteo, Jun-Jul 2025). Synthetic: validates the pipeline and gives the service a schema-compatible model; it is not evidence about real traffic (see forecasting_pems08.yaml for that).

- **Dataset:** synthetic (synthetic), 4 series, resolution 60 s
- **Periods (chronological):** train 2025-06-01T18:30:00+00:00 -> 2025-06-26T23:18:00+00:00; val 2025-06-26T23:18:00+00:00 -> 2025-07-05T08:54:00+00:00; test 2025-07-05T08:54:00+00:00 -> 2025-07-13T18:30:00+00:00
- **Samples:** {'train': 71809, 'val': 47994, 'test': 48185} (identical test origins for every model)
- **Inputs (23):** avg_speed_kmh, density_ratio, occupancy_pct, queue_length_m, flow_veh_h, vehicle_count, inflow, outflow, stopped_vehicles, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_holiday, is_rush_hour, temperature_c, humidity_pct, rain_mm, weather_clear, weather_cloudy, weather_drizzle, weather_rain
- **Targets:** avg_speed_kmh, density_ratio, occupancy_pct, queue_length_m, flow_veh_h; horizons [5, 15, 30] min
- **Model selection:** by validation RMSE on `avg_speed_kmh` -> **xgboost**

## avg_speed_kmh (km/h) - test, averaged over horizons

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| xgboost | 1.887 | 3.354 | 0.844 | 0.327 |
| random_forest | 1.982 | 3.449 | 0.835 | 0.309 |
| gru | 1.909 | 3.457 | 0.834 | 0.309 |
| lstm | 2.166 | 3.638 | 0.816 | 0.273 |
| transformer | 2.037 | 3.680 | 0.811 | 0.268 |
| linear | 2.194 | 3.822 | 0.794 | 0.244 |
| persistence | 2.612 | 5.100 | 0.628 | 0.000 |
| seasonal_naive | 3.509 | 7.334 | 0.256 | -0.487 |

## density_ratio (ratio) - test, averaged over horizons

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| xgboost | 0.031 | 0.081 | 0.817 | 0.305 |
| random_forest | 0.031 | 0.081 | 0.817 | 0.306 |
| gru | 0.031 | 0.084 | 0.800 | 0.279 |
| lstm | 0.033 | 0.085 | 0.798 | 0.275 |
| transformer | 0.034 | 0.089 | 0.776 | 0.244 |
| linear | 0.041 | 0.094 | 0.750 | 0.209 |
| persistence | 0.041 | 0.121 | 0.570 | 0.000 |
| seasonal_naive | 0.063 | 0.185 | 0.048 | -0.644 |

## occupancy_pct (%) - test, averaged over horizons

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| xgboost | 2.444 | 6.510 | 0.813 | 0.301 |
| random_forest | 2.440 | 6.533 | 0.811 | 0.299 |
| gru | 2.474 | 6.832 | 0.793 | 0.271 |
| lstm | 2.613 | 6.841 | 0.792 | 0.269 |
| transformer | 2.661 | 7.175 | 0.770 | 0.239 |
| linear | 3.239 | 7.526 | 0.745 | 0.206 |
| persistence | 3.276 | 9.680 | 0.564 | 0.000 |
| seasonal_naive | 5.040 | 14.9 | 0.030 | -0.651 |

## queue_length_m (m) - test, averaged over horizons

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| random_forest | 1.392 | 5.651 | 0.749 | 0.288 |
| xgboost | 1.364 | 5.662 | 0.748 | 0.285 |
| gru | 1.553 | 5.900 | 0.725 | 0.259 |
| lstm | 1.740 | 5.901 | 0.725 | 0.259 |
| transformer | 1.733 | 6.132 | 0.701 | 0.234 |
| linear | 2.460 | 6.422 | 0.670 | 0.201 |
| persistence | 1.542 | 8.204 | 0.446 | 0.000 |
| seasonal_naive | 2.858 | 12.2 | -0.151 | -0.583 |

## flow_veh_h (veh/h) - test, averaged over horizons

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| xgboost | 116.8 | 163.8 | 0.819 | 0.309 |
| gru | 118.9 | 165.9 | 0.814 | 0.301 |
| lstm | 121.6 | 169.9 | 0.805 | 0.284 |
| transformer | 121.2 | 171.1 | 0.802 | 0.279 |
| random_forest | 121.1 | 172.7 | 0.798 | 0.272 |
| linear | 123.9 | 173.6 | 0.796 | 0.270 |
| persistence | 165.7 | 238.3 | 0.614 | 0.000 |
| seasonal_naive | 194.6 | 300.3 | 0.391 | -0.269 |

## avg_speed_kmh by horizon (test)

**+5 min**

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| xgboost | 1.748 | 3.067 | 0.870 | 0.237 |
| gru | 1.763 | 3.096 | 0.867 | 0.230 |
| random_forest | 1.832 | 3.152 | 0.863 | 0.216 |
| transformer | 1.845 | 3.184 | 0.860 | 0.208 |
| linear | 1.825 | 3.202 | 0.858 | 0.204 |
| lstm | 1.969 | 3.266 | 0.852 | 0.188 |
| persistence | 2.196 | 4.022 | 0.776 | 0.000 |
| seasonal_naive | 3.509 | 7.334 | 0.256 | -0.824 |

**+15 min**

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| xgboost | 1.909 | 3.418 | 0.838 | 0.312 |
| gru | 1.916 | 3.502 | 0.830 | 0.295 |
| random_forest | 1.999 | 3.504 | 0.830 | 0.294 |
| lstm | 2.190 | 3.666 | 0.814 | 0.262 |
| transformer | 2.049 | 3.709 | 0.810 | 0.253 |
| linear | 2.160 | 3.801 | 0.800 | 0.235 |
| persistence | 2.538 | 4.965 | 0.659 | 0.000 |
| seasonal_naive | 3.508 | 7.334 | 0.256 | -0.477 |

**+30 min**

| model | MAE | RMSE | R2 | skill |
|---|---|---|---|---|
| xgboost | 2.005 | 3.578 | 0.823 | 0.433 |
| random_forest | 2.114 | 3.690 | 0.812 | 0.416 |
| gru | 2.047 | 3.772 | 0.803 | 0.402 |
| lstm | 2.340 | 3.980 | 0.781 | 0.369 |
| transformer | 2.217 | 4.147 | 0.762 | 0.343 |
| linear | 2.596 | 4.463 | 0.725 | 0.293 |
| persistence | 3.103 | 6.313 | 0.449 | 0.000 |
| seasonal_naive | 3.508 | 7.334 | 0.256 | -0.162 |

## Seed variability of deep models (avg_speed_kmh, test RMSE, mean over horizons)

| model | mean | std | min | max | seeds |
|---|---|---|---|---|---|
| gru | 3.491 | 0.041 | 3.468 | 3.538 | 3 |
| lstm | 3.677 | 0.140 | 3.553 | 3.830 | 3 |
| transformer | 3.665 | 0.051 | 3.607 | 3.701 | 3 |

The tables above use, for each deep model, the seed with the best *validation* RMSE.

## Observations (derived automatically from the numbers above)

- +5 min: lowest test RMSE = **xgboost** (3.067); persistence 4.022; linear 3.202
- +15 min: lowest test RMSE = **xgboost** (3.418); persistence 4.965; linear 3.801
- +30 min: lowest test RMSE = **xgboost** (3.578); persistence 6.313; linear 4.463
- Best deep model gru vs best tree model xgboost: RMSE 3.457 vs 3.354 (tree better by 3.1%).
- Best tree model xgboost vs linear baseline: +12.2% RMSE reduction.
- Models beating persistence on average: linear, random_forest, xgboost, lstm, gru, transformer.
- Deep models whose mean-over-seeds RMSE (+1 std) is below the best tree model (xgboost): none.

## Training cost

| model | family | fit (s) | predict (ms/sample) | epochs | parameters |
|---|---|---|---|---|---|
| persistence | baseline | 0.0 | 0.0002 | - | - |
| seasonal_naive | baseline | 0.0 | 0.0004 | - | - |
| linear | linear | 1.44 | 0.0057 | - | - |
| random_forest | tree | 19.37 | 0.0064 | - | - |
| xgboost | tree | 18.92 | 0.0055 | - | - |
| lstm | deep | 12.04 | 0.004 | 12 | 57167 |
| gru | deep | 21.19 | 0.0037 | 23 | 43151 |
| transformer | deep | 43.45 | 0.0094 | 14 | 70671 |

## Figures

![rmse_by_horizon_avg_speed_kmh.png](plots/rmse_by_horizon_avg_speed_kmh.png)
![rmse_by_horizon_density_ratio.png](plots/rmse_by_horizon_density_ratio.png)
![rmse_by_horizon_occupancy_pct.png](plots/rmse_by_horizon_occupancy_pct.png)
![rmse_by_horizon_queue_length_m.png](plots/rmse_by_horizon_queue_length_m.png)
![rmse_by_horizon_flow_veh_h.png](plots/rmse_by_horizon_flow_veh_h.png)
![pred_vs_actual_linear_avg_speed_kmh.png](plots/pred_vs_actual_linear_avg_speed_kmh.png)
![pred_vs_actual_random_forest_avg_speed_kmh.png](plots/pred_vs_actual_random_forest_avg_speed_kmh.png)
![pred_vs_actual_xgboost_avg_speed_kmh.png](plots/pred_vs_actual_xgboost_avg_speed_kmh.png)
![pred_vs_actual_lstm_avg_speed_kmh.png](plots/pred_vs_actual_lstm_avg_speed_kmh.png)
![pred_vs_actual_gru_avg_speed_kmh.png](plots/pred_vs_actual_gru_avg_speed_kmh.png)
![pred_vs_actual_transformer_avg_speed_kmh.png](plots/pred_vs_actual_transformer_avg_speed_kmh.png)
