# Incident detection on real video: 20260930-012228_aic21_track4

AI City 2021 Track 4 training videos, split by video. Pixel-space features (uncalibrated cameras);
the fitted models are evaluation artefacts, not deployed.

- Detector: `models/detection/yolo11m.pt` @ 1280, 5.0 processed fps
- Videos: 100; windows {'train': 10438, 'val': 3532, 'test': 3535}; positive windows {'train': 1309, 'val': 262, 'test': 342}
- Split by video: train 60 (11 with an incident), val 20 (3 with an incident), test 20 (3 with an incident)

| method | window_roc_auc | window_pr_auc | window_precision | window_recall | window_f1 | event_precision | event_recall | event_f1 | event_median_delay_s |
|---|---|---|---|---|---|---|---|---|---|
| stationary_rule | 0.987 | 0.891 | 0.703 | 0.956 | 0.810 | 0.222 | 0.667 | 0.333 | 33.500 |
| isolation_forest | 0.846 | 0.596 | 0.712 | 0.702 | 0.707 | 0.333 | 0.667 | 0.444 | 93.500 |
| autoencoder | 0.925 | 0.659 | 0.709 | 0.883 | 0.786 | 0.111 | 0.333 | 0.167 | 30.000 |

Thresholds above: best window-level F1 on validation windows.

Per-video alarms with the threshold chosen for alarms (best per-video F1 on the validation videos, lowest such threshold):

| method | alarm_threshold | alarm_tp | alarm_fp | alarm_fn | alarm_precision | alarm_recall | alarm_f1 | alarm_median_delay_s |
|---|---|---|---|---|---|---|---|---|
| stationary_rule | 45.800 | 2.000 | 4.000 | 1.000 | 0.333 | 0.667 | 0.444 | 48.500 |
| isolation_forest | 0.003 | 2.000 | 4.000 | 1.000 | 0.333 | 0.667 | 0.444 | 93.500 |
| autoencoder | 0.085 | 1.000 | 7.000 | 2.000 | 0.125 | 0.333 | 0.182 | 40.000 |

AIC21 timing score (stationary rule, onset back-dated): `{'tp': 2, 'fp': 7, 'fn': 2, 'precision': 0.2222, 'recall': 0.5, 'f1': 0.3077, 'rmse_s': 0.707, 'nrmse': 0.0024, 's4': 0.307}`

... with the alarm threshold: `{'tp': 2, 'fp': 4, 'fn': 2, 'precision': 0.3333, 'recall': 0.5, 'f1': 0.4, 'rmse_s': 1.581, 'nrmse': 0.0053, 's4': 0.3979}`

Calibration: `{'method': 'stationary_rule', 'fitted_on': 'validation windows', 'evaluated_on': 'test windows', 'brier_uncalibrated_scaled': 0.030865397244398646, 'brier_platt': 0.040292023285332115, 'brier_isotonic': 0.03935496008968708, 'brier_constant_base_rate': 0.08789618013709066}`
