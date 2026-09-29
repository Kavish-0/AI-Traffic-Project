# Incident detection on real video: 20260925-021232_aic21_track4_smoke

AI City 2021 Track 4 training videos, split by video. Pixel-space features (uncalibrated cameras);
the fitted models are evaluation artefacts, not deployed.

- Detector: `models/detection/yolo11m.pt` @ 832, 5.0 processed fps
- Videos: 6; windows {'train': 354, 'val': 354, 'test': 353}; positive windows {'train': 60, 'val': 176, 'test': 57}
- Split (labelled videos): train 2, val 2, test 2

| method | window_roc_auc | window_pr_auc | window_precision | window_recall | window_f1 | event_precision | event_recall | event_f1 | event_median_delay_s |
|---|---|---|---|---|---|---|---|---|---|
| stationary_rule | 0.492 | 0.161 | 0.161 | 1.000 | 0.278 | 0.500 | 1.000 | 0.667 | 10.000 |
| isolation_forest | 0.500 | 0.161 | 0.161 | 1.000 | 0.278 | 0.500 | 1.000 | 0.667 | 10.000 |
| autoencoder | 0.913 | 0.522 | 0.510 | 0.930 | 0.658 | 1.000 | 1.000 | 1.000 | 20.000 |

AIC21 timing score (stationary rule, onset back-dated): `{'tp': 1, 'fp': 1, 'fn': 0, 'precision': 0.5, 'recall': 1.0, 'f1': 0.6667, 'rmse_s': 10.0, 'nrmse': 0.0333, 's4': 0.6444}`

Calibration: `{'method': 'autoencoder', 'fitted_on': 'validation windows', 'evaluated_on': 'test windows', 'brier_uncalibrated_scaled': 0.1138285829540425, 'brier_platt': 0.10541505341341068, 'brier_isotonic': 0.11960058094323406, 'brier_constant_base_rate': 0.24809539840224426}`
