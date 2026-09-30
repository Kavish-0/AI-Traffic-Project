# Congestion-level forecast: congestion_level_synthetic

Experiment `20260925-021745_congestion_level_synthetic`.

Future congestion level on the live feature set (synthetic junction, 1-min windows).

- Dataset: synthetic (4 series, 60 s)
- Periods: train 2025-06-01..2025-06-26, val 2025-06-26..2025-07-05, test 2025-07-05..2025-07-13
- Samples: {'train': 71809, 'val': 47994, 'test': 48185}
- Label = rule level of the observed value at t + h, indicators: ['avg_speed_kmh', 'density_ratio', 'occupancy_pct', 'queue_length_m']
- Test class distribution: {5: {'free': 44889, 'moderate': 949, 'high': 361, 'severe': 1986}, 15: {'free': 44889, 'moderate': 949, 'high': 361, 'severe': 1986}, 30: {'free': 44889, 'moderate': 949, 'high': 361, 'severe': 1986}}

## +5 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| regression_to_rules | 0.960 | 0.607 | 0.575 | 0.607 | 0.586 | 0.964 | 0.563 |
| persistence_level | 0.961 | 0.562 | 0.562 | 0.562 | 0.562 | 0.961 | 0.558 |
| random_forest | 0.963 | 0.617 | 0.551 | 0.617 | 0.571 | 0.964 | 0.555 |
| xgboost | 0.970 | 0.550 | 0.604 | 0.550 | 0.555 | 0.965 | 0.544 |
| logistic_regression | 0.896 | 0.625 | 0.549 | 0.625 | 0.547 | 0.930 | 0.540 |

## +15 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| regression_to_rules | 0.953 | 0.573 | 0.535 | 0.573 | 0.549 | 0.958 | 0.526 |
| random_forest | 0.946 | 0.587 | 0.489 | 0.587 | 0.523 | 0.951 | 0.520 |
| xgboost | 0.966 | 0.510 | 0.531 | 0.510 | 0.513 | 0.960 | 0.512 |
| logistic_regression | 0.879 | 0.597 | 0.506 | 0.597 | 0.510 | 0.919 | 0.498 |
| persistence_level | 0.949 | 0.495 | 0.495 | 0.495 | 0.495 | 0.949 | 0.485 |

## +30 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| regression_to_rules | 0.951 | 0.567 | 0.526 | 0.567 | 0.541 | 0.956 | 0.524 |
| xgboost | 0.966 | 0.509 | 0.549 | 0.509 | 0.510 | 0.959 | 0.502 |
| random_forest | 0.951 | 0.557 | 0.475 | 0.557 | 0.508 | 0.951 | 0.500 |
| logistic_regression | 0.865 | 0.584 | 0.466 | 0.584 | 0.476 | 0.908 | 0.463 |
| persistence_level | 0.931 | 0.415 | 0.415 | 0.415 | 0.415 | 0.931 | 0.417 |

Figures: macro_f1_by_model.png, confusion_h5.png, confusion_h15.png, confusion_h30.png
