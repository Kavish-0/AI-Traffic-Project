# Congestion-level forecast: congestion_level_pems_bay

Experiment `20260925-021104_congestion_level_pems_bay`.

Future congestion level from the past hour of speeds (PEMS-BAY). Compares the "no change" baseline, the production approach (forecast the speed, then apply the rules) and direct classifiers.

- Dataset: pems_bay (11 series, 300 s)
- Periods: train 2017-01-01..2017-05-08, val 2017-05-08..2017-05-26, test 2017-05-26..2017-07-01
- Samples: {'train': 401104, 'val': 57255, 'test': 114598}
- Label = rule level of the observed value at t + h, indicators: ['avg_speed_kmh']
- Test class distribution: {5: {'free': 106492, 'moderate': 2931, 'high': 2451, 'severe': 2724}, 15: {'free': 106495, 'moderate': 2930, 'high': 2449, 'severe': 2724}, 30: {'free': 106497, 'moderate': 2929, 'high': 2448, 'severe': 2724}}

## +5 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| regression_to_rules | 0.982 | 0.844 | 0.852 | 0.844 | 0.848 | 0.982 | 0.851 |
| xgboost | 0.982 | 0.840 | 0.851 | 0.840 | 0.845 | 0.982 | 0.847 |
| random_forest | 0.982 | 0.841 | 0.852 | 0.841 | 0.846 | 0.982 | 0.847 |
| logistic_regression | 0.982 | 0.826 | 0.849 | 0.826 | 0.837 | 0.981 | 0.844 |
| persistence_level | 0.980 | 0.829 | 0.829 | 0.829 | 0.829 | 0.980 | 0.829 |

## +15 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| random_forest | 0.967 | 0.702 | 0.743 | 0.702 | 0.720 | 0.965 | 0.736 |
| xgboost | 0.967 | 0.707 | 0.743 | 0.707 | 0.723 | 0.966 | 0.735 |
| regression_to_rules | 0.965 | 0.709 | 0.726 | 0.709 | 0.715 | 0.965 | 0.725 |
| persistence_level | 0.963 | 0.697 | 0.697 | 0.697 | 0.697 | 0.963 | 0.700 |
| logistic_regression | 0.964 | 0.666 | 0.720 | 0.666 | 0.689 | 0.962 | 0.700 |

## +30 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| xgboost | 0.958 | 0.626 | 0.698 | 0.626 | 0.656 | 0.955 | 0.680 |
| random_forest | 0.958 | 0.620 | 0.701 | 0.620 | 0.651 | 0.954 | 0.675 |
| regression_to_rules | 0.952 | 0.621 | 0.664 | 0.621 | 0.632 | 0.952 | 0.644 |
| persistence_level | 0.949 | 0.615 | 0.615 | 0.615 | 0.615 | 0.949 | 0.612 |
| logistic_regression | 0.949 | 0.532 | 0.607 | 0.532 | 0.560 | 0.944 | 0.586 |

Figures: macro_f1_by_model.png, confusion_h5.png, confusion_h15.png, confusion_h30.png
