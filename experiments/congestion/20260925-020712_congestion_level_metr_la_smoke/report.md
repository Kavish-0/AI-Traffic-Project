# Congestion-level forecast: congestion_level_metr_la_smoke

Experiment `20260925-020712_congestion_level_metr_la_smoke`.

Future congestion level from the past hour of speeds (METR-LA). Compares the "no change" baseline, the production approach (forecast the speed, then apply the rules) and direct classifiers.

- Dataset: metr_la (11 series, 300 s)
- Periods: train 2012-03-01..2012-05-23, val 2012-05-23..2012-06-04, test 2012-06-04..2012-06-28
- Samples: {'train': 12016, 'val': 32543, 'test': 61657}
- Label = rule level of the observed value at t + h, indicators: ['avg_speed_kmh']
- Test class distribution: {5: {'free': 50178, 'moderate': 3362, 'high': 2887, 'severe': 5230}, 15: {'free': 50179, 'moderate': 3368, 'high': 2884, 'severe': 5226}, 30: {'free': 50191, 'moderate': 3384, 'high': 2867, 'severe': 5215}}

## +5 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| regression_to_rules | 0.904 | 0.638 | 0.666 | 0.638 | 0.650 | 0.901 | 0.654 |
| persistence_level | 0.897 | 0.649 | 0.649 | 0.649 | 0.649 | 0.897 | 0.643 |
| random_forest | 0.908 | 0.616 | 0.677 | 0.616 | 0.631 | 0.896 | 0.641 |
| xgboost | 0.905 | 0.608 | 0.653 | 0.608 | 0.621 | 0.894 | 0.623 |
| logistic_regression | 0.903 | 0.573 | 0.630 | 0.573 | 0.576 | 0.884 | 0.591 |

## +15 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| regression_to_rules | 0.877 | 0.572 | 0.610 | 0.572 | 0.588 | 0.876 | 0.595 |
| persistence_level | 0.873 | 0.589 | 0.589 | 0.589 | 0.589 | 0.873 | 0.595 |
| xgboost | 0.885 | 0.538 | 0.599 | 0.538 | 0.552 | 0.867 | 0.561 |
| random_forest | 0.889 | 0.535 | 0.624 | 0.535 | 0.537 | 0.864 | 0.541 |
| logistic_regression | 0.880 | 0.475 | 0.503 | 0.475 | 0.423 | 0.839 | 0.434 |

## +30 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| persistence_level | 0.851 | 0.543 | 0.542 | 0.543 | 0.542 | 0.851 | 0.543 |
| regression_to_rules | 0.853 | 0.512 | 0.572 | 0.512 | 0.531 | 0.853 | 0.540 |
| xgboost | 0.874 | 0.496 | 0.573 | 0.496 | 0.510 | 0.849 | 0.515 |
| random_forest | 0.875 | 0.479 | 0.593 | 0.479 | 0.475 | 0.840 | 0.477 |
| logistic_regression | 0.868 | 0.449 | 0.442 | 0.449 | 0.405 | 0.825 | 0.417 |

Figures: macro_f1_by_model.png, confusion_h5.png, confusion_h15.png, confusion_h30.png
