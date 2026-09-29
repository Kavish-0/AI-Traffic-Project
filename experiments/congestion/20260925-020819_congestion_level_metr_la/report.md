# Congestion-level forecast: congestion_level_metr_la

Experiment `20260925-020819_congestion_level_metr_la`.

Future congestion level from the past hour of speeds (METR-LA). Compares the "no change" baseline, the production approach (forecast the speed, then apply the rules) and direct classifiers.

- Dataset: metr_la (11 series, 300 s)
- Periods: train 2012-03-01..2012-05-23, val 2012-05-23..2012-06-04, test 2012-06-04..2012-06-28
- Samples: {'train': 240241, 'val': 32543, 'test': 61657}
- Label = rule level of the observed value at t + h, indicators: ['avg_speed_kmh']
- Test class distribution: {5: {'free': 50178, 'moderate': 3362, 'high': 2887, 'severe': 5230}, 15: {'free': 50179, 'moderate': 3368, 'high': 2884, 'severe': 5226}, 30: {'free': 50191, 'moderate': 3384, 'high': 2867, 'severe': 5215}}

## +5 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| random_forest | 0.895 | 0.681 | 0.655 | 0.681 | 0.666 | 0.900 | 0.672 |
| regression_to_rules | 0.908 | 0.654 | 0.679 | 0.654 | 0.666 | 0.905 | 0.661 |
| xgboost | 0.912 | 0.635 | 0.687 | 0.635 | 0.650 | 0.902 | 0.650 |
| persistence_level | 0.897 | 0.649 | 0.649 | 0.649 | 0.649 | 0.897 | 0.643 |
| logistic_regression | 0.830 | 0.692 | 0.616 | 0.692 | 0.633 | 0.860 | 0.638 |

## +15 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| random_forest | 0.874 | 0.637 | 0.608 | 0.637 | 0.621 | 0.879 | 0.620 |
| regression_to_rules | 0.880 | 0.581 | 0.621 | 0.581 | 0.597 | 0.880 | 0.603 |
| persistence_level | 0.873 | 0.589 | 0.589 | 0.589 | 0.589 | 0.873 | 0.595 |
| xgboost | 0.892 | 0.559 | 0.636 | 0.559 | 0.577 | 0.875 | 0.577 |
| logistic_regression | 0.802 | 0.623 | 0.549 | 0.623 | 0.569 | 0.834 | 0.576 |

## +30 min (test; sorted by validation macro-F1)

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | val_macro_f1 |
|---|---|---|---|---|---|---|---|
| random_forest | 0.855 | 0.590 | 0.562 | 0.590 | 0.575 | 0.859 | 0.579 |
| regression_to_rules | 0.858 | 0.527 | 0.588 | 0.527 | 0.547 | 0.859 | 0.547 |
| persistence_level | 0.851 | 0.543 | 0.542 | 0.543 | 0.542 | 0.851 | 0.543 |
| xgboost | 0.879 | 0.510 | 0.606 | 0.510 | 0.531 | 0.855 | 0.527 |
| logistic_regression | 0.759 | 0.553 | 0.480 | 0.553 | 0.500 | 0.799 | 0.520 |

Figures: macro_f1_by_model.png, confusion_h5.png, confusion_h15.png, confusion_h30.png
