# Anomaly experiment `20260924-020138_synthetic_anomaly`

Synthetic junction series with injected capacity drops (ground truth). Learned detectors fitted on normal windows of the first 14 days; all detectors evaluated on the remaining days (40 injected incidents). Alarm threshold 0.5 (fused probability: 0.3); margin 15 min.

| detector | PR-AUC | event recall | recall (impactful) | false alarms / h | median delay (min) |
|---|---|---|---|---|---|
| flow_imbalance | 0.062 | 0.125 | 0.417 | 0.013 | 2.0 |
| fused_probability | 0.0154 | 0.375 | 0.833 | 0.312 | 2.0 |
| density_zscore | 0.011 | 0.375 | 0.833 | 0.408 | 2.0 |
| queue_growth | 0.0109 | 0.275 | 0.917 | 0.248 | 2.0 |
| sudden_slowdown | 0.0102 | 0.4 | 0.833 | 0.493 | 3.0 |
| autoencoder | 0.0093 | 0.05 | 0.167 | 0.026 | 7.5 |
| zscore | 0.0093 | 0.55 | 0.833 | 0.653 | 4.0 |
| isolation_forest | 0.0091 | 0.0 | 0.0 | 0.0 | None |
| ewma | 0.009 | 0.5 | 0.833 | 0.681 | 4.0 |

Incident share of test windows (PR-AUC of a random score): 0.0085.

Caveat: synthetic capacity drops in a queueing model - this validates the detectors' behaviour and the evaluation machinery, not real-world detection rates. Incidents during low demand do not change traffic and are undetectable from traffic data (see 'recall (impactful)').
