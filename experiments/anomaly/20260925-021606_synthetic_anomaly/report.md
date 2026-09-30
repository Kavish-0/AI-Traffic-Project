# Anomaly experiment `20260925-021606_synthetic_anomaly`

Synthetic junction series with injected capacity drops (ground truth). Learned detectors fitted on normal windows of the first 14 days; all detectors evaluated on the remaining days (40 injected incidents). Alarm threshold 0.5 (fused probability: 0.3); margin 15 min.

| detector | PR-AUC | ROC-AUC | precision | recall | F1 | event recall | recall (impactful) | false alarms / h | median delay (min) |
|---|---|---|---|---|---|---|---|---|---|
| flow_imbalance | 0.062 | 0.5401 | 0.5949 | 0.0459 | 0.0852 | 0.125 | 0.417 | 0.013 | 2.0 |
| fused_probability | 0.0154 | 0.5712 | 0.0337 | 0.0225 | 0.0269 | 0.375 | 0.833 | 0.312 | 2.0 |
| density_zscore | 0.011 | 0.5235 | 0.0438 | 0.0518 | 0.0474 | 0.375 | 0.833 | 0.408 | 2.0 |
| queue_growth | 0.0109 | 0.5253 | 0.0448 | 0.0439 | 0.0444 | 0.275 | 0.917 | 0.248 | 2.0 |
| sudden_slowdown | 0.0102 | 0.5194 | 0.0353 | 0.042 | 0.0384 | 0.4 | 0.833 | 0.493 | 3.0 |
| autoencoder | 0.0093 | 0.5119 | 0.0317 | 0.002 | 0.0037 | 0.05 | 0.167 | 0.026 | 7.5 |
| zscore | 0.0093 | 0.5145 | 0.0208 | 0.0283 | 0.024 | 0.55 | 0.833 | 0.653 | 4.0 |
| isolation_forest | 0.0091 | 0.5123 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | None |
| ewma | 0.009 | 0.5089 | 0.0165 | 0.0225 | 0.019 | 0.5 | 0.833 | 0.681 | 4.0 |

Incident share of test windows (PR-AUC of a random score): 0.0085.

Caveat: synthetic capacity drops in a queueing model - this validates the detectors' behaviour and the evaluation machinery, not real-world detection rates. Incidents during low demand do not change traffic and are undetectable from traffic data (see 'recall (impactful)').
