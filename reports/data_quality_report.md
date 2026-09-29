# Data quality report

Generated 2026-09-24T20:21:58+00:00 by `python -m traffic_intel.datasets quality`.
Every number below was measured on the files on disk.

| Dataset | Kind | ok | warn | fail | info |
|---|---|--:|--:|--:|--:|
| ua_detrac | object annotations (detection / tracking) | 5 | 2 | 0 | 4 |
| aicity21_anomaly | incident intervals (video level) | 3 | 2 | 0 | 1 |
| metr_la | time series | 6 | 1 | 0 | 2 |
| pems_bay | time series | 7 | 0 | 0 | 2 |
| pems08 | time series | 10 | 0 | 0 | 1 |
| aicity21_reid | identity labels (re-identification) | 1 | 0 | 0 | 2 |
| lisa_traffic_lights | box annotations (CSV) | 2 | 2 | 0 | 0 |
| weather_history_szeged | context table (weather) | 1 | 5 | 0 | 0 |
| external/weather (Open-Meteo cache) | context cache (Parquet) | 0 | 2 | 0 | 0 |

## ua_detrac (object annotations (detection / tracking))

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\raw\uadetrac`

| Check | Status | Result |
|---|---|---|
| train: images vs annotations | **ok** | 82085 annotated frames, 83791 images; 1706 images have no annotation entry (excluded - never assumed empty); 0 annotations without an image |
| train: box validity | **ok** | 0 zero/negative-size, 0 outside the image, 0 duplicate ids in a frame, 625 smaller than 8 px |
| train: ignored regions | **info** | 24940 annotated boxes centred inside ignored regions (ignored regions are masked for training and excluded from evaluation) |
| train: class balance | **warn** | car 560904 (93.8%), bus 33651 (5.6%), truck 3726 (0.6%) |
| train: adjacent-frame redundancy | **info** | median IoU of the same object in consecutive frames 0.85 - adjacent frames are near-duplicates, so splits are by sequence and training frames are subsampled |
| test: images vs annotations | **ok** | 56167 annotated frames, 56340 images; 173 images have no annotation entry (excluded - never assumed empty); 0 annotations without an image |
| test: box validity | **ok** | 0 zero/negative-size, 0 outside the image, 0 duplicate ids in a frame, 4 smaller than 8 px |
| test: ignored regions | **info** | 47780 annotated boxes centred inside ignored regions (ignored regions are masked for training and excluded from evaluation) |
| test: class balance | **warn** | car 587074 (86.9%), bus 71785 (10.6%), truck 16915 (2.5%) |
| test: adjacent-frame redundancy | **info** | median IoU of the same object in consecutive frames 0.94 - adjacent frames are near-duplicates, so splits are by sequence and training frames are subsampled |
| train/test sequence overlap | **ok** | 60 train and 40 test sequences; overlap: none |

## aicity21_anomaly (incident intervals (video level))

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\raw\anomaly\aic21_track4`

| Check | Status | Result |
|---|---|---|
| labels reference existing training videos | **ok** | 17 labelled videos; without a file: none |
| interval validity | **ok** | 18 intervals; 0 with start >= end or end beyond the video ([]) |
| overlapping intervals | **ok** | 0 overlapping pairs; 1 videos have more than one interval |
| class balance | **warn** | 17/100 training videos contain an anomaly; anomalous time 2.7 h of 24.5 h (10.9%); 5 intervals start at 0 s (anomaly already present when the video starts) |
| frame rates | **warn** | 2 training videos not at 30 fps: {'58': 14.3, '66': 23.56} - labels are in seconds, so times are converted with each video's own fps |
| test split | **info** | 150 test videos without public labels - not usable for supervised evaluation; excluded |

## metr_la (time series)

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\raw\metr_la`

| Check | Status | Result |
|---|---|---|
| duplicate (series, timestamp) | **ok** | 0 duplicates |
| regular sampling | **ok** | 0 steps differ from 300 s (largest 0 days 00:05:00); DST: 12 non-existent local timestamps dropped |
| missing avg_speed_kmh | **warn** | 8.08% missing overall; worst series 20.1%, 1 series > 20% missing |
| zeros = sensor outage | **info** | 572818 zero readings converted to missing (documented source convention) |
| longest gap (avg_speed_kmh) | **info** | longest run of missing steps per series: max 1683, median 381 (gaps > max_gap_fill_steps are never interpolated; windows touching them are excluded) |
| physical range avg_speed_kmh | **ok** | 0 values outside [0, 200] |
| stuck sensor (avg_speed_kmh) | **ok** | 0.05% of readings sit in runs of identical non-missing values lasting >= 1 h |
| chronological split (forecasting_metr_la_context.yaml) | **ok** | train < val < test, disjoint periods; targets never cross a boundary (windows.py) |
| chronological split (forecasting_metr_la_traffic_only.yaml) | **ok** | train < val < test, disjoint periods; targets never cross a boundary (windows.py) |

## pems_bay (time series)

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\raw\pems_bay`

| Check | Status | Result |
|---|---|---|
| duplicate (series, timestamp) | **ok** | 0 duplicates |
| regular sampling | **ok** | 0 steps differ from 300 s (largest 0 days 00:05:00); DST: 0 non-existent local timestamps dropped |
| missing avg_speed_kmh | **ok** | 0.00% missing overall; worst series 0.0%, 0 series > 20% missing |
| zeros = sensor outage | **info** | 521 zero readings converted to missing (documented source convention) |
| longest gap (avg_speed_kmh) | **info** | longest run of missing steps per series: max 1, median 1 (gaps > max_gap_fill_steps are never interpolated; windows touching them are excluded) |
| physical range avg_speed_kmh | **ok** | 0 values outside [0, 200] |
| stuck sensor (avg_speed_kmh) | **ok** | 0.70% of readings sit in runs of identical non-missing values lasting >= 1 h |
| chronological split (forecasting_pems_bay_context.yaml) | **ok** | train < val < test, disjoint periods; targets never cross a boundary (windows.py) |
| chronological split (forecasting_pems_bay_traffic_only.yaml) | **ok** | train < val < test, disjoint periods; targets never cross a boundary (windows.py) |

## pems08 (time series)

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\external\pems08`

| Check | Status | Result |
|---|---|---|
| duplicate (series, timestamp) | **ok** | 0 duplicates |
| regular sampling | **ok** | 0 steps differ from 300 s (largest 0 days 00:05:00); DST: 0 non-existent local timestamps dropped |
| missing flow_veh_h | **ok** | 0.27% missing overall; worst series 3.4%, 0 series > 20% missing |
| missing occupancy_pct | **ok** | 0.27% missing overall; worst series 3.4%, 0 series > 20% missing |
| missing avg_speed_kmh | **ok** | 0.27% missing overall; worst series 3.4%, 0 series > 20% missing |
| longest gap (avg_speed_kmh) | **info** | longest run of missing steps per series: max 29, median 0 (gaps > max_gap_fill_steps are never interpolated; windows touching them are excluded) |
| physical range avg_speed_kmh | **ok** | 0 values outside [0, 200] |
| physical range occupancy_pct | **ok** | 0 values outside [0, 100] |
| physical range flow_veh_h | **ok** | 0 values outside [0, 20000] |
| stuck sensor (avg_speed_kmh) | **ok** | 0.03% of readings sit in runs of identical non-missing values lasting >= 1 h |
| chronological split (forecasting_pems08.yaml) | **ok** | train < val < test, disjoint periods; targets never cross a boundary (windows.py) |

## aicity21_reid (identity labels (re-identification))

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\raw\aicity_reid\aic21_track2`

| Check | Status | Result |
|---|---|---|
| label file vs images | **ok** | 52717 labelled, 52717 listed, 52717 on disk |
| images per identity | **info** | 440 identities; images per identity min 19, median 92, max 682; cameras per identity median 4 |
| test identities | **info** | test/query labels carry camera ids only - identities are withheld (challenge evaluation), so a held-out evaluation must split the TRAINING identities |

## lisa_traffic_lights (box annotations (CSV))

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\raw\traffic_lights\lisa`

| Check | Status | Result |
|---|---|---|
| box validity | **ok** | 112370 boxes in 24 files; 0 zero/negative-size, 0 outside the 1280x960 frame |
| tag balance | **warn** | go 48892, stop 44730, stopLeft 13048, warning 2669, goLeft 2476, warningLeft 350, goForward 205 |
| referenced frames exist | **ok** | 100.0% of annotation rows point to an image on disk (36265 distinct frames referenced, 43016 images) |
| duplicate rows | **warn** | 2911 duplicate annotation rows |

## weather_history_szeged (context table (weather))

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\raw\weather\kaggle_szeged`

| Check | Status | Result |
|---|---|---|
| timestamps | **ok** | 0 unparseable; 2 UTC offsets present (['+0100', '+0200']); range 2005-12-31 23:00:00+00:00 .. 2016-12-31 22:00:00+00:00 |
| duplicates | **warn** | 24 duplicate rows, 24 duplicate timestamps |
| impossible values | **warn** | Humidity: 22, Pressure (millibars): 1288 (pressure outside 850-1090 mbar, humidity 0) |
| constant columns | **warn** | ['Loud Cover'] carry no information |
| missing values | **warn** | {'Precip Type': 517} |
| overlap with traffic datasets | **warn** | time overlap: metr_la: yes, pems_bay: no, pems08: yes - and a single site; weather must match the traffic's location AND time to be a feature |

## external/weather (Open-Meteo cache) (context cache (Parquet))

`C:\Users\Kavish Gupta\AI-Project-Traffic\data\external\weather`

| Check | Status | Result |
|---|---|---|
| openmeteo_12.970_77.590_2025-06-01_2025-07-13.parquet | **warn** | 1032 rows 2025-06-01 00:00:00+00:00 .. 2025-07-13 23:00:00+00:00; missing: {'visibility_km': '100%'} |
| openmeteo_34.060_-117.340_2016-07-01_2016-09-01.parquet | **warn** | 1512 rows 2016-07-01 00:00:00+00:00 .. 2016-09-01 23:00:00+00:00; missing: {'visibility_km': '100%'} |
