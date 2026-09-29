# AI City Challenge 2021 Track 4 anomaly data (aicity21_anomaly) -> put it in aic21_track4/

Place the files here by hand. Nothing is downloaded automatically and nothing in this folder
is committed (see `.gitignore`). Read the licence first:

    python -m traffic_intel.datasets info <dataset_id>      # source, licence, download steps
    python -m traffic_intel.datasets status <dataset_id>    # checks that the files below exist

Expected layout (relative to this folder):

```
aic21_track4/aic21-track4-train-data/1.mp4 .. 100.mp4   labelled training videos
aic21_track4/aic21-track4-test-data/1.mp4 .. 150.mp4    unlabelled test videos (never paired with labels)
aic21_track4/train-anomaly-results.txt                  video_id start_s end_s
```

Details: `configs/datasets/manifest.yaml` and `docs/DATASETS.md`.
