# AI City Challenge CityFlowV2, Track 1 (aicity_mtmc)

Place the files here by hand. Nothing is downloaded automatically and nothing in this folder
is committed (see `.gitignore`). Read the licence first:

    python -m traffic_intel.datasets info <dataset_id>      # source, licence, download steps
    python -m traffic_intel.datasets status <dataset_id>    # checks that the files below exist

Expected layout (relative to this folder):

```
train/S01/c001/vdo.avi
train/S01/c001/gt/gt.txt
train/S01/c001/calibration.txt
cam_timestamp/S01.txt
```

Details: `configs/datasets/manifest.yaml` and `docs/DATASETS.md`.
