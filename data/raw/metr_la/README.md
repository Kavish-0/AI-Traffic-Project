# METR-LA (metr_la)

Place the files here by hand. Nothing is downloaded automatically and nothing in this folder
is committed (see `.gitignore`). Read the licence first:

    python -m traffic_intel.datasets info <dataset_id>      # source, licence, download steps
    python -m traffic_intel.datasets status <dataset_id>    # checks that the files below exist

Expected layout (relative to this folder):

```
metr-la.h5
graph_sensor_locations.csv   (optional)
```

Details: `configs/datasets/manifest.yaml` and `docs/DATASETS.md`.
