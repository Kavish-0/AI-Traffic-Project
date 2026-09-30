# Own camera recordings (own_recordings)

Place the files here by hand. Nothing is downloaded automatically and nothing in this folder
is committed (see `.gitignore`). Read the licence first:

    python -m traffic_intel.datasets info <dataset_id>      # source, licence, download steps
    python -m traffic_intel.datasets status <dataset_id>    # checks that the files below exist

Expected layout (relative to this folder):

```
<camera>/<YYYY-MM-DD_HHMM>.mp4
```

Details: `configs/datasets/manifest.yaml` and `docs/DATASETS.md`.
