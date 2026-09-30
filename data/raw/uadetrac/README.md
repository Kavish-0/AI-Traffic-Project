# UA-DETRAC (ua_detrac)

Place the files here by hand. Nothing is downloaded automatically and nothing in this folder
is committed (see `.gitignore`). Read the licence first:

    python -m traffic_intel.datasets info <dataset_id>      # source, licence, download steps
    python -m traffic_intel.datasets status <dataset_id>    # checks that the files below exist

Expected layout (relative to this folder):

```
DETRAC-Train-Annotations-XML/MVI_20011.xml     60 training sequences (used by the adapter)
DETRAC-Test-Annotations-XML/MVI_39031.xml      40 test sequences (kept apart)
DETRAC-Images/MVI_20011/img00001.jpg           frames of all 100 sequences
                                               (the original name Insight-MVT_Annotation_Train/ also works)
DETRAC-MOT-toolkit/                            official MATLAB evaluation toolkit
```

Details: `configs/datasets/manifest.yaml` and `docs/DATASETS.md`.
