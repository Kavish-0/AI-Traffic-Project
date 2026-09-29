# LISA Traffic Light Dataset (lisa_traffic_lights) -> lisa/

Placed by hand; nothing in this folder is committed (see `.gitignore`). Non-commercial research use
(licence as stated by the provider / Kaggle mirror - verify before any use beyond this project).

```
lisa/Annotations/<clip>/frameAnnotationsBOX.csv, frameAnnotationsBULB.csv
lisa/dayTrain/dayClipN/frames/*.jpg,  lisa/nightTrain/nightClipN/frames/*.jpg
lisa/daySequence1|2/, lisa/nightSequence1|2/, lisa/sample-dayClip6/, lisa/sample-nightClip1/
```

Stored for a possible signal-phase module. Not used by experiments E1-E6; the current detector has
no traffic-light class mapped.

    python -m traffic_intel.datasets status lisa_traffic_lights
