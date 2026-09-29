# AI City Challenge 2021 Track 2 - vehicle re-identification (aicity21_reid)

Placed by hand; nothing in this folder is committed (see `.gitignore`). Research use only under the
AI City Challenge data licence agreement (`aic21_track2/DataLicenseAgreement_AICityChallenge_2021.pdf`).

```
aic21_track2/image_train/*.jpg      training crops
aic21_track2/image_query/*.jpg      query crops
aic21_track2/image_test/*.jpg       gallery crops
aic21_track2/train_label.xml        imageName -> vehicleID, cameraID
aic21_track2/*_track.txt            crops grouped into tracks
aic21_track2/tools/                 official evaluation tools
```

Stored for future multi-camera work. It is **not** used by experiments E1-E6 and it is **not** the
CityFlowV2 Track 1 (MTMC) data that `aicity_mtmc` needs (that goes in `data/raw/aicity/`).

    python -m traffic_intel.datasets status aicity21_reid
