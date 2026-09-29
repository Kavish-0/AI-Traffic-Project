# Open-Meteo weather cache (open_meteo_weather)

Place the files here by hand. Nothing is downloaded automatically and nothing in this folder
is committed (see `.gitignore`). Read the licence first:

    python -m traffic_intel.datasets info <dataset_id>      # source, licence, download steps
    python -m traffic_intel.datasets status <dataset_id>    # checks that the files below exist

Expected layout (relative to this folder):

```
openmeteo_<lat>_<lon>_<start>_<end>.parquet   (written automatically)
kaggle_szeged/weatherHistory.csv              (weather_history_szeged, placed by hand)
```

`kaggle_szeged/` is hourly weather for Szeged, Hungary, 2006-2016. It is **not** joined to METR-LA
or PEMS-BAY: those are Los Angeles 2012 and the Bay Area 2017, so it would attach the wrong weather.
The forecasting experiments use Open-Meteo for the matching place and dates.

Details: `configs/datasets/manifest.yaml` and `docs/DATASETS.md`.
