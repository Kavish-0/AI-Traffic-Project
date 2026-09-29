# data/

Nothing here is committed except folder READMEs and `.gitkeep` files (see `.gitignore`).
Datasets are placed by hand; see [docs/DATASETS.md](../docs/DATASETS.md) and
`configs/datasets/manifest.yaml`. Check what is present with
`python -m traffic_intel.datasets status`.

| Folder | Contents | Written by |
|---|---|---|
| `raw/uadetrac/` | UA-DETRAC frames + XML annotations | you (licence: research, non-commercial) |
| `raw/aicity/` | AI City CityFlowV2 Track 1 (videos, gt, calibration) | you (AI City data licence) |
| `raw/anomaly/` | AI City 2021 Track 4 anomaly videos + labels (`aic21_track4/`) | you (AI City data licence) |
| `raw/metr_la/`, `raw/pems_bay/` | DCRNN HDF5 speed matrices | you |
| `raw/weather/` | Open-Meteo archive cache | forecasting context (automatic, small) |
| `raw/public/` | small public demo videos (checksum-verified catalog) | `python -m traffic_intel.datasets download` |
| `interim/` | per-camera Parquet: detections, tracks, track summaries | camera workers |
| `processed/` | canonical time series: `<dataset>/series.parquet` + `meta.json` | `datasets prepare` |
| `annotations/` | canonical ground truth: per-sequence tracks, incident intervals | `datasets prepare` |
| `features/` | feature tables for learned models | training |
| `predictions/` | stored model outputs for evaluation | evaluation |
| `detections/` | precomputed-detector tables (ground truth or detector output) | `datasets prepare`, pipeline |
| `tracks/` | tracker outputs for evaluation | evaluation |
| `recordings/` | own camera video | you |
| `external/` | PEMS08 (scripted), legacy weather cache, `SOURCES.md` provenance log | scripts |
| `labels/` | manual labels: congestion levels, incident intervals, speed ground truth | you |
| `synthetic/` | exported synthetic scenes | `python -m traffic_intel.synthetic export` |
| `intel/` | incident-intelligence SQLite DB (dev) | incidents store |
