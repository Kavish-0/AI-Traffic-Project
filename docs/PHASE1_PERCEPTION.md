# Phase 1: perception on GPU with public footage

This document covers the per-camera worker running on real (public) video with
CUDA inference. Metric definitions, assumptions and the calibration guide are in
[TRAFFIC_STATE_ESTIMATION.md](TRAFFIC_STATE_ESTIMATION.md).

## 1. Setup (GPU)

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev,cv]"
# PyPI's default torch wheel for Windows is CPU-only. Install the CUDA build that matches
# the torch version pip resolved (here 2.14.0, CUDA 13.0 wheels):
pip install --force-reinstall --no-deps torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Device selection is configured in `configs/pipeline.yaml -> detector.params`:

```yaml
device: auto            # auto = CUDA if available, else MPS, else CPU | cuda | cuda:N | mps | cpu
fallback_to_cpu: true   # requested GPU missing -> CPU with a WARNING; false -> startup error
half: auto              # fp16 on CUDA only
```

The device actually used is logged when the detector starts, and appears in
`summary.json` and the replay output:

```
Detector: YOLO yolo11m.pt | device cuda:0 [NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 13.0, torch 2.14.0+cu130, fp16] | imgsz 1280 | classes {...}
```

## 2. Public development footage

`configs/datasets/public_videos.yaml` is a catalog of URLs, target files,
licences and pinned SHA-256 hashes. Downloads are verified against the hashes
and recorded in `data/external/SOURCES.md`.

```bash
python -m traffic_intel.datasets list
python -m traffic_intel.datasets download roboflow_vehicles      # or --all
```

| Name | Content | Licence |
|---|---|---|
| `roboflow_vehicles` | 3840×2160, 25 fps, 21.5 s. Motorway from an overpass, both directions. **Used by `public_highway_01`.** | Not stated by the publisher. Local development only; do not redistribute frames or results. |
| `intel_car_detection` | 768×432, 12.5 fps, 30 s | CC BY 4.0 |
| `intel_person_bicycle_car` | 768×432, 12 fps, 54 s. Pedestrians, bicycles and cars. | CC BY 4.0 |

The pipeline never refers to these files directly. A video is used only
because a camera YAML's `source:` points at it.

## 3. Camera `public_highway_01` (`configs/cameras/public_highway.yaml`)

- **Approach:** the left carriageway, whose traffic comes towards the camera.
  It is mapped to `dev_highway_approach`, a development site added to
  `configs/network/demo_network.yaml` with placeholder coordinates.
- **Processing:** all 25 fps, analysed at 1920×1080. The 4K source is resized
  after decoding, and zones and calibration rescale automatically.
- **Calibration is approximate**, derived from the road markings because the
  site was not surveyed:
  1. A vehicle-free background (the median of 14 frames) was built. The white
     markings were located along image rows: the left edge line, the lane
     divider and the median-side line. All three are straight in the image,
     consistent with a planar road.
  2. Lateral scale: 2 lanes × 3.75 m (EU motorway standard), so the three
     lines are at x = 0, 3.75 and 7.5 m.
  3. Longitudinal: the divider's dash starts (image rows 994 … 1870) follow a
     1D projective progression to 0.22 px RMS. Four consecutive dashes give
     cross-ratios of 1.33, 1.34 and 1.33 against 4/3 expected for equal
     spacing. The dash-to-gap ratio is 1:1.92.
  4. Absolute scale: the dash period cannot be recovered from the image alone.
     (Three parallel lines plus a periodic pattern leave one degree of freedom
     even with a pinhole prior.) With a 9 m period, the only heavy truck
     measured 45 km/h. EU trucks are speed-limited to 90 km/h, so 9 m is
     wrong. **P = 18 m** (the common 6 m dash + 12 m gap pattern, matching
     1:1.92) gives the truck about 91 km/h and a carriageway mean of 110 km/h.
  5. Result: 12 correspondences with 0.05 m reprojection error. **Speeds are
     indicative (roughly ±10–15%)**, because the absolute scale rests on
     marking standards and one truck, not a survey.

## 4. Running the replay

```bash
cd backend
python -m traffic_intel.pipeline replay public_highway_01 --annotate --live
```

| Option | Effect |
|---|---|
| `--source <file / webcam:0 / rtsp://…>` | use another video with this camera's zones |
| `--device cpu` | override the device from the config |
| `--max-seconds N` | stop after N seconds of video |
| `--no-record` | skip Parquet |
| `--detections file.csv` | replay precomputed detections; no detector or GPU needed |

Outputs (`run_id` = start time, for example `20260923-124942`):

| What | Where |
|---|---|
| Annotated video | `runs/public_highway_01/<run_id>/annotated.mp4` |
| Window states / live states / trajectories | `runs/public_highway_01/<run_id>/traffic_states.jsonl`, `live_states.jsonl`, `tracks.jsonl` |
| Run summary (device, model, latency, published counts, files) | `runs/public_highway_01/<run_id>/summary.json` |
| Parquet | `data/interim/public_highway_01/<video date>/<run_id>/detections-00000.parquet`, `tracks-00000.parquet`, `track_summaries-00000.parquet` |

The directories are set by `outputs.runs_dir` and `outputs.parquet_dir` in
`configs/pipeline.yaml`. The Parquet date folder comes from the video
timestamps (`replay_start`), not the wall clock.

Parquet schemas:

- `detections`: one row per raw detection. Columns: camera_id, run_id,
  frame_index, timestamp, class_id, class_name, confidence, x1, y1, x2, y2.
- `tracks`: one row per tracked object per frame. Columns: track_id,
  class_name (majority vote), confidence, box, anchor_x/y, ground_x_m/y_m,
  speed_kmh, stopped_s, age_s.
- `track_summaries`: one row per finished trajectory. It includes the pixel
  and ground polylines, mean speed, maximum stop time and line crossings.

## 5. Per-camera worker and what it publishes

`pipeline/camera_worker.py`: a `CameraWorker` owns one camera's source,
detector, tracker, trajectory store and feature engine. It runs blocking (the
replay CLI) or in a thread (`start()`/`stop()`, bridged to the asyncio event
bus by `BusPublisher`).

`publish.WorkerPublisher` **only** allows these topics and raises on anything
else:

| Topic | Payload | Rate |
|---|---|---|
| `traffic.features.live` | `TrafficState` (rolling 5 s, `is_live=true`) | every `live_tick_s` (1 s) |
| `traffic.features.window` | `TrafficFeatures` | per 60 s window |
| `traffic.state` | `TrafficState` | per 60 s window |

Detections, tracks and trajectories never leave the worker. They go only to
Parquet, JSONL and the annotated video. The API process does not import torch
or ultralytics; a test checks this.

## 6. Replacing the footage with your own camera

1. Copy `configs/cameras/public_highway.yaml` to, for example,
   `configs/cameras/junction_a_north.yaml`.
2. Change `id`, `approach_id` (an approach in the network config), `source`
   (file path, `webcam:0` or `rtsp://…`) and `replay_start`, plus `resolution`
   and `process_fps` if needed.
3. Run the camera once **without** a `calibration:` section. This is
   perception-only mode: detections, tracks, annotated video and Parquet are
   produced, with no metric values. Use it to check detection quality.
4. Pick calibration points and zones on a frame
   (`python -m traffic_intel.computer_vision.calibration_tool <id>`; see
   TRAFFIC_STATE_ESTIMATION.md §8). Add them, then check with the same tool.
5. Add the file to `camera_files` in `configs/pipeline.yaml` and run
   `replay <id>`.

No code changes are needed. Detector, tracker, model file, device and class
mapping all come from `configs/pipeline.yaml`.

## 7. Model and performance (RTX 4060 Laptop, this clip)

Configured model: **`yolo11m.pt`** (COCO-pretrained, Ultralytics). It runs at
`imgsz` 1280, fp16 on CUDA. Class mapping by label: car, motorcycle, bus,
truck, bicycle, and person → pedestrian.

| Measurement | ms / frame |
|---|---|
| yolo11s @1280 (detection only, 1080p input) | 32.5 |
| yolo11m @1280 | 34.0 |
| yolo11m @960 | 35.5 |
| yolo11l @1280 | 46.3 |
| 4K decode + resize (CPU) | 30.2 |

Detection time barely changes between s and m, or between 960 and 1280 px:
the limit is CPU-side pre/post-processing, not the GPU. So the larger m model
costs almost nothing extra.

End to end, the replay runs at **12–14 frames/s** for this 4K source, with
decoding overlapped with inference through a read-ahead thread. File replays
never drop frames. Live sources at 1080p avoid most of the decode cost.

## 8. Limitations

- **Approximate calibration.** Speeds, density and ROI length rest on assumed
  lane width and dash period, sanity-checked with one truck. They are
  indicative, not measured.
- **Short clip.** 21.5 s gives only one partial 60 s window. The
  window/classification path is exercised, but meaningful traffic statistics
  need long recordings.
- **Unlabelled footage.** No ground truth exists for this clip, so detection
  recall and tracking accuracy on it are unmeasured. Accuracy figures come from
  the synthetic dataset (TRAFFIC_STATE_ESTIMATION.md §9).
- **Misdetections happen.** For example, a short "pedestrian" track appeared on
  the motorway. The continuous-presence rule kept it out of the counts, but it
  is still in the Parquet `tracks` data.
- **The ROI mask hides far vehicles** until they enter the analysis area, by
  design.
- **One carriageway per camera config.** The opposite carriageway would need a
  second camera YAML on the same video, with its own approach and zones.
- **Throughput** is bounded by CPU-side work: 4K decoding and Ultralytics
  pre/post-processing. Batching several cameras through one detector is Phase 9
  work.
