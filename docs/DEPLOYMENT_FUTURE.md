# Future Deployment / Scaling

**Status: future scope - nothing in this document is implemented or required today.**

| | |
|---|---|
| Current scope | Local development + full AI system + complete UI (see README "Run everything locally") |
| Future scope | Deployment, distributed processing, cloud scaling (this document) |

The local system was built so that none of the steps below needs a restructuring. Each step
swaps an implementation behind an existing seam.

## 1. Seams that already exist

| Seam | Today (local) | Future | Change needed |
|---|---|---|---|
| Configuration | `.env` + `configs/*.yaml` (`TI_*` settings) | environment variables / secrets manager in the target platform | none (pydantic-settings reads the environment) |
| Database | SQLite file by default; PostgreSQL / PostGIS through `TI_DATABASE_URL` | managed PostgreSQL + PostGIS | set `TI_DATABASE_URL`; freeze the schema as an Alembic revision (`database/migrations/`) |
| Event bus | in-memory asyncio bus inside the runtime process | Redis Streams (`bus_backend: redis`, `TI_REDIS_URL`) - ARCHITECTURE.md Phase 9 | implement `RedisStreamsEventBus` behind `EventBus`; the WebSocket gateway reads the same bus |
| Runtime → API → browser | database; the dashboard **polls the REST API** (live values every 3 s, per-window data on change) | camera workers → Redis Streams → backend → WebSocket → React | enable the existing optional `/ws/traffic` channel (`TI_WS_ENABLED=true`), switch LiveHub from database polling to a bus subscription, and add a WebSocket client to the dashboard in place of the live poll |
| Camera workers | threads in one runtime process | one process/container per camera (or per GPU) | `CameraWorker` already publishes only aggregates through `Publisher`; use a bus publisher per process |
| Detector | Ultralytics YOLO in each worker | shared GPU inference server (batched `detect(Sequence[Frame])`, e.g. Triton) | new `Detector` implementation (registry), selected in `pipeline.yaml` |
| Frame-level data | Parquet under `data/interim`, preview JPEGs under `runs/` | object storage (S3-compatible) | point the Parquet recorder / preview writer at the object store |
| Models | `models/<task>/<name>/<version>` on disk | object storage / model registry service | `ModelStore` interface (`FileModelStore` today) |
| Logging | text; `TI_LOG_FORMAT=json` for one JSON object per line | log collector (Loki / ELK / cloud logging) | none |
| Frontend | Vite dev server with an `/api` proxy | static build (`npm run build` → `frontend/dist`) behind a reverse proxy | reverse proxy routes `/api` (and `/ws` once push is enabled) to the API; existing `frontend/Dockerfile` + `nginx.conf` |

## 2. Possible target topology

```
cameras (RTSP) ──► camera workers (1 per camera, GPU nodes) ──► Redis Streams ──► analytics worker(s)
                         │ Parquet / previews                         │                 │
                         ▼                                            ▼                 ▼
                   object storage                            API replicas (no torch) ◄── PostgreSQL + PostGIS
                                                                   ▲   (REST; WS later)      (managed)
                                                  load balancer ───┘
                                                        ▲
                                              static frontend (CDN / nginx)
```

## 3. Scaling notes

- **Cameras.** CPU/GPU cost scales with cameras × processed fps (`process_fps`). A shared
  batched detector amortises the GPU.
- **Analytics.** Per-window work is milliseconds per approach, so one analytics worker handles
  hundreds of approaches. Partition by junction if needed; state per approach is small and
  rebuilt from the database at start.
- **API.** Today it is stateless (the dashboard polls REST), so replicas can run behind a load
  balancer as they are. If the WebSocket push channel is enabled later, each replica holds its
  clients' connections and subscribes to the Redis bus.
- **Polling vs push.** REST polling costs a few small requests per browser every few seconds.
  That is fine for a handful of operator screens. Push (Redis Streams → WebSocket) becomes
  worthwhile with many concurrent dashboards or sub-second update needs.
- **Database.** Time-series tables have composite keys on (approach, time), ready for monthly
  partitioning or TimescaleDB. Retention policies apply to `traffic_observation`, `forecast`
  and `risk_assessment`.
- **Docker.** `docker-compose.yml` (PostGIS database, API, frontend, optional Redis profile)
  remains usable but is optional. A CUDA image for camera workers is future work.

## 4. Before any real deployment (non-technical prerequisites)

- Privacy review of camera footage (faces and plates). Keep only aggregates long term; blur any
  published imagery.
- Calibrate the incident score on labelled real incidents, so it becomes a probability. Until
  then the UI must keep the "uncalibrated" label.
- Validate speed and count accuracy on the target cameras (labelled clips).
- Authentication and authorisation for the API and dashboard. There are none in the local
  scope.
- TLS, domain, monitoring and alerting, backups.
