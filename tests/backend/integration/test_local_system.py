"""Complete local flow on a SYNTHETIC fixture (test data only - never presented as real results):

    runtime: synthetic camera -> tracker -> features -> 60 s window -> analytics (anomaly, fusion, risk)
             -> database (windows, live values, risk, camera heartbeat, preview frame)
    API:     reads the same database -> REST responses the dashboard renders
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from traffic_intel.api.main import create_app
from traffic_intel.computer_vision.sources.live_replay import LiveReplaySource
from traffic_intel.core.types import Frame
from traffic_intel.database.store import TrafficStore
from traffic_intel.pipeline.runtime import (
    CameraRef,
    LiveReplay,
    PreviewCfg,
    Profile,
    Runtime,
    RuntimeConfig,
    WeatherCfg,
)
from traffic_intel.services.context import ServiceContext


def test_runtime_to_api_end_to_end(tmp_path: Path) -> None:
    store = TrafficStore(f"sqlite:///{(tmp_path / 'sys.db').as_posix()}")
    cfg = RuntimeConfig(
        default_profile="t",
        profiles={"t": Profile(cameras=[CameraRef(synthetic="default", camera_id="it_cam",
                                                  approach_id="junction_a_east", name="IT synthetic",
                                                  duration_s=70, seed=3)])},
        live_replay=LiveReplay(realtime=False, loop=False, wall_clock=True),   # not paced: fast test
        forecast=False, weather=WeatherCfg(enabled=False),
        preview=PreviewCfg(enabled=True, interval_s=5.0, dir=str(tmp_path / "preview")),
    )
    rt = Runtime("t", cfg=cfg, store=store, max_seconds=65)
    started = datetime.now(UTC)
    rt.run()

    windows = store.traffic_history("junction_a_east")
    assert windows and windows[0]["window_s"] >= 59 and windows[0]["vehicle_count"] > 0
    assert windows[0]["traffic_state"] in ("free", "moderate", "high", "severe")
    assert windows[0]["window_start"] >= started - timedelta(seconds=61)          # wall-clock (minute-aligned)
    assert store.latest_risk("junction_a_east")                                  # analytics ran
    st = store.camera_statuses()["it_cam"]
    assert st["state"] == "stopped" and st["frames"] > 600 and st["source_kind"] == "synthetic"
    assert st["preview_path"] and Path(st["preview_path"]).exists()
    assert store.latest_live_states()                                           # 1 s live values persisted

    app = create_app()
    app.state.ctx_override = ServiceContext.build(store)
    with TestClient(app) as c:
        snap = c.get("/api/v1/approaches/junction_a_east/snapshot").json()
        assert snap["window"]["vehicle_count"] == windows[-1]["vehicle_count"]
        assert snap["synthetic"] is True and snap["cameras"][0]["id"] == "it_cam"
        assert snap["forecast"]["status"] in ("insufficient_history", "stale", "ok")
        img = c.get("/api/v1/cameras/it_cam/frame.jpg")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and img.content[:2] == b"\xff\xd8"
        assert "X-Frame-Time" in img.headers


class _Src:
    camera_id = "s"
    fps = 10.0
    is_live = False
    frame_size = (4, 4)

    def frames(self):  # noqa: ANN201
        t0 = datetime(2020, 1, 1, tzinfo=UTC)
        for k in range(3):
            yield Frame("s", k, t0 + timedelta(seconds=k / 10), np.zeros((4, 4, 3), np.uint8))

    def close(self) -> None:
        pass


def test_live_replay_loops_with_wall_clock_and_keeps_indices() -> None:
    start = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    src = LiveReplaySource(_Src(), loop=True, realtime=False, start=start)  # type: ignore[arg-type]
    out = []
    for f in src.frames():
        out.append(f)
        if len(out) == 7:
            src.close()
    assert [f.index for f in out[:6]] == [0, 1, 2, 0, 1, 2]                  # source indices preserved
    ts = [(f.timestamp - start).total_seconds() for f in out[:6]]
    assert ts == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] or np.allclose(ts, [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert src.is_live and src.loops >= 2
