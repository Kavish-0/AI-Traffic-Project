"""Cameras: configuration + live status, and the latest annotated frame.

The frame endpoint serves the camera worker's low-rate preview JPEG (boxes, class, track id, speed,
zones) - one image, refreshed about once per second by the pipeline runtime. It is not a video
stream, and when no preview exists the API says so (404) instead of inventing one.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from traffic_intel.api.deps import get_ctx
from traffic_intel.api.routes.stubs import not_implemented
from traffic_intel.api.schemas import CameraOut
from traffic_intel.core.config import CalibrationConfig, ZoneConfig
from traffic_intel.services import network as net
from traffic_intel.services.context import ServiceContext

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("")
def list_cameras(ctx: ServiceContext = Depends(get_ctx)) -> list[CameraOut]:
    return net.camera_views(ctx)


@router.get("/{camera_id}")
def camera_detail(camera_id: str, ctx: ServiceContext = Depends(get_ctx)) -> CameraOut:
    cam = net.camera_view(ctx, camera_id)
    if cam is None:
        raise HTTPException(404, f"camera '{camera_id}' unknown")
    return cam


@router.get("/{camera_id}/status")
def camera_status(camera_id: str, ctx: ServiceContext = Depends(get_ctx)) -> dict[str, Any]:
    """Raw heartbeat: fps, frames, per-stage latency, last frame time, errors."""
    st = ctx.store.camera_statuses().get(camera_id)
    if st is None:
        raise HTTPException(404, f"no status for camera '{camera_id}' (never started by the pipeline runtime)")
    return st


@router.get("/{camera_id}/frame.jpg", response_class=FileResponse)
def camera_frame(camera_id: str, ctx: ServiceContext = Depends(get_ctx)) -> FileResponse:
    found = net.preview_file(ctx, camera_id)
    if found is None:
        raise HTTPException(404, "Video stream unavailable - no processed frame for this camera")
    path, ts = found
    headers = {"Cache-Control": "no-store"}
    if ts is not None:
        headers["X-Frame-Time"] = ts.isoformat()
    return FileResponse(path, media_type="image/jpeg", headers=headers)


@router.put("/{camera_id}/calibration")
async def set_calibration(camera_id: str, calibration: CalibrationConfig) -> dict[str, Any]:
    """Calibration is edited in configs/cameras/<id>.yaml (and checked with the calibration tool)."""
    raise not_implemented("future: calibration editing via API")


@router.put("/{camera_id}/zones")
async def set_zones(camera_id: str, zones: list[ZoneConfig]) -> list[ZoneConfig]:
    raise not_implemented("future: zone editing via API")
