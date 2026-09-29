import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Camera } from "../api/types";
import { ago, num, titleCase } from "../lib/format";
import { Card, Dot, StateMessage, SyntheticBadge } from "./ui";

const STATUS_TEXT: Record<Camera["status"], string> = {
  online: "Online",
  offline: "Camera offline",
  error: "Camera error",
  never_started: "Not started by the pipeline",
};

/**
 * Latest processed frame of the camera (boxes, class, track id, speed - drawn by the camera worker),
 * refreshed about once per second. This is a low-rate preview, not a video stream; if no processed
 * frame exists, the panel says so rather than showing anything else.
 */
export function CameraPanel({ camera, sensorNote }: { camera: Camera | null; sensorNote?: string | null }) {
  const [tick, setTick] = useState(0);
  const [imgError, setImgError] = useState(false);
  const online = camera?.status === "online";

  useEffect(() => {
    setImgError(false);
    if (!camera || !online || !camera.preview_available) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [camera, online]);

  if (!camera && sensorNote)
    return (
      <Card title="Road sensor">
        <StateMessage tone="info">No video on this road: it is measured by a loop detector in the road surface (speed only).</StateMessage>
        <p className="mt-2 text-[11px] text-slate-400">{sensorNote}</p>
      </Card>
    );
  if (!camera) return <Card title="Camera"><StateMessage>Select a camera (junction list or map) to see its processed view.</StateMessage></Card>;
  return (
    <Card title="Camera" subtitle={camera.name} right={camera.synthetic ? <SyntheticBadge /> : null}>
      <div className="relative aspect-video overflow-hidden rounded-lg border border-slate-800 bg-black">
        {camera.preview_available && !imgError ? (
          <img
            src={api.cameraFrameUrl(camera.id, tick)}
            alt={`Processed frame of ${camera.name}`}
            className="h-full w-full object-contain"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-slate-500">Video stream unavailable</div>
        )}
        {!online && camera.preview_available && !imgError && (
          <div className="absolute inset-x-0 top-0 bg-amber-900/80 px-2 py-1 text-xs text-amber-100">
            Last frame shown - {STATUS_TEXT[camera.status].toLowerCase()}
          </div>
        )}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <dt className="text-slate-500">Status</dt>
        <dd className="flex items-center gap-1.5 text-slate-200">
          <Dot ok={online ? true : camera.status === "error" ? false : null} /> {STATUS_TEXT[camera.status]}
        </dd>
        <dt className="text-slate-500">Source</dt>
        <dd className="text-slate-200">{titleCase(camera.source_kind)}</dd>
        <dt className="text-slate-500">Last frame</dt>
        <dd className="text-slate-200">{ago(camera.last_frame_at)}</dd>
        <dt className="text-slate-500">Processing</dt>
        <dd className="text-slate-200">{num(camera.processing_fps, 1, "frames/s")}</dd>
        <dt className="text-slate-500">Detector</dt>
        <dd className="truncate text-slate-200" title={camera.device ?? ""}>
          {camera.detector ?? "—"}
        </dd>
        <dt className="text-slate-500">Approach</dt>
        <dd className="text-slate-200">{camera.approach_id}</dd>
      </dl>
      {camera.error && <div className="mt-2"><StateMessage tone="error">{camera.error}</StateMessage></div>}
      {camera.synthetic && (
        <p className="mt-2 text-[11px] text-purple-300/80">
          Synthetic scene: simulated vehicles and simulated detections (test fixture). The overlay is drawn by the real
          tracker/feature pipeline.
        </p>
      )}
    </Card>
  );
}
