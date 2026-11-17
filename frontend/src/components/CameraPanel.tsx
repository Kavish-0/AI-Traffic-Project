import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Camera } from "../api/types";
import { ago, num, titleCase } from "../lib/format";
import { ACCENT } from "../theme/tokens";
import { Badge, Card, CloseButton, Dot, StateMessage, SyntheticBadge } from "./ui";

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
 *
 * `sample`: the camera belongs to another site (the video-mode clip, location unknown) and is shown
 * as an example of the perception stage, so it is labelled as such and not tied to the selected road.
 */
export function CameraPanel({
  camera,
  frameUrl = api.cameraFrameUrl,
  sample = false,
  onClose,
}: {
  camera: Camera | null;
  frameUrl?: (cameraId: string, bust: number) => string;
  sample?: boolean;
  onClose?: () => void;
}) {
  const [tick, setTick] = useState(0);
  const [imgError, setImgError] = useState(false);
  const online = camera?.status === "online";

  useEffect(() => {
    setImgError(false);
    if (!camera || !online || !camera.preview_available) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [camera, online]);

  if (!camera) return <Card title="Camera"><StateMessage>Select a camera (junction list or map) to see its processed view.</StateMessage></Card>;
  return (
    <Card
      title={sample ? "Sample camera feed" : "Camera"}
      subtitle={sample ? `${camera.name} - location unknown, not the selected road` : camera.name}
      right={<CameraBadges synthetic={camera.synthetic} sample={sample} onClose={onClose} />}
    >
      <div className="relative aspect-video overflow-hidden rounded-lg border border-slate-800 bg-black">
        {camera.preview_available && !imgError ? (
          <img
            src={frameUrl(camera.id, tick)}
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
        {!sample && (
          <>
            <dt className="text-slate-500">Approach</dt>
            <dd className="text-slate-200">{camera.approach_id}</dd>
          </>
        )}
      </dl>
      {camera.error && <div className="mt-2"><StateMessage tone="error">{camera.error}</StateMessage></div>}
      {camera.synthetic && (
        <p className="mt-2 text-[11px] text-purple-300/80">
          Synthetic scene: simulated vehicles and simulated detections (test fixture). The overlay is drawn by the real
          tracker/feature pipeline.
        </p>
      )}
      {sample && (
        <p className="mt-2 text-[11px] text-slate-400">
          Real public highway footage through the same YOLO11m + ByteTrack pipeline (video mode). The clip's location
          is unknown, so it is an example of what a camera on this road would show, not this road's video.
        </p>
      )}
    </Card>
  );
}

function CameraBadges({ synthetic, sample, onClose }: { synthetic: boolean; sample: boolean; onClose?: () => void }) {
  return (
    <div className="flex items-center gap-2">
      {synthetic && <SyntheticBadge />}
      {sample && <Badge color={ACCENT.data} title="Camera from the video-mode site, not on this map">sample feed</Badge>}
      {onClose && <CloseButton label="Close camera feed" onClick={onClose} />}
    </div>
  );
}
