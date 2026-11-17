import { api } from "../api/client";
import { POLL } from "../config";
import { useResource } from "../hooks/useResource";
import { CameraPanel } from "./CameraPanel";
import { Card, CloseButton, StateMessage } from "./ui";

/**
 * The video-mode camera (public highway clip through YOLO, location unknown) shown on the main dashboard
 * when the operator clicks a road that has no camera of its own. It reads the separate video-mode API
 * (port 8001 via the /video-api proxy); when that is not running the panel says how to start it.
 */
export function SampleFeedPanel({ onClose }: { onClose: () => void }) {
  const cams = useResource(api.sampleCameras, "sample-cameras", { intervalMs: POLL.statusMs });
  const real = (cams.data ?? []).filter((c) => c.source_kind !== "sensor_replay");
  const camera = real.find((c) => c.status === "online" && c.preview_available) ?? real.find((c) => c.preview_available) ?? real[0];

  if (camera) return <CameraPanel camera={camera} frameUrl={api.sampleFrameUrl} sample onClose={onClose} />;
  return (
    <Card title="Sample camera feed" right={<CloseButton label="Close camera feed" onClick={onClose} />}>
      {cams.loading ? (
        <StateMessage tone="loading">Connecting to the video-mode camera…</StateMessage>
      ) : cams.error ? (
        <StateMessage tone="warn">
          Sample camera feed unavailable - start video mode (pipeline with <code>pipeline_video.yaml</code> and its API on
          port 8001).
        </StateMessage>
      ) : (
        <StateMessage>Video mode is running but has no camera yet.</StateMessage>
      )}
    </Card>
  );
}
