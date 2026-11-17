/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies the REST API to FastAPI so the browser only sees one origin. The dashboard polls
// REST (no WebSocket), so there is no /ws proxy.
// The backend address is configurable (VITE_BACKEND_URL, default http://localhost:8000).
const backend = process.env.VITE_BACKEND_URL ?? "http://localhost:8000";
// The video-mode API (pipeline_video.yaml, port 8001): the main dashboard shows its camera as a
// "Sample camera feed" when a road is clicked. Optional - the panel says so when it is not running.
const videoBackend = process.env.VITE_VIDEO_BACKEND_URL ?? "http://localhost:8001";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": backend,
      "/video-api": { target: videoBackend, rewrite: (path) => path.replace(/^\/video-api/, "") },
    },
  },
  build: {
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: { manualChunks: { maplibre: ["maplibre-gl"], echarts: ["echarts/core", "echarts/charts", "echarts/components", "echarts/renderers"] } },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
