import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// The dashboard receives data by REST polling only. Any WebSocket construction is recorded and
// rejected, so a component that tries to open one fails its test (see app.test.tsx).
export const wsAttempts: string[] = [];
class NoWebSocket {
  constructor(url: string) {
    wsAttempts.push(url);
    throw new Error(`WebSocket connection attempted (${url}) - the dashboard must use REST polling`);
  }
}
vi.stubGlobal("WebSocket", NoWebSocket);
