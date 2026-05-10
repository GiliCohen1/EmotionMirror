import { useState, useRef, useCallback, useEffect } from "react";
import { createWSUrl } from "../api";
import type { PredictResult } from "../api";

/**
 * Manages a WebSocket connection to /api/ws/stream.
 *
 * connect() returns a Promise that resolves when the socket is open — always
 * await it before calling send() so the first frame is never lost.
 *
 * send(image, sessionId) passes sessionId at call-time so there are no
 * stale-closure issues between session creation and frame dispatch.
 */
export function useEmotionStream(onResult?: (result: PredictResult) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  const connect = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      const token = localStorage.getItem("token");
      if (!token) { reject(new Error("No auth token")); return; }

      // Already open — nothing to do
      if (wsRef.current?.readyState === WebSocket.OPEN) { resolve(); return; }

      // Close any stale socket first
      wsRef.current?.close();

      const ws = new WebSocket(createWSUrl(token));
      wsRef.current = ws;

      ws.onopen = () => { setConnected(true); resolve(); };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
      };

      ws.onerror = () => {
        ws.close();
        reject(new Error("WebSocket connection failed"));
      };

      ws.onmessage = (e) => {
        try { onResultRef.current?.(JSON.parse(e.data) as PredictResult); } catch {}
      };
    });
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  // sessionId passed at call-time — avoids stale closure issues
  const send = useCallback((imageBase64: string, sessionId: number | null) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ image: imageBase64, session_id: sessionId }));
    }
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return { connected, connect, disconnect, send };
}
