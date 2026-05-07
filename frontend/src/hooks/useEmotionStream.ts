import { useState, useRef, useCallback, useEffect } from "react";
import { createWSUrl } from "../api";
import type { PredictResult } from "../api";

export function useEmotionStream(sessionId: number | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [connected, setConnected] = useState(false);

  const connect = useCallback(() => {
    const token = localStorage.getItem("token");
    if (!token) return;
    const ws = new WebSocket(createWSUrl(token));
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        setResult(JSON.parse(e.data));
      } catch {}
    };
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const send = useCallback(
    (imageBase64: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ image: imageBase64, session_id: sessionId }));
      }
    },
    [sessionId]
  );

  useEffect(() => () => disconnect(), [disconnect]);

  return { result, connected, connect, disconnect, send };
}
