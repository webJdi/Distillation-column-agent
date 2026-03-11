/**
 * WebSocket hook for real-time training progress.
 */
import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = `ws://${window.location.hostname}:8000/api/training/ws`;

export function useTrainingWebSocket() {
  const ws = useRef(null);
  const [progress, setProgress] = useState(null);
  const [connected, setConnected] = useState(false);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    ws.current = new WebSocket(WS_URL);

    ws.current.onopen = () => {
      setConnected(true);
      console.log("[WS] Connected");
    };

    ws.current.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type !== "pong") {
          setProgress(data);
        }
      } catch {
        /* ignore non-JSON */
      }
    };

    ws.current.onclose = () => {
      setConnected(false);
      console.log("[WS] Disconnected — reconnecting in 3s");
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.current.onerror = () => {
      ws.current?.close();
    };
  }, []);

  useEffect(() => {
    connect();
    // Ping every 30s to keep alive
    const ping = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 30_000);

    return () => {
      clearInterval(ping);
      clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  const requestProgress = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "get_progress" }));
    }
  }, []);

  return { progress, connected, requestProgress };
}
