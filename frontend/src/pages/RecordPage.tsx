import { useState, useRef, useCallback } from "react";
import { EmotionDisplay } from "../components/EmotionDisplay";
import { EmotionTimeline } from "../components/EmotionTimeline";
import { Navbar } from "../components/Navbar";
import { sessions, journal, predict } from "../api";
import type { Reading, PredictResult } from "../api";
import { useWebcam } from "../hooks/useWebcam";
import { useEmotionStream } from "../hooks/useEmotionStream";
import type { Theme } from "../hooks/useTheme";

type Frame = { data: string; capturedAt: number };
type Mode  = "batch" | "live";

interface Props {
  onLogout: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

export function RecordPage({ onLogout, theme, onToggleTheme }: Props) {
  const [mode, setMode]           = useState<Mode>("batch");
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [readings, setReadings]   = useState<Reading[]>([]);
  const [lastResult, setLastResult] = useState<PredictResult | null>(null);
  const [journalPrompt, setJournalPrompt] = useState<string | null>(null);
  const [running, setRunning]     = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress]   = useState<{ done: number; total: number } | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [sessionNote, setSessionNote] = useState<string | null>(null);

  const { videoRef, start: startCam, stop: stopCam, captureBase64 } = useWebcam();
  const intervalRef    = useRef<number | null>(null);
  const framesRef      = useRef<Frame[]>([]);
  const sessionIdRef   = useRef<number | null>(null);
  const readingsRef    = useRef<Reading[]>([]); // mirror for live-mode accumulation

  // ── Live stream callback ──────────────────────────────────────────────
  const handleStreamResult = useCallback((result: PredictResult) => {
    setLastResult(result);
    if (result.face_found && result.emotion) {
      const reading: Reading = {
        id: readingsRef.current.length,
        emotion: result.emotion,
        confidence: result.confidence,
        probabilities: result.probabilities,
        timestamp: new Date().toISOString(),
      };
      readingsRef.current = [...readingsRef.current, reading];
      setReadings(readingsRef.current);
    }
  }, []);

  const { connect, disconnect, send } = useEmotionStream(handleStreamResult);

  // ── Start session ──────────────────────────────────────────────────────
  const startSession = async () => {
    const res = await sessions.start();
    const sid = res.data.id;
    setSessionId(sid);
    sessionIdRef.current = sid;
    setReadings([]);
    readingsRef.current = [];
    setLastResult(null);
    setJournalPrompt(null);
    setFrameCount(0);
    setSessionNote(null);
    framesRef.current = [];

    await startCam();

    if (mode === "live") {
      try {
        await connect(); // wait until socket is OPEN before sending any frames
      } catch {
        setSessionNote("Could not connect to live stream. Check that the backend is running.");
        stopCam();
        await sessions.end(sid);
        return;
      }
    }

    setRunning(true);

    intervalRef.current = window.setInterval(() => {
      const data = captureBase64();
      if (!data) return;
      setFrameCount(n => n + 1);
      if (mode === "live") {
        send(data, sessionIdRef.current);
      } else {
        framesRef.current.push({ data, capturedAt: Date.now() });
      }
    }, 2000);
  };

  // ── Stop session ───────────────────────────────────────────────────────
  const stopSession = async () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    stopCam();
    setRunning(false);

    const sid = sessionIdRef.current;
    if (!sid) return;

    // ── Live mode: readings already saved by backend WebSocket ──
    if (mode === "live") {
      disconnect();
      await sessions.end(sid);
      if (readingsRef.current.length === 0) {
        setSessionNote("No emotions detected live. Make sure your face is visible and well-lit.");
      }
      return;
    }

    // ── Batch mode: analyze captured frames now ──
    const frames = [...framesRef.current];
    framesRef.current = [];

    if (frames.length === 0) {
      await sessions.end(sid);
      setSessionNote("Session was too short — no frames captured. Record for at least 5 seconds.");
      return;
    }

    setAnalyzing(true);
    setProgress({ done: 0, total: frames.length });

    const newReadings: Reading[] = [];
    let latestResult: PredictResult | null = null;
    let anyFaceFound = false;
    let analyzeError: string | null = null;

    for (let i = 0; i < frames.length; i++) {
      try {
        const res = await predict.analyze(frames[i].data, sid);
        latestResult = res.data;
        if (res.data.face_found) anyFaceFound = true;
        if (res.data.face_found && res.data.emotion) {
          newReadings.push({
            id: i,
            emotion: res.data.emotion,
            confidence: res.data.confidence,
            probabilities: res.data.probabilities,
            timestamp: new Date(frames[i].capturedAt).toISOString(),
          });
        }
      } catch (err) {
        console.error("Batch predict error on frame", i, err);
        analyzeError = err instanceof Error ? err.message : String(err);
      }
      setProgress({ done: i + 1, total: frames.length });
    }

    await sessions.end(sid);

    setAnalyzing(false);
    setProgress(null);
    setLastResult(latestResult);
    setReadings(newReadings);

    if (!anyFaceFound && analyzeError) {
      setSessionNote(`Analysis failed: ${analyzeError}. Check the browser console for details.`);
    } else if (!anyFaceFound) {
      setSessionNote("No face detected in any frame. Make sure your face is well-lit and centered.");
    } else if (newReadings.length === 0) {
      setSessionNote("Face detected but confidence was too low. Try better lighting or move closer.");
    }
  };

  const getJournalPrompt = async () => {
    if (!sessionId) return;
    const res = await journal.prompt(sessionId);
    setJournalPrompt(res.data.prompt);
  };

  // ── Derived display flags ──────────────────────────────────────────────
  const showBatchIdle   = running && mode === "batch";
  const showLiveDetecting = running && mode === "live" && !lastResult;
  const showResult      = !showBatchIdle && !showLiveDetecting;

  return (
    <div className="app">
      <Navbar onLogout={onLogout} theme={theme} onToggleTheme={onToggleTheme} />

      <main className="record-page">

        {/* ── LEFT: square webcam ── */}
        <div className="webcam-col">
          <div className={`webcam-card${running ? (mode === "live" ? " webcam-card--live" : " webcam-card--recording") : ""}`}>

            <div className="webcam-video-wrap">
              <video ref={videoRef} autoPlay muted playsInline className="webcam-video" />

              {running && (
                <div className={`webcam-top-badge${mode === "live" ? " webcam-top-badge--live" : ""}`}>
                  <span className="recording-dot" />
                  <span>{mode === "live" ? "LIVE" : "REC"}</span>
                  <span className="webcam-frame-count">{frameCount} frames</span>
                </div>
              )}
            </div>

            {/* Controls bar — always visible inside the card */}
            <div className="webcam-controls-bar">
              {!running && !analyzing && (
                <>
                  {/* Mode selector */}
                  <div className="mode-toggle">
                    <button
                      className={`mode-btn${mode === "batch" ? " mode-btn--active" : ""}`}
                      onClick={() => setMode("batch")}
                      title="Analyze after session ends"
                    >
                      📹 After session
                    </button>
                    <button
                      className={`mode-btn${mode === "live" ? " mode-btn--active mode-btn--live" : ""}`}
                      onClick={() => setMode("live")}
                      title="Detect emotions in real time"
                    >
                      ⚡ Live
                    </button>
                  </div>
                  <button className="btn btn--primary btn--lg" onClick={startSession}>
                    ▶ Start
                  </button>
                </>
              )}

              {running && (
                <>
                  <span className="webcam-hint">
                    {mode === "live" ? "Detecting live…" : "≥ 5 s for best results"}
                  </span>
                  <button className="btn btn--danger btn--lg" onClick={stopSession}>
                    ■ End Session
                  </button>
                </>
              )}

              {analyzing && progress && (
                <div className="progress-wrap">
                  <div className="progress-header">
                    <span className="progress-label">Analyzing {progress.total} frames…</span>
                    <span className="progress-count">{progress.done}/{progress.total}</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar__fill" style={{ width: `${(progress.done / progress.total) * 100}%` }} />
                  </div>
                </div>
              )}
            </div>
          </div>

          {sessionNote && (
            <div className="session-note session-note--warn">⚠️ {sessionNote}</div>
          )}
        </div>

        {/* ── RIGHT: detection result + session summary ── */}
        <div className="record-col-right">
          <EmotionDisplay
            result={showResult ? lastResult : null}
            running={showBatchIdle}
            detecting={showLiveDetecting}
          />

          <div className="timeline-card">
            <EmotionTimeline readings={readings} />
            {!running && !analyzing && readings.length > 0 && (
              <div className="journal-section">
                <button className="btn btn--secondary" onClick={getJournalPrompt}>
                  Get journaling prompt
                </button>
                {journalPrompt && (
                  <blockquote className="journal-prompt">{journalPrompt}</blockquote>
                )}
              </div>
            )}
          </div>
        </div>

      </main>
    </div>
  );
}
