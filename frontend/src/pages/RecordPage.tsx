import { useState, useRef } from "react";
import { EmotionDisplay } from "../components/EmotionDisplay";
import { EmotionTimeline } from "../components/EmotionTimeline";
import { Navbar } from "../components/Navbar";
import { sessions, journal, predict } from "../api";
import type { Reading, PredictResult } from "../api";
import { useWebcam } from "../hooks/useWebcam";
import type { Theme } from "../hooks/useTheme";

type Frame = { data: string; capturedAt: number };

interface Props {
  onLogout: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

export function RecordPage({ onLogout, theme, onToggleTheme }: Props) {
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [lastResult, setLastResult] = useState<PredictResult | null>(null);
  const [journalPrompt, setJournalPrompt] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [sessionNote, setSessionNote] = useState<string | null>(null);

  const { videoRef, start: startCam, stop: stopCam, captureBase64 } = useWebcam();
  const intervalRef = useRef<number | null>(null);
  const framesRef = useRef<Frame[]>([]);
  const sessionIdRef = useRef<number | null>(null);

  const startSession = async () => {
    const res = await sessions.start();
    const sid = res.data.id;
    setSessionId(sid);
    sessionIdRef.current = sid;
    setReadings([]);
    setLastResult(null);
    setJournalPrompt(null);
    setFrameCount(0);
    setSessionNote(null);
    framesRef.current = [];
    await startCam();
    setRunning(true);
    intervalRef.current = window.setInterval(() => {
      const data = captureBase64();
      if (data) {
        framesRef.current.push({ data, capturedAt: Date.now() });
        setFrameCount((n) => n + 1);
      }
    }, 2000);
  };

  const stopSession = async () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    stopCam();
    setRunning(false);

    const sid = sessionIdRef.current;
    if (!sid) return;

    await sessions.end(sid);

    const frames = [...framesRef.current];
    framesRef.current = [];

    if (frames.length === 0) {
      setSessionNote("Session was too short — no frames were captured. Record for at least 5 seconds.");
      return;
    }

    setAnalyzing(true);
    setProgress({ done: 0, total: frames.length });

    const newReadings: Reading[] = [];
    let latestResult: PredictResult | null = null;
    let anyFaceFound = false;

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
      } catch {}
      setProgress({ done: i + 1, total: frames.length });
    }

    setAnalyzing(false);
    setProgress(null);
    setLastResult(latestResult);
    setReadings(newReadings);

    if (!anyFaceFound) {
      setSessionNote("No face was detected in any frame. Make sure your face is well-lit and centered in the camera.");
    } else if (newReadings.length === 0) {
      setSessionNote("Face was detected but emotion confidence was too low. Try better lighting or move closer to the camera.");
    }
  };

  const getJournalPrompt = async () => {
    if (!sessionId) return;
    const res = await journal.prompt(sessionId);
    setJournalPrompt(res.data.prompt);
  };

  return (
    <div className="app">
      <Navbar onLogout={onLogout} theme={theme} onToggleTheme={onToggleTheme} />
      <main className="record-page">
        <div className="record-page__left">

          {/* Webcam card — controls overlaid at the bottom so they're always visible */}
          <div className={`webcam-card ${running ? "webcam-card--recording" : ""}`}>
            <video ref={videoRef} autoPlay muted playsInline className="webcam-video" />

            {running && (
              <div className="webcam-top-badge">
                <span className="recording-dot" />
                <span>Recording</span>
                <span className="webcam-frame-count">{frameCount} frames</span>
              </div>
            )}

            <div className="webcam-controls-bar">
              {!running && !analyzing && (
                <button className="btn btn--primary btn--lg" onClick={startSession}>
                  ▶ Start Session
                </button>
              )}
              {running && (
                <>
                  <span className="webcam-hint">
                    Record at least 5 seconds for best results
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
                    <span className="progress-count">{progress.done} / {progress.total}</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-bar__fill"
                      style={{ width: `${(progress.done / progress.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {sessionNote && (
            <div className="session-note session-note--warn">
              ⚠️ {sessionNote}
            </div>
          )}

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

        <aside className="record-page__right">
          <EmotionDisplay result={lastResult} running={running} />
        </aside>
      </main>
    </div>
  );
}
