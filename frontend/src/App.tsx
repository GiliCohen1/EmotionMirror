import { useState, useRef } from "react";
import { AuthForm } from "./components/AuthForm";
import { EmotionDisplay } from "./components/EmotionDisplay";
import { EmotionTimeline } from "./components/EmotionTimeline";
import { sessions, journal, predict } from "./api";
import type { Reading, PredictResult } from "./api";
import { useWebcam } from "./hooks/useWebcam";
import "./App.css";

type Frame = { data: string; capturedAt: number };

export default function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem("token"));
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [lastResult, setLastResult] = useState<PredictResult | null>(null);
  const [journalPrompt, setJournalPrompt] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);

  const { videoRef, start: startCam, stop: stopCam, captureBase64 } = useWebcam();
  const intervalRef = useRef<number | null>(null);
  const framesRef = useRef<Frame[]>([]);
  const sessionIdRef = useRef<number | null>(null);

  const handleLogin = (t: string) => {
    localStorage.setItem("token", t);
    setToken(t);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setToken(null);
  };

  const startSession = async () => {
    const res = await sessions.start();
    const sid = res.data.id;
    setSessionId(sid);
    sessionIdRef.current = sid;
    setReadings([]);
    setLastResult(null);
    setJournalPrompt(null);
    framesRef.current = [];
    await startCam();
    setRunning(true);
    // Capture a frame every 2 seconds and store with its timestamp
    intervalRef.current = window.setInterval(() => {
      const data = captureBase64();
      if (data) framesRef.current.push({ data, capturedAt: Date.now() });
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
    if (frames.length === 0) return;

    setAnalyzing(true);
    setProgress({ done: 0, total: frames.length });

    const newReadings: Reading[] = [];
    let latestResult: PredictResult | null = null;

    for (let i = 0; i < frames.length; i++) {
      try {
        const res = await predict.analyze(frames[i].data, sid);
        latestResult = res.data;
        if (res.data.face_found && res.data.emotion) {
          newReadings.push({
            id: i,
            emotion: res.data.emotion,
            confidence: res.data.confidence,
            probabilities: res.data.probabilities,
            // Use the actual capture time so the timeline is accurate
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
  };

  const getJournalPrompt = async () => {
    if (!sessionId) return;
    const res = await journal.prompt(sessionId);
    setJournalPrompt(res.data.prompt);
  };

  if (!token) return <AuthForm onSuccess={handleLogin} />;

  return (
    <div className="app">
      <header className="app__header">
        <h1>EmotionMirror</h1>
        <button className="btn btn--ghost" onClick={handleLogout}>Sign out</button>
      </header>

      <main className="app__main">
        <section className="webcam-section">
          <video ref={videoRef} autoPlay muted playsInline className="webcam-video" />
          <div className="webcam-controls">
            {!running && !analyzing ? (
              <button className="btn btn--primary" onClick={startSession}>Start session</button>
            ) : running ? (
              <button className="btn btn--danger" onClick={stopSession}>End session</button>
            ) : null}
            {running && <span className="badge badge--green">Recording</span>}
          </div>
        </section>

        {analyzing && progress && (
          <section className="analyzing-section">
            <p className="analyzing-label">
              Analyzing emotions… {progress.done} / {progress.total} frames
            </p>
            <div className="progress-bar">
              <div
                className="progress-bar__fill"
                style={{ width: `${(progress.done / progress.total) * 100}%` }}
              />
            </div>
          </section>
        )}

        {!running && lastResult && (
          <section className="result-section">
            <EmotionDisplay result={lastResult} />
          </section>
        )}

        <section className="timeline-section">
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
        </section>
      </main>
    </div>
  );
}
