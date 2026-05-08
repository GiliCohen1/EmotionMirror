import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Navbar } from "../components/Navbar";
import { sessions } from "../api";
import type { Session, Emotion } from "../api";
import type { Theme } from "../hooks/useTheme";

const EMOJI: Record<Emotion, string> = {
  angry: "😠", disgust: "🤢", fear: "😨",
  happy: "😊", neutral: "😐", sad: "😢", surprise: "😲",
};

const COLOR: Record<Emotion, string> = {
  angry: "#f87171", disgust: "#86efac", fear: "#c084fc",
  happy: "#fbbf24", neutral: "#94a3b8", sad: "#60a5fa", surprise: "#fb923c",
};

interface Props {
  onLogout: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

function dominantEmotion(readings: Session["readings"]): Emotion | null {
  if (!readings.length) return null;
  const counts = readings.reduce((acc, r) => {
    const e = r.emotion as Emotion;
    acc[e] = (acc[e] ?? 0) + 1;
    return acc;
  }, {} as Record<Emotion, number>);
  return Object.entries(counts).sort(([, a], [, b]) => b - a)[0][0] as Emotion;
}

function formatDuration(s: Session): string {
  if (!s.ended_at) return "In progress";
  const secs = Math.round(
    (new Date(s.ended_at).getTime() - new Date(s.started_at).getTime()) / 1000
  );
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

function formatDate(d: string): string {
  return new Date(d).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function DashboardPage({ onLogout, theme, onToggleTheme }: Props) {
  const [sessionList, setSessionList] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    sessions
      .list()
      .then((r) => setSessionList(r.data))
      .finally(() => setLoading(false));
  }, []);

  const allReadings = sessionList.flatMap((s) => s.readings);

  const totalMs = sessionList.reduce((acc, s) => {
    if (!s.ended_at) return acc;
    return acc + (new Date(s.ended_at).getTime() - new Date(s.started_at).getTime());
  }, 0);
  const totalMin = Math.round(totalMs / 60000);

  const emotionCounts = allReadings.reduce((acc, r) => {
    const e = r.emotion as Emotion;
    acc[e] = (acc[e] ?? 0) + 1;
    return acc;
  }, {} as Record<Emotion, number>);

  const topEmotion = (
    Object.entries(emotionCounts).sort(([, a], [, b]) => b - a)[0]?.[0] as Emotion | undefined
  );

  return (
    <div className="app">
      <Navbar onLogout={onLogout} theme={theme} onToggleTheme={onToggleTheme} />

      <main className="dashboard-page">
        <div className="dashboard-header">
          <h2 className="dashboard-title">Dashboard</h2>
          <Link to="/" className="btn btn--primary btn--sm">
            ▶ New session
          </Link>
        </div>

        {/* Stats row */}
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-card__value">{sessionList.length}</div>
            <div className="stat-card__label">Sessions</div>
          </div>
          <div className="stat-card">
            <div className="stat-card__value">{totalMin}m</div>
            <div className="stat-card__label">Recorded</div>
          </div>
          <div className="stat-card">
            <div className="stat-card__value">{allReadings.length}</div>
            <div className="stat-card__label">Detections</div>
          </div>
          <div className="stat-card">
            {topEmotion ? (
              <>
                <div className="stat-card__value" style={{ color: COLOR[topEmotion] }}>
                  {EMOJI[topEmotion]} {topEmotion}
                </div>
                <div className="stat-card__label">Top emotion</div>
              </>
            ) : (
              <>
                <div className="stat-card__value">—</div>
                <div className="stat-card__label">Top emotion</div>
              </>
            )}
          </div>
        </div>

        {/* Overall emotion distribution */}
        {allReadings.length > 0 && (
          <div className="dash-card">
            <h3 className="dash-card__title">Overall Emotion Distribution</h3>
            <div className="dist-bars">
              {(Object.entries(emotionCounts) as [Emotion, number][])
                .sort(([, a], [, b]) => b - a)
                .map(([e, count]) => {
                  const pct = Math.round((count / allReadings.length) * 100);
                  return (
                    <div key={e} className="dist-bar-row">
                      <span className="dist-bar-row__emoji">{EMOJI[e]}</span>
                      <span className="dist-bar-row__label">{e}</span>
                      <div className="dist-bar-row__track">
                        <div
                          className="dist-bar-row__fill"
                          style={{ width: `${pct}%`, background: COLOR[e] }}
                        />
                      </div>
                      <span className="dist-bar-row__pct">{pct}%</span>
                    </div>
                  );
                })}
            </div>
          </div>
        )}

        {/* Session history */}
        <div className="dash-card">
          <h3 className="dash-card__title">Session History</h3>

          {loading ? (
            <div className="dashboard-empty">Loading…</div>
          ) : sessionList.length === 0 ? (
            <div className="dashboard-empty">
              <span>📭</span>
              <p>No sessions yet. <Link to="/" className="link">Record your first session!</Link></p>
            </div>
          ) : (
            <div className="session-list">
              {sessionList.map((s) => {
                const dominant = dominantEmotion(s.readings);
                const isExpanded = expandedId === s.id;

                return (
                  <div key={s.id} className={`session-item ${isExpanded ? "session-item--expanded" : ""}`}>
                    <button
                      className="session-item__row"
                      onClick={() => setExpandedId(isExpanded ? null : s.id)}
                    >
                      <div className="session-item__date">{formatDate(s.started_at)}</div>
                      <div className="session-item__meta">
                        <span className="session-item__duration">{formatDuration(s)}</span>
                        <span className="session-item__count">{s.readings.length} detections</span>
                      </div>
                      <div className="session-item__emotion">
                        {dominant ? (
                          <span style={{ color: COLOR[dominant] }}>
                            {EMOJI[dominant]} {dominant}
                          </span>
                        ) : (
                          <span className="text-subtle">No detections</span>
                        )}
                      </div>
                      <span className="session-item__chevron">{isExpanded ? "▲" : "▼"}</span>
                    </button>

                    {isExpanded && s.readings.length > 0 && (
                      <div className="session-item__detail">
                        <div className="detail-bars">
                          {(() => {
                            const avgProbs = (["happy","neutral","sad","angry","fear","surprise","disgust"] as Emotion[]).reduce((acc, e) => {
                              acc[e] = s.readings.reduce((sum, r) => sum + (r.probabilities[e] ?? 0), 0) / s.readings.length;
                              return acc;
                            }, {} as Record<Emotion, number>);
                            return (Object.entries(avgProbs) as [Emotion, number][])
                              .sort(([, a], [, b]) => b - a)
                              .map(([e, v]) => (
                                <div key={e} className="dist-bar-row dist-bar-row--sm">
                                  <span className="dist-bar-row__emoji">{EMOJI[e]}</span>
                                  <span className="dist-bar-row__label">{e}</span>
                                  <div className="dist-bar-row__track">
                                    <div
                                      className="dist-bar-row__fill"
                                      style={{ width: `${v * 100}%`, background: COLOR[e] }}
                                    />
                                  </div>
                                  <span className="dist-bar-row__pct">{Math.round(v * 100)}%</span>
                                </div>
                              ));
                          })()}
                        </div>
                      </div>
                    )}

                    {isExpanded && s.readings.length === 0 && (
                      <div className="session-item__detail text-subtle" style={{ padding: "0.75rem 1rem" }}>
                        No emotion detections in this session.
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
