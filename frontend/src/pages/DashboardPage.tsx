import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { Navbar } from "../components/Navbar";
import { sessions } from "../api";
import type { Session, Emotion, Reading } from "../api";
import type { Theme } from "../hooks/useTheme";

/* ── constants ─────────────────────────────────────────── */

const EMOTIONS: Emotion[] = ["happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"];

const EMOJI: Record<Emotion, string> = {
  angry: "😠", disgust: "🤢", fear: "😨",
  happy: "😊", neutral: "😐", sad: "😢", surprise: "😲",
};

const COLOR: Record<Emotion, string> = {
  angry: "#f87171", disgust: "#86efac", fear: "#c084fc",
  happy: "#fbbf24", neutral: "#94a3b8", sad: "#60a5fa", surprise: "#fb923c",
};

type Period = "day" | "week" | "month" | "year";

interface Bucket {
  label: string;
  fullLabel: string;
  readings: Reading[];
  total: number;
  happy: number; neutral: number; sad: number;
  angry: number; fear: number; surprise: number; disgust: number;
}

/* ── helpers ────────────────────────────────────────────── */

function toPercents(readings: Reading[]): Record<Emotion, number> {
  const total = readings.length;
  if (total === 0) return Object.fromEntries(EMOTIONS.map(e => [e, 0])) as Record<Emotion, number>;
  const counts = Object.fromEntries(
    EMOTIONS.map(e => [e, readings.filter(r => r.emotion === e).length])
  ) as Record<Emotion, number>;
  return Object.fromEntries(
    EMOTIONS.map(e => [e, parseFloat(((counts[e] / total) * 100).toFixed(1))])
  ) as Record<Emotion, number>;
}

function makeBucket(label: string, fullLabel: string, readings: Reading[]): Bucket {
  return { label, fullLabel, readings, total: readings.length, ...toPercents(readings) };
}

function readingsInRange(all: Reading[], from: Date, to: Date): Reading[] {
  const f = from.getTime(), t = to.getTime();
  return all.filter(r => { const ms = new Date(r.timestamp).getTime(); return ms >= f && ms < t; });
}

function buildBuckets(period: Period, allReadings: Reading[]): Bucket[] {
  const now = new Date();
  if (period === "day") {
    return Array.from({ length: 24 }, (_, h) => {
      const from = new Date(now.getFullYear(), now.getMonth(), now.getDate(), h);
      const to   = new Date(from.getTime() + 3_600_000);
      const lbl  = from.toLocaleTimeString([], { hour: "numeric", hour12: true });
      const full = `${from.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} – ${to.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
      return makeBucket(lbl, full, readingsInRange(allReadings, from, to));
    });
  }
  if (period === "week") {
    return Array.from({ length: 7 }, (_, i) => {
      const d = new Date(now); d.setDate(now.getDate() - 6 + i);
      const from = new Date(d.getFullYear(), d.getMonth(), d.getDate());
      const to   = new Date(from.getTime() + 86_400_000);
      const lbl  = from.toLocaleDateString([], { weekday: "short" });
      const full = from.toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" });
      return makeBucket(lbl, full, readingsInRange(allReadings, from, to));
    });
  }
  if (period === "month") {
    return Array.from({ length: 30 }, (_, i) => {
      const d = new Date(now); d.setDate(now.getDate() - 29 + i);
      const from = new Date(d.getFullYear(), d.getMonth(), d.getDate());
      const to   = new Date(from.getTime() + 86_400_000);
      const lbl  = String(from.getDate());
      const full = from.toLocaleDateString([], { month: "long", day: "numeric" });
      return makeBucket(lbl, full, readingsInRange(allReadings, from, to));
    });
  }
  // year — 12 months
  return Array.from({ length: 12 }, (_, i) => {
    const from = new Date(now.getFullYear(), i, 1);
    const to   = new Date(now.getFullYear(), i + 1, 1);
    const lbl  = from.toLocaleDateString([], { month: "short" });
    const full = from.toLocaleDateString([], { month: "long", year: "numeric" });
    return makeBucket(lbl, full, readingsInRange(allReadings, from, to));
  });
}

function periodLabel(period: Period): string {
  const now = new Date();
  if (period === "day")   return `Today, ${now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" })}`;
  if (period === "week")  return `Last 7 days`;
  if (period === "month") return `Last 30 days`;
  return `${now.getFullYear()}`;
}

function formatDuration(s: Session): string {
  if (!s.ended_at) return "In progress";
  const secs = Math.round((new Date(s.ended_at).getTime() - new Date(s.started_at).getTime()) / 1000);
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

function dominantEmotion(readings: Session["readings"]): Emotion | null {
  if (!readings.length) return null;
  const counts = readings.reduce((a, r) => { const e = r.emotion as Emotion; a[e] = (a[e] ?? 0) + 1; return a; }, {} as Record<Emotion, number>);
  return Object.entries(counts).sort(([, a], [, b]) => b - a)[0][0] as Emotion;
}

/* ── custom tooltip ─────────────────────────────────────── */
function ChartTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const bucket: Bucket = payload[0]?.payload;
  if (!bucket?.total) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__header">{bucket.fullLabel} · {bucket.total} detection{bucket.total !== 1 ? "s" : ""}</div>
      {EMOTIONS
        .filter(e => bucket[e] > 0)
        .sort((a, b) => bucket[b] - bucket[a])
        .map(e => (
          <div key={e} className="chart-tooltip__row">
            <span>{EMOJI[e]}</span>
            <span className="chart-tooltip__name">{e}</span>
            <span className="chart-tooltip__pct" style={{ color: COLOR[e] }}>{bucket[e]}%</span>
          </div>
        ))}
    </div>
  );
}

/* ── component ──────────────────────────────────────────── */

interface Props { onLogout: () => void; theme: Theme; onToggleTheme: () => void; }

export function DashboardPage({ onLogout, theme, onToggleTheme }: Props) {
  const [sessionList, setSessionList] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState<Period>("week");
  const [selectedBucket, setSelectedBucket] = useState<Bucket | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    sessions.list().then(r => setSessionList(r.data)).finally(() => setLoading(false));
  }, []);

  const allReadings = sessionList.flatMap(s => s.readings);
  const buckets = buildBuckets(period, allReadings);

  const totalMs = sessionList.reduce((a, s) => {
    if (!s.ended_at) return a;
    return a + new Date(s.ended_at).getTime() - new Date(s.started_at).getTime();
  }, 0);

  const emotionCounts = allReadings.reduce((a, r) => {
    const e = r.emotion as Emotion; a[e] = (a[e] ?? 0) + 1; return a;
  }, {} as Record<Emotion, number>);

  const topEmotion = (Object.entries(emotionCounts).sort(([, a], [, b]) => b - a)[0]?.[0] as Emotion | undefined);

  const onBarClick = useCallback((data: any) => {
    const b: Bucket = data?.activePayload?.[0]?.payload;
    if (!b) return;
    setSelectedBucket(prev => (prev?.label === b.label && prev?.fullLabel === b.fullLabel ? null : b));
  }, []);

  const periodHasData = buckets.some(b => b.total > 0);

  const xInterval = period === "day" ? 3 : period === "month" ? 4 : 0;

  return (
    <div className="app">
      <Navbar onLogout={onLogout} theme={theme} onToggleTheme={onToggleTheme} />
      <main className="dashboard-page">

        {/* ── Header + stats ── */}
        <div className="dashboard-header">
          <h2 className="dashboard-title">Dashboard</h2>
          <Link to="/" className="btn btn--primary btn--sm">▶ New session</Link>
        </div>

        <div className="stats-strip">
          <div className="stat-pill"><span className="stat-pill__val">{sessionList.length}</span><span className="stat-pill__lbl">sessions</span></div>
          <div className="stat-pill"><span className="stat-pill__val">{Math.round(totalMs / 60000)}m</span><span className="stat-pill__lbl">recorded</span></div>
          <div className="stat-pill"><span className="stat-pill__val">{allReadings.length}</span><span className="stat-pill__lbl">detections</span></div>
          {topEmotion && (
            <div className="stat-pill">
              <span className="stat-pill__val" style={{ color: COLOR[topEmotion] }}>{EMOJI[topEmotion]} {topEmotion}</span>
              <span className="stat-pill__lbl">top emotion</span>
            </div>
          )}
        </div>

        {/* ── Emotion timeline chart ── */}
        <div className="dash-card">
          <div className="chart-header">
            <h3 className="dash-card__title" style={{ marginBottom: 0 }}>Emotions Over Time</h3>
            <div className="period-tabs">
              {(["day", "week", "month", "year"] as Period[]).map(p => (
                <button
                  key={p}
                  className={`period-tab${period === p ? " period-tab--active" : ""}`}
                  onClick={() => { setPeriod(p); setSelectedBucket(null); }}
                >
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div className="period-date-label">{periodLabel(period)}</div>

          {loading ? (
            <div className="chart-empty">Loading…</div>
          ) : !periodHasData ? (
            <div className="chart-empty">
              <span>📭</span>
              <p>No recordings for this period</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={buckets} onClick={onBarClick} style={{ cursor: "pointer" }}
                margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <XAxis
                  dataKey="label"
                  tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                  axisLine={false} tickLine={false}
                  interval={xInterval}
                />
                <YAxis
                  tickFormatter={v => `${v}%`}
                  domain={[0, 100]}
                  tick={{ fontSize: 10, fill: "var(--text-muted)" }}
                  axisLine={false} tickLine={false}
                  width={36}
                />
                <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--surface-3)", radius: 4 }} />
                {EMOTIONS.map(e => (
                  <Bar key={e} dataKey={e} stackId="s" fill={COLOR[e]} maxBarSize={32} radius={e === "disgust" ? [3, 3, 0, 0] : [0, 0, 0, 0]}>
                    {buckets.map((b, i) => (
                      <Cell
                        key={i}
                        fill={COLOR[e]}
                        opacity={selectedBucket ? (selectedBucket.label === b.label && selectedBucket.fullLabel === b.fullLabel ? 1 : 0.35) : 0.88}
                      />
                    ))}
                  </Bar>
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}

          {/* Emotion legend */}
          <div className="chart-legend">
            {EMOTIONS.map(e => (
              <span key={e} className="legend-item">
                <span className="legend-dot" style={{ background: COLOR[e] }} />
                {e}
              </span>
            ))}
          </div>

          {/* ── Drill-down detail panel ── */}
          {selectedBucket && (
            <div className="drill-panel">
              <div className="drill-panel__header">
                <span className="drill-panel__title">{selectedBucket.fullLabel}</span>
                <span className="drill-panel__count">{selectedBucket.total} detection{selectedBucket.total !== 1 ? "s" : ""}</span>
                <button className="drill-panel__close" onClick={() => setSelectedBucket(null)}>✕</button>
              </div>

              {selectedBucket.total === 0 ? (
                <p className="drill-panel__empty">No recordings in this period.</p>
              ) : (
                <div className="drill-readings">
                  {selectedBucket.readings
                    .slice()
                    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
                    .map((r, i) => {
                      const e = r.emotion as Emotion;
                      return (
                        <div key={i} className="drill-reading">
                          <span className="drill-reading__time">
                            {new Date(r.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                          </span>
                          <span className="drill-reading__emoji">{EMOJI[e]}</span>
                          <span className="drill-reading__emotion" style={{ color: COLOR[e] }}>{e}</span>
                          <div className="drill-reading__bar-wrap">
                            <div className="drill-reading__bar" style={{ width: `${r.confidence * 100}%`, background: COLOR[e] }} />
                          </div>
                          <span className="drill-reading__conf">{Math.round(r.confidence * 100)}%</span>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Session history ── */}
        <div className="dash-card">
          <h3 className="dash-card__title">Session History</h3>
          {loading ? (
            <div className="dashboard-empty">Loading…</div>
          ) : sessionList.length === 0 ? (
            <div className="dashboard-empty">
              <span>📭</span>
              <p>No sessions yet. <Link to="/" className="link">Record your first!</Link></p>
            </div>
          ) : (
            <div className="session-list">
              {sessionList.map(s => {
                const dom = dominantEmotion(s.readings);
                const expanded = expandedId === s.id;
                return (
                  <div key={s.id} className={`session-item${expanded ? " session-item--expanded" : ""}`}>
                    <button className="session-item__row" onClick={() => setExpandedId(expanded ? null : s.id)}>
                      <span className="session-item__date">
                        {new Date(s.started_at).toLocaleDateString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                      </span>
                      <span className="session-item__duration">{formatDuration(s)}</span>
                      <span className="session-item__count">{s.readings.length} det.</span>
                      <span className="session-item__emotion">
                        {dom ? <span style={{ color: COLOR[dom] }}>{EMOJI[dom]} {dom}</span> : <span className="text-subtle">—</span>}
                      </span>
                      <span className="session-item__chevron">{expanded ? "▲" : "▼"}</span>
                    </button>
                    {expanded && s.readings.length > 0 && (
                      <div className="session-item__detail">
                        {(() => {
                          const avg = EMOTIONS.reduce((a, e) => {
                            a[e] = s.readings.reduce((s, r) => s + (r.probabilities[e] ?? 0), 0) / s.readings.length;
                            return a;
                          }, {} as Record<Emotion, number>);
                          return (Object.entries(avg) as [Emotion, number][]).sort(([,a],[,b]) => b-a).map(([e, v]) => (
                            <div key={e} className="dist-bar-row dist-bar-row--sm">
                              <span className="dist-bar-row__emoji">{EMOJI[e]}</span>
                              <span className="dist-bar-row__label">{e}</span>
                              <div className="dist-bar-row__track"><div className="dist-bar-row__fill" style={{ width: `${v*100}%`, background: COLOR[e] }} /></div>
                              <span className="dist-bar-row__pct">{Math.round(v*100)}%</span>
                            </div>
                          ));
                        })()}
                      </div>
                    )}
                    {expanded && s.readings.length === 0 && (
                      <div className="session-item__detail text-subtle" style={{ padding: "0.75rem 1rem" }}>No detections in this session.</div>
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
