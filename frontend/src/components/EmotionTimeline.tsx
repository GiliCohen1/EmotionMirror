import type { Reading, Emotion } from "../api";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";

const EMOTION_COLORS: Record<Emotion, string> = {
  angry: "#f87171",
  disgust: "#86efac",
  fear: "#c084fc",
  happy: "#fbbf24",
  neutral: "#94a3b8",
  sad: "#60a5fa",
  surprise: "#fb923c",
};

const EMOJI: Record<Emotion, string> = {
  angry: "😠", disgust: "🤢", fear: "😨",
  happy: "😊", neutral: "😐", sad: "😢", surprise: "😲",
};

const EMOTIONS: Emotion[] = ["happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"];

interface Props {
  readings: Reading[];
}

export function EmotionTimeline({ readings }: Props) {
  if (readings.length === 0) {
    return (
      <>
        <h3 className="timeline__title">Session Summary</h3>
        <div className="timeline-empty">
          <span>📊</span>
          <p>Timeline and summary will appear here after your session ends</p>
        </div>
      </>
    );
  }

  // Average probability per emotion across all readings
  const avgProbs = EMOTIONS.reduce((acc, e) => {
    acc[e] = readings.reduce((s, r) => s + (r.probabilities[e] ?? 0), 0) / readings.length;
    return acc;
  }, {} as Record<Emotion, number>);

  const summaryData = EMOTIONS
    .map((e) => ({ emotion: e, value: avgProbs[e] }))
    .sort((a, b) => b.value - a.value);

  const dominantEmotion = summaryData[0].emotion;

  // Timeline data — map each reading to a data point with all probabilities
  const timelineData = readings.map((r) => ({
    time: new Date(r.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    ...r.probabilities,
  }));

  // Emotions that had any meaningful presence (avg > 3%) — keeps chart readable
  const activeEmotions = EMOTIONS.filter((e) => avgProbs[e] > 0.03);

  return (
    <>
      {/* ── Session Summary ── */}
      <h3 className="timeline__title">Session Summary</h3>
      <div className="summary-dominant">
        <span className="summary-dominant__emoji">{EMOJI[dominantEmotion]}</span>
        <span className="summary-dominant__label" style={{ color: EMOTION_COLORS[dominantEmotion] }}>
          {dominantEmotion}
        </span>
        <span className="summary-dominant__sub">dominant emotion</span>
      </div>

      <div className="summary-bars">
        {summaryData.map(({ emotion, value }) => (
          <div key={emotion} className="summary-bar-row">
            <span className="summary-bar-row__emoji">{EMOJI[emotion]}</span>
            <span className="summary-bar-row__label">{emotion}</span>
            <div className="summary-bar-row__track">
              <div
                className="summary-bar-row__fill"
                style={{ width: `${value * 100}%`, background: EMOTION_COLORS[emotion] }}
              />
            </div>
            <span className="summary-bar-row__pct">{Math.round(value * 100)}%</span>
          </div>
        ))}
      </div>

      {/* ── Timeline Chart ── */}
      {readings.length >= 2 ? (
        <>
          <h3 className="timeline__title" style={{ marginTop: "1.5rem" }}>Emotion Timeline</h3>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={timelineData} margin={{ top: 5, right: 16, bottom: 5, left: 0 }}>
              <defs>
                {activeEmotions.map((e) => (
                  <linearGradient key={e} id={`grad-${e}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={EMOTION_COLORS[e]} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={EMOTION_COLORS[e]} stopOpacity={0} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: "var(--text-muted)" }}
                axisLine={{ stroke: "var(--border)" }}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                domain={[0, 1]}
                tick={{ fontSize: 10, fill: "var(--text-muted)" }}
                axisLine={false}
                tickLine={false}
                width={38}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--border)",
                  borderRadius: "10px",
                  fontSize: "12px",
                  color: "var(--text)",
                }}
                formatter={(v) => [`${Math.round((v as number) * 100)}%`]}
                labelStyle={{ color: "var(--text-muted)", marginBottom: "4px" }}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px", paddingTop: "8px", color: "var(--text-muted)" }}
              />
              {activeEmotions.map((e) => (
                <Area
                  key={e}
                  type="monotone"
                  dataKey={e}
                  stroke={EMOTION_COLORS[e]}
                  fill={`url(#grad-${e})`}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </>
      ) : (
        <p className="timeline-single-note">
          Only 1 data point captured — record for longer to see the emotion timeline.
        </p>
      )}
    </>
  );
}
