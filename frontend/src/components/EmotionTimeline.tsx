import type { Reading, Emotion } from "../api";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  Legend, ResponsiveContainer, CartesianGrid,
} from "recharts";

const EMOTION_COLORS: Record<Emotion, string> = {
  angry: "#ef4444", disgust: "#84cc16", fear: "#a855f7",
  happy: "#eab308", neutral: "#6b7280", sad: "#3b82f6", surprise: "#f97316",
};

const EMOTIONS: Emotion[] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];

interface Props {
  readings: Reading[];
}

export function EmotionTimeline({ readings }: Props) {
  if (readings.length === 0) {
    return <p className="timeline-empty">No readings yet in this session.</p>;
  }

  const data = readings.map((r) => ({
    time: new Date(r.timestamp).toLocaleTimeString(),
    ...r.probabilities,
  }));

  return (
    <div className="timeline">
      <h3 className="timeline__title">Emotion Timeline</h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} domain={[0, 1]} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v) => `${Math.round((v as number) * 100)}%`} />
          <Legend />
          {EMOTIONS.map((e) => (
            <Line
              key={e}
              type="monotone"
              dataKey={e}
              stroke={EMOTION_COLORS[e]}
              dot={false}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
