import type { PredictResult, Emotion } from "../api";

const EMOJI: Record<Emotion, string> = {
  angry: "😠", disgust: "🤢", fear: "😨",
  happy: "😊", neutral: "😐", sad: "😢", surprise: "😲",
};

const COLOR: Record<Emotion, string> = {
  angry: "#ef4444", disgust: "#84cc16", fear: "#a855f7",
  happy: "#eab308", neutral: "#6b7280", sad: "#3b82f6", surprise: "#f97316",
};

interface Props {
  result: PredictResult | null;
}

export function EmotionDisplay({ result }: Props) {
  if (!result) {
    return <div className="emotion-display emotion-display--empty">Waiting for feed…</div>;
  }
  if (!result.face_found) {
    return <div className="emotion-display emotion-display--no-face">No face detected</div>;
  }

  const emotion = result.emotion!;
  const probs = result.probabilities;

  return (
    <div className="emotion-display">
      <div className="emotion-display__primary" style={{ color: COLOR[emotion] }}>
        <span className="emotion-display__emoji">{EMOJI[emotion]}</span>
        <span className="emotion-display__label">{emotion}</span>
        <span className="emotion-display__confidence">{Math.round(result.confidence * 100)}%</span>
      </div>

      <div className="emotion-display__bars">
        {(Object.entries(probs) as [Emotion, number][])
          .sort(([, a], [, b]) => b - a)
          .map(([e, p]) => (
            <div key={e} className="emotion-bar">
              <span className="emotion-bar__label">{e}</span>
              <div className="emotion-bar__track">
                <div
                  className="emotion-bar__fill"
                  style={{ width: `${p * 100}%`, background: COLOR[e] }}
                />
              </div>
              <span className="emotion-bar__pct">{Math.round(p * 100)}%</span>
            </div>
          ))}
      </div>
    </div>
  );
}
