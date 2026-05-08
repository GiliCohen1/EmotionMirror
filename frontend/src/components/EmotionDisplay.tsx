import type { PredictResult, Emotion } from "../api";

const EMOJI: Record<Emotion, string> = {
  angry: "😠", disgust: "🤢", fear: "😨",
  happy: "😊", neutral: "😐", sad: "😢", surprise: "😲",
};

const COLOR: Record<Emotion, string> = {
  angry: "#f87171",
  disgust: "#86efac",
  fear: "#c084fc",
  happy: "#fbbf24",
  neutral: "#94a3b8",
  sad: "#60a5fa",
  surprise: "#fb923c",
};

interface Props {
  result: PredictResult | null;
  running?: boolean;
}

export function EmotionDisplay({ result, running }: Props) {
  if (running) {
    return (
      <div className="emotion-card emotion-card--idle">
        <div className="emotion-card__pulse" />
        <p className="emotion-card__idle-text">Session in progress…</p>
        <p className="emotion-card__idle-sub">Results will appear after you end the session</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="emotion-card emotion-card--idle">
        <div className="emotion-card__idle-icon">🪞</div>
        <p className="emotion-card__idle-text">Ready to record</p>
        <p className="emotion-card__idle-sub">Press Start Session to begin</p>
      </div>
    );
  }

  if (!result.face_found) {
    return (
      <div className="emotion-card emotion-card--idle">
        <div className="emotion-card__idle-icon">👤</div>
        <p className="emotion-card__idle-text">No face detected</p>
        <p className="emotion-card__idle-sub">Make sure your face is visible in frame</p>
      </div>
    );
  }

  const emotion = result.emotion!;
  const color = COLOR[emotion];
  const pct = Math.round(result.confidence * 100);
  const probs = result.probabilities;

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="emotion-card">
      <div className="emotion-card__ring-wrap">
        <svg className="emotion-card__ring" viewBox="0 0 120 120" width="120" height="120">
          <circle
            cx="60" cy="60" r={radius}
            fill="none"
            stroke="var(--surface-3)"
            strokeWidth="8"
          />
          <circle
            cx="60" cy="60" r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            transform="rotate(-90 60 60)"
            style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.4s ease" }}
          />
        </svg>
        <div className="emotion-card__ring-center">
          <span className="emotion-card__emoji">{EMOJI[emotion]}</span>
        </div>
      </div>

      <div className="emotion-card__label" style={{ color }}>
        {emotion}
      </div>
      <div className="emotion-card__confidence">{pct}% confidence</div>

      <div className="emotion-card__bars">
        {(Object.entries(probs) as [Emotion, number][])
          .sort(([, a], [, b]) => b - a)
          .map(([e, p]) => (
            <div key={e} className="emo-bar">
              <span className="emo-bar__emoji">{EMOJI[e]}</span>
              <span className="emo-bar__label">{e}</span>
              <div className="emo-bar__track">
                <div
                  className="emo-bar__fill"
                  style={{ width: `${p * 100}%`, background: COLOR[e] }}
                />
              </div>
              <span className="emo-bar__pct">{Math.round(p * 100)}%</span>
            </div>
          ))}
      </div>
    </div>
  );
}
