import axios from "axios";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const api = axios.create({ baseURL: BASE });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default api;

export type Emotion = "angry" | "disgust" | "fear" | "happy" | "neutral" | "sad" | "surprise";

export interface PredictResult {
  emotion: Emotion | null;
  confidence: number;
  probabilities: Record<Emotion, number>;
  face_found: boolean;
}

export interface Reading {
  id: number;
  emotion: Emotion;
  confidence: number;
  probabilities: Record<Emotion, number>;
  timestamp: string;
}

export interface Session {
  id: number;
  started_at: string;
  ended_at: string | null;
  note: string | null;
  readings: Reading[];
}

export const auth = {
  register: (email: string, password: string) =>
    api.post<{ access_token: string }>("/api/auth/register", { email, password }),
  login: (email: string, password: string) => {
    const form = new URLSearchParams({ username: email, password });
    return api.post<{ access_token: string }>("/api/auth/login", form, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
  },
};

export const sessions = {
  start: (note?: string) => api.post<Session>("/api/sessions", { note }),
  end: (id: number, note?: string) => api.patch<Session>(`/api/sessions/${id}/end`, { note }),
  list: () => api.get<Session[]>("/api/sessions"),
  get: (id: number) => api.get<Session>(`/api/sessions/${id}`),
};

export const journal = {
  prompt: (sessionId: number, note?: string) =>
    api.post<{ prompt: string; session_id: number }>("/api/journal/prompt", {
      session_id: sessionId,
      note,
    }),
};

export const predict = {
  analyze: (image: string, sessionId: number) =>
    api.post<PredictResult>("/api/predict", { image, session_id: sessionId }),
};

export function createWSUrl(token: string) {
  const wsBase = BASE.replace(/^http/, "ws");
  return `${wsBase}/api/ws/stream?token=${token}`;
}
