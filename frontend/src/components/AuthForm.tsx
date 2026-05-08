import { useState } from "react";
import { auth } from "../api";
import type { Theme } from "../hooks/useTheme";

interface Props {
  onSuccess: (token: string) => void;
  theme: Theme;
  onToggleTheme: () => void;
}

export function AuthForm({ onSuccess, theme, onToggleTheme }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const fn = mode === "login" ? auth.login : auth.register;
      const res = await fn(email, password);
      onSuccess(res.data.access_token);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      if (Array.isArray(detail)) {
        setError(detail.map((d: any) => d.msg ?? String(d)).join(", "));
      } else {
        setError(detail ?? "Something went wrong");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-page__bg" />
      <button
        className="auth-theme-toggle"
        onClick={onToggleTheme}
        title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      >
        {theme === "dark" ? "☀️" : "🌙"}
      </button>
      <div className="auth-card">
        <div className="auth-card__brand">
          <span className="auth-card__logo">🪞</span>
          <h1 className="auth-card__name">EmotionMirror</h1>
          <p className="auth-card__tagline">Understand your emotions in real time</p>
        </div>

        <form className="auth-card__form" onSubmit={submit}>
          <div className="form-field">
            <label className="form-label">Email</label>
            <input
              className="form-input"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="form-field">
            <label className="form-label">Password</label>
            <input
              className="form-input"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {error && <p className="form-error">{error}</p>}

          <button className="btn btn--primary btn--full btn--lg" type="submit" disabled={loading}>
            {loading ? "…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <button
          className="auth-card__toggle"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setError("");
          }}
        >
          {mode === "login"
            ? "Don't have an account? Register"
            : "Already have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}
