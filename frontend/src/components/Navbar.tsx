import { NavLink } from "react-router-dom";
import type { Theme } from "../hooks/useTheme";

interface Props {
  onLogout: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

export function Navbar({ onLogout, theme, onToggleTheme }: Props) {
  return (
    <nav className="navbar">
      <div className="navbar__brand">
        <span className="navbar__logo">🪞</span>
        <span className="navbar__title">EmotionMirror</span>
      </div>

      <div className="navbar__links">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `navbar__link${isActive ? " navbar__link--active" : ""}`
          }
        >
          Record
        </NavLink>
        <NavLink
          to="/dashboard"
          className={({ isActive }) =>
            `navbar__link${isActive ? " navbar__link--active" : ""}`
          }
        >
          Dashboard
        </NavLink>
      </div>

      <div className="navbar__actions">
        <button
          className="btn btn--ghost btn--sm btn--icon"
          onClick={onToggleTheme}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
        <button className="btn btn--ghost btn--sm" onClick={onLogout}>
          Sign out
        </button>
      </div>
    </nav>
  );
}
