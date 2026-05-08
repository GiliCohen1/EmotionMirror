import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthForm } from "./components/AuthForm";
import { RecordPage } from "./pages/RecordPage";
import { DashboardPage } from "./pages/DashboardPage";
import { useTheme } from "./hooks/useTheme";
import "./App.css";

export default function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem("token"));
  const { theme, toggle: toggleTheme } = useTheme();

  const handleLogin = (t: string) => {
    localStorage.setItem("token", t);
    setToken(t);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setToken(null);
  };

  if (!token) return <AuthForm onSuccess={handleLogin} theme={theme} onToggleTheme={toggleTheme} />;

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={<RecordPage onLogout={handleLogout} theme={theme} onToggleTheme={toggleTheme} />}
        />
        <Route
          path="/dashboard"
          element={<DashboardPage onLogout={handleLogout} theme={theme} onToggleTheme={toggleTheme} />}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
