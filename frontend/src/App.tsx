import { useState } from "react";
import AuthPage from "./AuthPage";
import HistoryPage from "./HistoryPage";
import PipelinePage from "./PipelinePage";
import { FloatingMLBackground, NeuralNetPageBackground } from "./MLDecorations";

export interface UserProfile {
  nom: string;
  prenom: string;
  email: string;
  isExpert: boolean;
}

export default function App() {
  const [page, setPage] = useState<"auth" | "pipeline" | "history">("auth");
  const [user, setUser] = useState<UserProfile | null>(null);

  function handleLogin(u: UserProfile) {
    setUser(u);
    setPage("pipeline");
    // Fire-and-forget login notification — never blocks the UI.
    fetch("/api/notify/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email:     u.email,
        prenom:    u.prenom,
        nom:       u.nom,
        is_expert: u.isExpert,
      }),
    }).catch(() => { /* silent — email is non-critical */ });
  }

  function handleLogout() {
    setUser(null);
    setPage("auth");
  }

  if (!user || page === "auth") {
    return <AuthPage onLogin={handleLogin} />;
  }

  return (
    <>
      <FloatingMLBackground />
      <NeuralNetPageBackground />
      <nav className="app-navbar">
        <div className="app-navbar-brand">
          <span className="app-navbar-logo">✦</span>
          <span className="app-navbar-title">ModelTrainer</span>
        </div>

        <div className="app-navbar-links">
          <button
            className={`app-nav-link ${page === "pipeline" ? "active" : ""}`}
            onClick={() => setPage("pipeline")}
          >
            ▶ Pipeline
          </button>
          <button
            className={`app-nav-link ${page === "history" ? "active" : ""}`}
            onClick={() => setPage("history")}
          >
            📚 Historique
          </button>
        </div>

        <div className="app-navbar-user">
          <span className="app-navbar-name">{user.prenom} {user.nom}</span>
          <span className="app-navbar-badge">{user.isExpert ? "Expert" : "Technicien"}</span>
          <button className="app-navbar-logout" onClick={handleLogout}>
            Changer de profil
          </button>
        </div>
      </nav>

      {page === "history"
        ? <HistoryPage user={user} onBack={() => setPage("pipeline")} />
        : <PipelinePage user={user} onHistory={() => setPage("history")} />
      }
    </>
  );
}
