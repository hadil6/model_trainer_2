import { useState } from "react";
import type { UserProfile } from "./App";
import { NeuralNetDecoration, FloatingMLBackground, NeuralNetPageBackground } from "./MLDecorations";

export default function AuthPage({ onLogin }: { onLogin: (u: UserProfile) => void }) {
  const [prenom, setPrenom]       = useState("");
  const [nom,    setNom]          = useState("");
  const [isExpert, setIsExpert]   = useState<boolean | null>(null);
  const [error,  setError]        = useState("");

  function handleSubmit() {
    if (!prenom.trim()) { setError("Veuillez entrer votre prénom."); return; }
    if (!nom.trim())    { setError("Veuillez entrer votre nom."); return; }
    if (isExpert === null) { setError("Veuillez sélectionner votre profil."); return; }
    setError("");
    onLogin({ prenom: prenom.trim(), nom: nom.trim(), isExpert });
  }

  return (
    <div className="auth-page">
      <FloatingMLBackground />
      <NeuralNetPageBackground />
      {/* Gradient banner */}
      <div className="auth-banner">
        <div className="hero-noise" />
        <NeuralNetDecoration />
        <div className="auth-banner-inner">
          <div className="hero-eyebrow">AI-Powered Model Trainer · DATA&amp;AI-2025</div>
          <h1 className="hero-title" style={{ fontSize: 32 }}>
            Vos données ont du potentiel.{" "}
            <span className="hero-title-accent">Exploitez-le.</span>
          </h1>
          <p className="hero-subtitle" style={{ fontSize: 14 }}>
            Du brut au déployable — fine-tuning IA entièrement automatisé, sans expertise technique requise.
          </p>
        </div>
      </div>

      {/* Form card */}
      <div className="auth-card">
        <h2 className="auth-title">Créer votre profil</h2>
        <p className="auth-subtitle">Ces informations permettent de personnaliser votre expérience.</p>

        <div className="auth-fields">
          <label className="field">
            <span>Prénom</span>
            <input
              type="text"
              placeholder="Ex. : Hadil"
              value={prenom}
              onChange={e => setPrenom(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSubmit()}
            />
          </label>

          <label className="field">
            <span>Nom</span>
            <input
              type="text"
              placeholder="Ex. : Souilem"
              value={nom}
              onChange={e => setNom(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSubmit()}
            />
          </label>
        </div>

        {/* Profile selector */}
        <div className="auth-profile-title">Quel est votre profil ?</div>
        <div className="auth-profiles">
          {/* Expert */}
          <div
            className={`auth-profile-card ${isExpert === true ? "selected" : ""}`}
            onClick={() => setIsExpert(true)}
          >
            <div className="auth-profile-icon">🎓</div>
            <div className="auth-profile-name">Expert</div>
            <div className="auth-profile-desc">
              Je connais le machine learning, les métriques ROUGE/BLEU et le fine-tuning.
              Je veux l'interface complète sans explications supplémentaires.
            </div>
            <div className="auth-profile-features">
              <span>✓ Interface complète</span>
              <span>✓ Métriques techniques</span>
              <span>✓ Logs détaillés</span>
            </div>
          </div>

          {/* Non-expert */}
          <div
            className={`auth-profile-card ${isExpert === false ? "selected" : ""}`}
            onClick={() => setIsExpert(false)}
          >
            <div className="auth-profile-icon">🌱</div>
            <div className="auth-profile-name">Technicien</div>
            <div className="auth-profile-desc">
              Je découvre le fine-tuning et j'ai besoin d'explications à chaque étape
              pour comprendre ce que fait le pipeline.
            </div>
            <div className="auth-profile-features">
              <span>✓ Guides étape par étape</span>
              <span>✓ Explications simplifiées</span>
              <span>✓ Conseils contextuels</span>
            </div>
          </div>
        </div>

        {error && (
          <div className="error-box" style={{ marginBottom: 0 }}>⚠ {error}</div>
        )}

        <button
          className="run-btn"
          style={{ width: "100%", marginTop: 20, fontSize: 16 }}
          onClick={handleSubmit}
          disabled={!prenom || !nom || isExpert === null}
        >
          Commencer →
        </button>
      </div>
    </div>
  );
}
