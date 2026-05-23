import { useEffect, useRef, useState } from "react";
import { fetchRunHistory, reportDownloadUrl, modelDownloadUrl, loadChatModel, sendChatMessage, unloadChatModel, type RunSummary } from "./api";
import type { UserProfile } from "./App";

interface ChatMsg { role: "user" | "assistant"; content: string; }
type ChatLoadState = "idle" | "loading" | "ready" | "error";

function InlineChat({ runId }: { runId: string }) {
  const [loadState, setLoadState] = useState<ChatLoadState>("idle");
  const [messages,  setMessages]  = useState<ChatMsg[]>([]);
  const [input,     setInput]     = useState("");
  const [generating,setGenerating]= useState(false);
  const [error,     setError]     = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function onLoad() {
    setLoadState("loading"); setError("");
    try {
      await loadChatModel(runId);
      setLoadState("ready");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Erreur de chargement");
      setLoadState("error");
    }
  }

  async function onSend() {
    const msg = input.trim();
    if (!msg || generating) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setGenerating(true);
    try {
      const reply = await sendChatMessage(runId, msg);
      setMessages(prev => [...prev, { role: "assistant", content: reply }]);
    } catch {
      setMessages(prev => [...prev, { role: "assistant", content: "⚠ Erreur de génération." }]);
    } finally {
      setGenerating(false);
    }
  }

  async function onUnload() {
    await unloadChatModel(runId);
    setLoadState("idle");
    setMessages([]);
  }

  if (loadState === "idle" || loadState === "loading" || loadState === "error") {
    return (
      <div style={{ marginTop: 12, padding: "14px 16px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--border)" }}>
        {error && <div style={{ color: "var(--err)", fontSize: 12, marginBottom: 8 }}>⚠ {error}</div>}
        <button
          className="action-btn action-accent"
          style={{ fontSize: 13 }}
          onClick={onLoad}
          disabled={loadState === "loading"}
        >
          {loadState === "loading" ? "⏳ Chargement du modèle…" : "▶ Charger le modèle et démarrer le chat"}
        </button>
      </div>
    );
  }

  return (
    <div style={{ marginTop: 12, border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
      <div style={{ background: "var(--panel)", padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 700, fontSize: 14 }}>💬 Chat avec le modèle fine-tuné</span>
        <button className="action-btn action-secondary" style={{ fontSize: 11, padding: "4px 10px" }} onClick={onUnload}>
          ✕ Décharger
        </button>
      </div>
      <div style={{ height: 260, overflowY: "auto", padding: "12px 14px", background: "var(--surface)", display: "flex", flexDirection: "column", gap: 10 }}>
        {messages.length === 0 && (
          <div style={{ color: "var(--muted)", fontSize: 13, textAlign: "center", marginTop: 60 }}>
            Posez une question sur le contenu du document…
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === "user" ? "flex-end" : "flex-start",
            maxWidth: "80%", padding: "8px 12px", borderRadius: 10, fontSize: 13,
            background: m.role === "user" ? "var(--primary)" : "var(--panel)",
            color: m.role === "user" ? "#fff" : "var(--text)",
          }}>{m.content}</div>
        ))}
        {generating && <div style={{ color: "var(--muted)", fontSize: 12 }}>⏳ Génération…</div>}
        <div ref={bottomRef} />
      </div>
      <div style={{ display: "flex", gap: 8, padding: "10px 14px", borderTop: "1px solid var(--border)", background: "var(--panel)" }}>
        <input
          style={{ flex: 1, padding: "8px 12px", borderRadius: 8, border: "1.5px solid var(--border)", fontSize: 13, background: "var(--surface)", color: "var(--text)", fontFamily: "inherit" }}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && onSend()}
          placeholder="Votre question…"
          disabled={generating}
        />
        <button className="action-btn action-primary" style={{ fontSize: 13 }} onClick={onSend} disabled={generating || !input.trim()}>
          Envoyer
        </button>
      </div>
    </div>
  );
}

function statusInfo(status: string): { label: string; icon: string; cls: string } {
  if (status === "trained")             return { label: "Réussi",            icon: "✅", cls: "badge-done"    };
  if (status === "trained_low_quality") return { label: "Qualité limitée",   icon: "⚠️", cls: "badge-blocked" };
  if (status === "blocked")             return { label: "Bloqué",            icon: "⛔", cls: "badge-error"   };
  if (status === "running")             return { label: "En cours",          icon: "⏳", cls: "badge-running"  };
  if (status === "error")               return { label: "Erreur",            icon: "✕",  cls: "badge-error"   };
  return { label: status, icon: "•", cls: "badge-idle" };
}

function tierColor(tier?: string): string {
  if (tier === "good")       return "var(--ok)";
  if (tier === "acceptable") return "var(--warn)";
  return "var(--err)";
}

function formatDate(ts?: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleDateString("fr-FR", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

export default function HistoryPage({
  user, onBack,
}: {
  user: UserProfile;
  onBack: () => void;
}) {
  const [runs,    setRuns]    = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [search,  setSearch]  = useState("");

  useEffect(() => {
    fetchRunHistory()
      .then(setRuns)
      .finally(() => setLoading(false));
  }, []);

  const filtered = runs.filter(r => {
    if (!search.trim()) return true;
    const q = search.toLowerCase();
    return (
      r.job_id.toLowerCase().includes(q) ||
      (r.model_id ?? "").toLowerCase().includes(q) ||
      (r.task ?? "").toLowerCase().includes(q) ||
      (r.domain ?? "").toLowerCase().includes(q) ||
      (r.status ?? "").toLowerCase().includes(q)
    );
  });

  return (
    <div className="page">
      {/* Header */}
      <div className="history-header">
        <div>
          <h1 className="history-title">📚 Historique des modèles</h1>
          <p className="muted-text">
            Bonjour {user.prenom} — retrouvez ici tous vos fine-tunings réalisés.
          </p>
        </div>
        <button className="action-btn action-secondary" onClick={onBack}>
          ← Nouveau pipeline
        </button>
      </div>

      {/* Stats bar */}
      {!loading && runs.length > 0 && (
        <div className="history-stats">
          <div className="history-stat">
            <span className="history-stat-value">{runs.length}</span>
            <span className="history-stat-label">Total</span>
          </div>
          <div className="history-stat">
            <span className="history-stat-value" style={{ color: "var(--ok)" }}>
              {runs.filter(r => r.status === "trained").length}
            </span>
            <span className="history-stat-label">Réussis</span>
          </div>
          <div className="history-stat">
            <span className="history-stat-value" style={{ color: "var(--warn)" }}>
              {runs.filter(r => r.status === "trained_low_quality").length}
            </span>
            <span className="history-stat-label">Qualité limitée</span>
          </div>
          <div className="history-stat">
            <span className="history-stat-value" style={{ color: "var(--err)" }}>
              {runs.filter(r => r.status === "blocked" || r.status === "error").length}
            </span>
            <span className="history-stat-label">Échoués</span>
          </div>
        </div>
      )}

      {/* Search */}
      {runs.length > 3 && (
        <input
          type="text"
          placeholder="🔍 Rechercher par modèle, tâche, domaine, ID…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{
            width: "100%", padding: "10px 14px",
            borderRadius: 9, border: "1.5px solid var(--border)",
            fontSize: 14, background: "var(--surface)", color: "var(--text)",
            fontFamily: "inherit",
          }}
        />
      )}

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: "center", padding: "40px 0", color: "var(--muted)" }}>
          <div className="spinner" style={{ margin: "0 auto 12px" }} />
          Chargement de l'historique…
        </div>
      )}

      {/* Empty state */}
      {!loading && runs.length === 0 && (
        <div className="card" style={{ textAlign: "center", padding: "48px 24px" }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🗂</div>
          <h3 style={{ margin: "0 0 8px", color: "var(--primary)" }}>Aucun modèle réalisé</h3>
          <p className="muted-text">Lancez votre premier pipeline pour voir vos modèles ici.</p>
          <button className="action-btn action-primary" style={{ marginTop: 20 }} onClick={onBack}>
            ▶ Lancer un pipeline
          </button>
        </div>
      )}

      {/* No results */}
      {!loading && runs.length > 0 && filtered.length === 0 && (
        <div className="card" style={{ textAlign: "center", padding: "32px 24px" }}>
          <p className="muted-text">Aucun résultat pour « {search} ».</p>
        </div>
      )}

      {/* Run cards */}
      {!loading && filtered.map(run => {
        const st  = statusInfo(run.status);
        const metric = run.primary_metric?.toUpperCase().replace(/_/g, "-") ?? "—";
        const metricVal = run.primary_metric_value;
        const canChat = run.status === "trained" || run.status === "trained_low_quality";

        return (
          <div className="card history-run-card" key={run.job_id}>
            {/* Top row */}
            <div className="history-run-top">
              <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <span style={{ fontSize: 20 }}>{st.icon}</span>
                <span className="mono" style={{ fontSize: 13, color: "var(--muted)" }}>
                  {run.job_id}
                </span>
                <span className={`badge ${st.cls}`}>{st.label}</span>
                {run.quality_tier && (
                  <span className="badge" style={{
                    background: run.quality_tier === "good" ? "#DCFCE7" : run.quality_tier === "acceptable" ? "#FEF8E7" : "#FEE2E2",
                    color: tierColor(run.quality_tier),
                    borderColor: tierColor(run.quality_tier),
                  }}>
                    {run.quality_tier === "good" ? "Bonne qualité" : run.quality_tier === "acceptable" ? "Acceptable" : "Insuffisant"}
                  </span>
                )}
              </div>
              <span className="muted-text" style={{ fontSize: 12 }}>{formatDate(run.created_at)}</span>
            </div>

            {/* Info grid */}
            <div className="history-run-grid">
              {run.model_id && (
                <div className="history-run-item">
                  <span className="history-run-label">Modèle</span>
                  <span className="history-run-value mono">{run.model_id.split("/").pop()}</span>
                </div>
              )}
              {run.peft_method && (
                <div className="history-run-item">
                  <span className="history-run-label">Méthode</span>
                  <span className="history-run-value">{run.peft_method.toUpperCase()}</span>
                </div>
              )}
              {run.task && (
                <div className="history-run-item">
                  <span className="history-run-label">Tâche</span>
                  <span className="history-run-value">{run.task}</span>
                </div>
              )}
              {run.domain && (
                <div className="history-run-item">
                  <span className="history-run-label">Domaine</span>
                  <span className="history-run-value">{run.domain}</span>
                </div>
              )}
              {run.n_pairs != null && (
                <div className="history-run-item">
                  <span className="history-run-label">Paires QA</span>
                  <span className="history-run-value">{run.n_pairs}</span>
                </div>
              )}
              {metricVal != null && (
                <div className="history-run-item">
                  <span className="history-run-label">{metric}</span>
                  <span className="history-run-value" style={{ color: tierColor(run.quality_tier), fontFamily: "var(--mono)", fontWeight: 800 }}>
                    {metricVal.toFixed(3)}
                  </span>
                </div>
              )}
            </div>

            {/* Summary */}
            {run.summary && (
              <div style={{
                fontSize: 12.5, color: "var(--muted)", marginTop: 10,
                padding: "7px 11px", borderRadius: 6,
                background: "var(--panel)", borderLeft: "3px solid var(--border)",
              }}>
                {run.summary}
              </div>
            )}

            {/* Actions */}
            {canChat && (
              <div style={{ display: "flex", gap: 10, marginTop: 14, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
                <a href={modelDownloadUrl(run.job_id)} download>
                  <button className="action-btn action-primary" style={{ fontSize: 12 }}>
                    ⬇ Adaptateur LoRA
                  </button>
                </a>
                <a href={reportDownloadUrl(run.job_id)} download>
                  <button className="action-btn action-secondary" style={{ fontSize: 12 }}>
                    ⬇ Rapport JSON
                  </button>
                </a>
              </div>
            )}
            {canChat && <InlineChat runId={run.job_id} />}
          </div>
        );
      })}
    </div>
  );
}
