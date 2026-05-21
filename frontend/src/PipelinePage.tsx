import { useEffect, useRef, useState } from "react";
import { NeuralNetDecoration } from "./MLDecorations";
import {
  startRun, subscribeToRun,
  fetchDataset, datasetDownloadUrl,
  fetchTrainingReport, fetchEvalReport, triggerEval,
  fetchLossHistory, fetchTrainingImages,
  fetchRunResult, confirmAction, uploadAdditionalFiles,
  fetchGpuInfo,
  loadChatModel, sendChatMessage, unloadChatModel,
  reportDownloadUrl, modelDownloadUrl,
  type RunEvent, type DatasetResponse,
  type TrainingReport, type EvalReport,
  type LossHistory, type TrainingImage,
  type ConfirmPayload,
  type DecisionJournalEntry,
  type GpuInfo,
} from "./api";
import type { UserProfile } from "./App";

const LS_RUN_KEY = "active_run_id";

type Status    = "idle" | "running" | "done" | "error" | "blocked";
type EvalState = "idle" | "loading" | "done" | "error";

// ── Constants ─────────────────────────────────────────────────────────────────

const PEFT_LABELS: Record<string, { label: string; desc: string }> = {
  qlora:     { label: "QLoRA",     desc: "Quantifié 4-bit — économe en VRAM" },
  lora:      { label: "LoRA",      desc: "Léger bf16 — meilleure précision" },
  lora_int8: { label: "LoRA int8", desc: "Semi-quantifié 8-bit" },
  dora:      { label: "DoRA",      desc: "LoRA avec décomposition améliorée" },
  full:      { label: "Full",      desc: "Entraînement complet — VRAM élevée" },
};

const HPARAM_INFO: Record<string, { label: string; desc: string }> = {
  lora_rank:                   { label: "LoRA Rank",    desc: "Dimension de l'espace d'adaptation" },
  lora_alpha:                  { label: "LoRA Alpha",   desc: "Facteur de mise à l'échelle" },
  lora_dropout:                { label: "Dropout",      desc: "Taux de désactivation pour éviter le surapprentissage" },
  learning_rate:               { label: "Learning Rate",desc: "Vitesse d'adaptation — trop élevé = instable" },
  num_train_epochs:            { label: "Epochs",       desc: "Passages complets sur le dataset" },
  per_device_train_batch_size: { label: "Batch Size",   desc: "Exemples traités simultanément" },
  gradient_accumulation_steps: { label: "Grad. Accum.", desc: "Simule un batch plus grand" },
  warmup_ratio:                { label: "Warmup",       desc: "Proportion de montée en régime" },
  weight_decay:                { label: "Weight Decay", desc: "Régularisation L2" },
};

const TASKS: { id: string; icon: string; label: string; desc: string }[] = [
  { id: "question-answering", icon: "🔍", label: "Question & Réponse", desc: "Répondre à des questions depuis vos documents" },
  { id: "code-generation",    icon: "💻", label: "Génération de code", desc: "Produire ou compléter du code source" },
  { id: "summarization",      icon: "📝", label: "Résumé de texte",    desc: "Condenser des documents longs en résumés clairs" },
  { id: "chatbot",            icon: "🤖", label: "Chatbot",            desc: "Assistant conversationnel polyvalent" },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function tierColor(tier?: string): string {
  if (tier === "good")       return "var(--ok)";
  if (tier === "acceptable") return "var(--warn)";
  return "var(--err)";
}

function classifyLog(line: string): string {
  const l = line.toLowerCase();
  if (l.includes("error") || l.includes("failed") || l.includes("exception")) return "log-line-error";
  if (l.includes("warn"))  return "log-line-warn";
  if (l.includes("done") || l.includes("finish") || l.includes("success") || l.includes("✓")) return "log-line-success";
  return "log-line-info";
}

// ── Pipeline step detection ───────────────────────────────────────────────────

function stepIcon(msg: string): string {
  const m = msg.toLowerCase();
  if (m.includes("profil") || (m.includes("fichier") && m.includes("analysé"))) return "📂";
  if (m.includes("compatib") || (m.includes("domaine") && m.includes("vérif"))) return "🔍";
  if (m.includes("stratégie d'évaluation") || m.includes("métrique principal")) return "📊";
  if (m.includes("intention") || m.includes("tâche :")) return "🎯";
  if (m.includes("faisabilité")) return "✅";
  if (m.includes("nouveau modèle sélectionné")) return "🔄";
  if (m.includes("sélectionné") && m.includes("modèle")) return "🤖";
  if (m.includes("données prêtes") || m.includes("paires qa générées")) return "📚";
  if (m.includes("diagnostic itération")) return "🔬";
  if (m.includes("dataset insuffisant") || m.includes("générer automatiquement")) return "⚠️";
  if (m.includes("entraînem") && (m.includes("lancement") || m.includes("terminé") || m.includes("démarré"))) return "⚙️";
  if (m.includes("évaluation itération") || m.includes("évaluation terminée")) return "📈";
  if (m.includes("incompatibil") || m.includes("bloqué")) return "⛔";
  return "▸";
}

function stepTitle(msg: string): string {
  const m = msg.toLowerCase();
  if (m.includes("profil") && (m.includes("terminé") || m.includes("analysé"))) return "Analyse des fichiers";
  if (m.includes("compatib")) return "Compatibilité des domaines";
  if (m.includes("stratégie d'évaluation")) return "Stratégie d'évaluation détectée";
  if (m.includes("intention") || (m.includes("tâche") && m.includes(":"))) return "Extraction de l'intention";
  if (m.includes("faisabilité")) return "Vérification de faisabilité";
  if (m.includes("nouveau modèle sélectionné")) return "Resélection du modèle";
  if (m.includes("sélectionné") && m.includes("modèle")) return "Sélection du modèle";
  if (m.includes("données prêtes") || m.includes("paires qa")) return "Préparation des données";
  if (m.includes("diagnostic itération")) return "Diagnostic & action corrective";
  if (m.includes("dataset insuffisant") || m.includes("générer automatiquement")) return "Augmentation du dataset";
  if (m.includes("entraînem") && m.includes("lancement")) return "Lancement de l'entraînement";
  if (m.includes("entraînem") && m.includes("terminé")) return "Entraînement terminé";
  if (m.includes("entraînem")) return "Entraînement";
  if (m.includes("évaluation itération")) return "Évaluation du modèle";
  if (m.includes("incompatibil") || m.includes("bloqué")) return "Blocage — domaines incompatibles";
  return "Étape pipeline";
}

// ── Non-expert explanations per step ─────────────────────────────────────────

const STEP_GUIDE: Record<string, string> = {
  "Analyse des fichiers":          "Le pipeline lit vos documents pour comprendre leur format (PDF, CSV, texte), leur langue et leur taille.",
  "Compatibilité des domaines":    "On vérifie que tous vos fichiers parlent du même sujet (ex. tous médicaux). Des fichiers de domaines différents produiraient un modèle incohérent.",
  "Extraction de l'intention":     "L'IA analyse votre objectif en langage naturel pour déterminer le type de tâche : questions-réponses, classification, extraction d'entités…",
  "Stratégie d'évaluation détectée": "En fonction de votre tâche, le pipeline choisit les bons critères de mesure. Une classification n'est pas évaluée comme une génération de texte.",
  "Vérification de faisabilité":   "On s'assure que vos données contiennent assez de texte pour entraîner un modèle. Un dataset trop petit donnerait un modèle inutilisable.",
  "Sélection du modèle":           "L'IA choisit le meilleur modèle pré-entraîné selon votre GPU, votre tâche et votre objectif (vitesse vs qualité).",
  "Resélection du modèle":         "Le modèle précédent n'a pas donné de bons résultats. L'orchestrateur en essaie un autre mieux adapté.",
  "Préparation des données":       "Vos documents sont transformés en paires question-réponse que le modèle va apprendre. C'est l'équivalent de créer un manuel d'exercices.",
  "Augmentation du dataset":       "Pas assez d'exemples d'entraînement. Le pipeline génère automatiquement des questions supplémentaires à partir de vos documents.",
  "Lancement de l'entraînement":   "Le modèle apprend à partir de vos exemples. C'est l'étape la plus longue — le modèle ajuste ses paramètres pour répondre comme vos documents.",
  "Entraînement terminé":          "Le modèle a fini d'apprendre. Un adaptateur (fichier léger) a été créé — il encode tout ce que le modèle a appris de vos données.",
  "Entraînement":                  "Entraînement en cours — le modèle s'adapte à vos données.",
  "Évaluation du modèle":          "On mesure la qualité du modèle sur des exemples qu'il n'a jamais vus pendant l'entraînement. Plus le score est proche de 1, meilleur est le modèle.",
  "Diagnostic & action corrective":"Le score n'est pas assez bon. L'orchestrateur identifie la cause (trop peu de données ? mauvais hyperparamètres ?) et prend une action corrective.",
};

// ── SVG Loss Chart ─────────────────────────────────────────────────────────────

function LossChart({ history }: { history: LossHistory }) {
  const W = 900, H = 180, PADL = 52, PADB = 30, PADR = 20, PADT = 16;
  const points = history.points;
  if (points.length < 2) return <p className="muted-text">Pas assez de points pour tracer la courbe.</p>;

  const steps   = points.map(p => p.step);
  const minStep = Math.min(...steps), maxStep = Math.max(...steps);
  const trainPts = points.filter(p => p.train_loss != null);
  const evalPts  = points.filter(p => p.eval_loss  != null);
  const allLoss  = [...trainPts.map(p => p.train_loss!), ...evalPts.map(p => p.eval_loss!)];
  const minLoss  = Math.min(...allLoss) * 0.9;
  const maxLoss  = Math.max(...allLoss) * 1.05;

  const cx = (s: number) => PADL + ((s - minStep) / (maxStep - minStep || 1)) * (W - PADL - PADR);
  const cy = (l: number) => PADT + (1 - (l - minLoss) / (maxLoss - minLoss || 1)) * (H - PADT - PADB);
  const poly = (pts: typeof trainPts, key: "train_loss" | "eval_loss") =>
    pts.map(p => `${cx(p.step).toFixed(1)},${cy(p[key]!).toFixed(1)}`).join(" ");
  const yTicks = Array.from({ length: 5 }, (_, i) => minLoss + (i / 4) * (maxLoss - minLoss));

  return (
    <svg className="loss-chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" style={{ height: 200 }}>
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={PADL} y1={cy(v)} x2={W - PADR} y2={cy(v)} stroke="#D0D9E8" strokeWidth="1" />
          <text x={PADL - 6} y={cy(v) + 4} textAnchor="end" fontSize="10" fill="#617498">{v.toFixed(2)}</text>
        </g>
      ))}
      {trainPts.length >= 2 && <polyline points={poly(trainPts, "train_loss")} fill="none" stroke="#D9467A" strokeWidth="2" strokeLinejoin="round" />}
      {evalPts.length  >= 2 && <polyline points={poly(evalPts,  "eval_loss")}  fill="none" stroke="#5B7FAE" strokeWidth="2.5" strokeLinejoin="round" />}
      {evalPts.map((p, i) => <circle key={i} cx={cx(p.step)} cy={cy(p.eval_loss!)} r="3.5" fill="#5B7FAE" />)}
      <line x1={PADL} y1={H - PADB} x2={W - PADR} y2={H - PADB} stroke="#D0D9E8" strokeWidth="1.5" />
      <text x={(W - PADL - PADR) / 2 + PADL} y={H - 4} textAnchor="middle" fontSize="10" fill="#617498">Steps</text>
    </svg>
  );
}

// ── Training Images ───────────────────────────────────────────────────────────

function TrainingImagesSection({ images }: { images: TrainingImage[] }) {
  const [lightbox, setLightbox] = useState<string | null>(null);
  if (!images.length) return null;
  return (
    <div className="loss-chart-wrap" style={{ marginBottom: 0 }}>
      <div className="loss-chart-title">📸 Graphiques d'entraînement</div>
      <div className="images-grid">
        {images.map(img => (
          <div className="image-card" key={img.name}>
            <img src={img.url} alt={img.name} onClick={() => setLightbox(img.url)} loading="lazy" />
            <div className="image-card-label">{img.name.replace(/\.(png|jpg|jpeg)$/i, "")}</div>
          </div>
        ))}
      </div>
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="zoom" />
        </div>
      )}
    </div>
  );
}

// ── Improvement Log ───────────────────────────────────────────────────────────

function ImprovementLog({ log }: { log: Array<Record<string, unknown>> }) {
  if (!log.length) return null;
  return (
    <div>
      <div className="loss-chart-title" style={{ marginBottom: 10 }}>🔄 Historique des itérations</div>
      <div className="improvement-log">
        {log.map((entry, i) => {
          const tier       = String(entry.quality_tier ?? "poor");
          const primary    = String(entry.primary_metric ?? "rouge1").toUpperCase().replace(/_/g, "-");
          const primaryVal = Number(entry.primary_metric_value ?? entry.rouge1 ?? 0);
          const prevEntry  = i > 0 ? log[i - 1] : null;
          const prevVal    = prevEntry ? Number(prevEntry.primary_metric_value ?? prevEntry.rouge1 ?? 0) : null;
          const delta      = prevVal !== null ? primaryVal - prevVal : null;
          const overrides  = entry.hparam_overrides as Record<string, unknown> | null;
          const modelShort = String(entry.model_id ?? "—").split("/").pop() ?? "—";

          return (
            <div key={i} className="iter-row" style={{ flexDirection: "column", alignItems: "stretch", gap: 10 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div className="iter-num">{Number(entry.iteration ?? i) + 1}</div>
                <div className="iter-metrics">
                  <div className="iter-metric">
                    <span className="iter-metric-label">{primary}</span>
                    <span className={`iter-metric-value iter-tier-${tier}`} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      {primaryVal.toFixed(3)}
                      {delta !== null && (
                        <span style={{ fontSize: 11, fontWeight: 700, color: delta > 0.001 ? "var(--ok)" : delta < -0.001 ? "var(--err)" : "var(--muted)" }}>
                          {delta > 0.001 ? `↑+${delta.toFixed(3)}` : delta < -0.001 ? `↓${delta.toFixed(3)}` : "→"}
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="iter-metric">
                    <span className="iter-metric-label">Paires</span>
                    <span className="iter-metric-value" style={{ color: "var(--muted)", fontSize: 13 }}>{String(entry.n_pairs ?? "—")}</span>
                  </div>
                  <div className="iter-metric">
                    <span className="iter-metric-label">Modèle</span>
                    <span className="iter-metric-value" style={{ color: "var(--muted)", fontSize: 11, maxWidth: 130, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{modelShort}</span>
                  </div>
                </div>
                <div style={{ marginLeft: "auto" }}>
                  <span className="badge" style={{
                    background: tier === "good" ? "#E6F7F1" : tier === "acceptable" ? "#FEF8E7" : "#FDECEA",
                    color: tierColor(tier), borderColor: tierColor(tier),
                  }}>
                    {tier === "good" ? "Bon" : tier === "acceptable" ? "Acceptable" : "Insuffisant"}
                  </span>
                </div>
              </div>
              {overrides && Object.keys(overrides).filter(k => k !== "_source").length > 0 && (
                <div style={{
                  marginLeft: 40, padding: "7px 12px", borderRadius: 7,
                  background: "#EDE9FE", border: "1px solid #C4B5FD",
                  fontSize: 12, color: "#5B21B6", display: "flex", flexWrap: "wrap", gap: "4px 14px", alignItems: "center",
                }}>
                  <span style={{ fontWeight: 700, marginRight: 2 }}>🔧 Hyperparamètres ajustés :</span>
                  {Object.entries(overrides).filter(([k]) => k !== "_source").map(([k, v]) => (
                    <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <span style={{ fontWeight: 600 }}>{HPARAM_INFO[k]?.label ?? k}</span>
                      <span style={{ color: "var(--primary)" }}>→</span>
                      <code style={{ background: "#fff", padding: "1px 5px", borderRadius: 4, fontSize: 11, color: "var(--primary)" }}>{String(v)}</code>
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Metric Row ────────────────────────────────────────────────────────────────

function MetricRow({ label, desc, value, metric, thresholdGood, thresholdAcceptable, noviceHint }: {
  label: string; desc: string; value: number; metric: string;
  thresholdGood?: number; thresholdAcceptable?: number; noviceHint?: string;
}) {
  const goodFallback = metric === "bleu4" ? 0.14 : metric === "rouge2" ? 0.22 : metric === "rougeL" ? 0.42 : 0.45;
  const okFallback   = metric === "bleu4" ? 0.05 : metric === "rouge2" ? 0.12 : metric === "rougeL" ? 0.25 : 0.28;
  const good = thresholdGood       ?? goodFallback;
  const ok   = thresholdAcceptable ?? okFallback;
  const max  = metric === "bleu4" ? 0.35 : 1.0;
  const color = value >= good ? "var(--ok)" : value >= ok ? "var(--warn)" : "var(--err)";
  const fillW = `${Math.min(100, (value / max) * 100).toFixed(1)}%`;
  const goodW = `${Math.min(100, (good / max) * 100).toFixed(1)}%`;

  return (
    <div className="metric-row">
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        <div>
          <span className="metric-value" style={{ color }}>{value.toFixed(3)}</span>
          {thresholdGood && <span className="metric-threshold">seuil : {thresholdGood.toFixed(2)}</span>}
        </div>
      </div>
      <div className="metric-bar-bg" style={{ position: "relative" }}>
        <div className="metric-bar-fill" style={{ width: fillW, background: color }} />
        <div className="metric-threshold-line" style={{ left: goodW }} title={`Seuil "bon" = ${good.toFixed(2)}`} />
      </div>
      <p className="metric-desc">{desc}</p>
      {noviceHint && (
        <div style={{ marginTop: 6, fontSize: 12, color: "var(--primary)", background: "#EDE9FE", borderRadius: 6, padding: "5px 9px" }}>
          🎓 {noviceHint}
        </div>
      )}
    </div>
  );
}

// ── InfoItem ──────────────────────────────────────────────────────────────────

function InfoItem({ icon, label, value, sub, mono }: {
  icon: string; label: string; value: string; sub?: string; mono?: boolean;
}) {
  return (
    <div className="info-item">
      <div className="info-icon-wrap">{icon}</div>
      <div>
        <div className="info-label">{label}</div>
        <div className={`info-value ${mono ? "mono" : ""}`}>{value}</div>
        {sub && <div className="info-sub">{sub}</div>}
      </div>
    </div>
  );
}

// ── Training Dashboard ────────────────────────────────────────────────────────

function TrainingDashboard({
  report, runId, lossHistory, trainingImages, isExpert,
}: {
  report: TrainingReport; runId: string;
  lossHistory: LossHistory | null; trainingImages: TrainingImage[]; isExpert: boolean;
}) {
  const [hparamsOpen, setHparamsOpen] = useState(false);
  const peft      = PEFT_LABELS[report.peft_method] ?? { label: report.peft_method, desc: "" };
  const success   = report.result?.success;
  const evalLoss  = report.result?.metrics?.eval_loss;
  const trainLoss = report.result?.metrics?.loss;

  return (
    <div className="card">
      <div className="card-header">
        <h2><span className="card-icon">🤖</span>Entraînement du modèle</h2>
        <span className={`badge ${success ? "badge-done" : "badge-error"}`}>{success ? "Réussi" : "Échoué"}</span>
      </div>

      {!isExpert && (
        <div className="novice-guide-box">
          <strong>Qu'est-ce qui s'est passé ?</strong> Le modèle a appris de vos données grâce à la méthode PEFT.
          Au lieu de réentraîner tout le modèle (très coûteux), on ajoute un petit adaptateur qui encode
          les nouvelles connaissances. C'est ce fichier que vous pouvez télécharger.
        </div>
      )}

      <div className="info-grid">
        <InfoItem icon="🤖" label="Modèle"   value={report.model_id} mono />
        <InfoItem icon="⚙️" label="Méthode"  value={peft.label} sub={peft.desc} />
        <InfoItem icon="📚" label="Paires"   value={String(report.n_pairs)} sub={!isExpert ? "exemples d'entraînement utilisés" : undefined} />
        <InfoItem icon="🔁" label="Epochs"   value={String(report.hparams?.num_train_epochs ?? "—")} sub={!isExpert ? "passages sur le dataset" : undefined} />
        {evalLoss  != null && <InfoItem icon="📉" label="Eval Loss"  value={evalLoss.toFixed(4)}  sub="Plus bas = meilleur apprentissage" />}
        {trainLoss != null && <InfoItem icon="📊" label="Train Loss" value={trainLoss.toFixed(4)} />}
      </div>

      {report.peft_rationale && (
        <div className="rationale">
          <span className="rationale-icon">💡</span>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 3, fontSize: 12 }}>Pourquoi cette méthode PEFT ?</div>
            {report.peft_rationale}
          </div>
        </div>
      )}

      {report.hparam_rationale && (
        <div className="rationale" style={{ borderLeftColor: "var(--ok)", background: "#F0FDF4", border: "1px solid #BBF7D0" }}>
          <span className="rationale-icon">🎛️</span>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 3, fontSize: 12, color: "var(--ok)" }}>
              Pourquoi ces hyperparamètres ?
              {report.hparam_source && (
                <span style={{ marginLeft: 8, fontSize: 10, fontWeight: 400, color: "var(--muted)" }}>source : {report.hparam_source}</span>
              )}
            </div>
            <span style={{ color: "#166534" }}>{report.hparam_rationale}</span>
          </div>
        </div>
      )}

      {lossHistory && lossHistory.points.length >= 2 && (
        <div className="loss-chart-wrap">
          <div className="loss-chart-title">
            📈 Courbes de loss
            {!isExpert && (
              <span style={{ fontWeight: 400, fontSize: 11, color: "var(--muted)", marginLeft: 8 }}>
                — une courbe qui descend = le modèle apprend bien
              </span>
            )}
            <div className="loss-chart-legend">
              <div className="legend-item"><div className="legend-dot" style={{ background: "#D9467A" }} />Train loss</div>
              <div className="legend-item"><div className="legend-dot" style={{ background: "#5B7FAE" }} />Eval loss</div>
            </div>
          </div>
          <LossChart history={lossHistory} />
        </div>
      )}

      <TrainingImagesSection images={trainingImages} />

      <button className="toggle-btn" onClick={() => setHparamsOpen(o => !o)}>
        {hparamsOpen ? "▲ Masquer les hyperparamètres" : "▼ Voir les hyperparamètres utilisés"}
      </button>
      {hparamsOpen && (
        <>
          {!isExpert && (
            <div className="novice-guide-box" style={{ marginBottom: 12 }}>
              Les hyperparamètres sont les réglages de l'entraînement : à quelle vitesse le modèle apprend,
              combien de fois il voit les données, etc. Ils ont été choisis automatiquement selon votre GPU et votre dataset.
            </div>
          )}
          <table className="hparam-table">
            <thead><tr><th>Paramètre</th><th>Valeur</th><th>Description</th></tr></thead>
            <tbody>
              {Object.entries(report.hparams).map(([k, v]) => {
                const info = HPARAM_INFO[k] ?? { label: k, desc: "" };
                return (
                  <tr key={k}>
                    <td className="mono">{info.label}</td>
                    <td className="mono val">{String(v)}</td>
                    <td className="muted">{info.desc}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      <div className="card-actions">
        <a href={modelDownloadUrl(runId)} download>
          <button className="action-btn action-primary">⬇ Télécharger l'adaptateur</button>
        </a>
        <a href={reportDownloadUrl(runId)} download>
          <button className="action-btn action-secondary">⬇ Rapport JSON</button>
        </a>
      </div>
    </div>
  );
}

// ── Evaluation Card ───────────────────────────────────────────────────────────

function EvalCard({ evalState, evalReport, onRunEval, isExpert }: {
  evalState: EvalState; evalReport: EvalReport | null; onRunEval: () => void; isExpert: boolean;
}) {
  const metrics = evalReport?.metrics;
  const tier    = metrics?.quality_tier as string | undefined;
  const qualityBadge = (() => {
    if (!metrics) return null;
    if (tier === "good")       return { label: "Bonne qualité",      color: "var(--ok)" };
    if (tier === "acceptable") return { label: "Qualité acceptable", color: "var(--warn)" };
    return { label: "Qualité insuffisante", color: "var(--err)" };
  })();

  return (
    <div className="card">
      <div className="card-header">
        <h2><span className="card-icon">📊</span>Évaluation du modèle</h2>
        {qualityBadge && (
          <span className="quality-badge" style={{ color: qualityBadge.color, borderColor: qualityBadge.color }}>
            {qualityBadge.label}
          </span>
        )}
      </div>

      {evalState === "idle" && (
        <div className="eval-prompt">
          {!isExpert ? (
            <div className="novice-guide-box">
              <strong>Qu'est-ce que l'évaluation ?</strong> On donne au modèle des questions qu'il n'a jamais vues
              pendant l'entraînement. On compare ses réponses aux réponses correctes pour mesurer sa qualité.
              Un score de 1.0 = réponses parfaites. En pratique, 0.4+ est un bon résultat.
            </div>
          ) : (
            <p>L'évaluation mesure la qualité des réponses générées par le modèle fine-tuné
              par rapport aux réponses de référence (ROUGE, BLEU, LLM-as-judge).</p>
          )}
          <button className="action-btn action-accent" onClick={onRunEval}>▶ Lancer l'évaluation</button>
        </div>
      )}

      {evalState === "loading" && (
        <div className="eval-loading">
          <div className="spinner" />
          <p>Évaluation en cours — génération des réponses sur le jeu de test…<br />
            <span className="muted-text">Cela peut prendre plusieurs minutes.</span></p>
        </div>
      )}

      {evalState === "error" && (
        <div className="error-box">⚠ L'évaluation a échoué. Consultez les logs du serveur.</div>
      )}

      {metrics && (
        <>
          <div className="quality-summary">{metrics.quality_summary}</div>
          <div className="metrics-list">
            <MetricRow label="ROUGE-1" metric="rouge1" value={metrics.rouge1}
              thresholdGood={metrics.threshold_rouge1_good}
              thresholdAcceptable={metrics.threshold_rouge1_acceptable}
              desc="Proportion des mots de la référence retrouvés dans la réponse du modèle"
              noviceHint={!isExpert ? `Score actuel : ${metrics.rouge1.toFixed(2)} — ${metrics.rouge1 >= (metrics.threshold_rouge1_good ?? 0.45) ? "✅ Bon résultat !" : metrics.rouge1 >= (metrics.threshold_rouge1_acceptable ?? 0.28) ? "⚠ Résultat acceptable, peut être amélioré" : "❌ Insuffisant, le modèle a besoin de plus de données ou d'entraînement"}` : undefined}
            />
            <MetricRow label="ROUGE-2" metric="rouge2" value={metrics.rouge2}
              thresholdGood={metrics.threshold_rouge2_good}
              desc="Proportion des bigrammes (paires de mots consécutifs) en commun avec la référence" />
            <MetricRow label="ROUGE-L" metric="rougeL" value={metrics.rougeL}
              thresholdGood={metrics.threshold_rougeL_good}
              desc="Qualité structurelle — les mots apparaissent-ils dans le bon ordre ?" />
            <MetricRow label="BLEU-4" metric="bleu4" value={metrics.bleu4}
              thresholdGood={metrics.threshold_bleu4_good}
              desc="Précision des séquences de 4 mots — mesure stricte, souvent bas même pour un bon modèle"
              noviceHint={!isExpert ? "Ce score est naturellement bas. Un BLEU-4 > 0.10 est déjà un bon signe." : undefined}
            />
          </div>

          {metrics.llm_judge_accuracy != null && (
            <div className="llm-judge">
              <h3>Évaluation par LLM-as-judge ({metrics.llm_judge_n} exemples)</h3>
              <div className="judge-scores">
                <div className="judge-score">
                  <span className="judge-label">Précision</span>
                  <span className="judge-value">{metrics.llm_judge_accuracy.toFixed(1)}<span style={{ fontSize: 14, color: "var(--muted)" }}>/5</span></span>
                </div>
                <div className="judge-score">
                  <span className="judge-label">Complétude</span>
                  <span className="judge-value">{metrics.llm_judge_completeness?.toFixed(1)}<span style={{ fontSize: 14, color: "var(--muted)" }}>/5</span></span>
                </div>
              </div>
              {!isExpert && (
                <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6 }}>
                  Un autre modèle IA a noté les réponses de votre modèle sur 5. C'est une évaluation plus proche de ce que ressentirait un humain.
                </p>
              )}
            </div>
          )}
          <div className="eval-meta muted-text">Évalué sur {metrics.n_test} exemples du jeu de test</div>
        </>
      )}
    </div>
  );
}

// ── Live Pipeline Feed ────────────────────────────────────────────────────────

function LivePipelineFeed({ logs, status, isExpert }: { logs: string[]; status: Status; isExpert: boolean }) {
  const pipelineLogs = logs.filter(l =>
    l.startsWith("[PIPELINE]") || /^\d{2}:\d{2}:\d{2}\s+(ERROR|CRITICAL)\b/.test(l)
  );

  if (pipelineLogs.length === 0) {
    return (
      <div style={{ padding: "24px 0", color: "var(--muted)", textAlign: "center", fontSize: 14 }}>
        {status === "running" ? (
          <>
            <div className="spinner" style={{ width: 18, height: 18, border: "2.5px solid rgba(124,58,237,0.2)", borderTopColor: "var(--primary)", display: "inline-block", marginRight: 10, verticalAlign: "middle" }} />
            Démarrage du pipeline…
          </>
        ) : "Aucune étape pipeline pour l'instant."}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
      {pipelineLogs.map((line, i) => {
        const isError   = /^\d{2}:\d{2}:\d{2}\s+(ERROR|CRITICAL)\b/.test(line);
        const msg       = line.startsWith("[PIPELINE]") ? line.replace(/^\[PIPELINE\]\s*/, "") : line;
        const icon      = isError ? "✕" : stepIcon(msg);
        const title     = isError ? "Erreur" : stepTitle(msg);
        const guide     = !isExpert ? STEP_GUIDE[title] : undefined;
        const isLast    = i === pipelineLogs.length - 1;
        const isRunning = isLast && status === "running";

        return (
          <div key={i} style={{
            display: "flex", gap: 12, alignItems: "flex-start", padding: "11px 14px", borderRadius: 10,
            background: isError ? "#FFF5F5" : isRunning ? "#F5F0FF" : "var(--panel)",
            border: `1px solid ${isError ? "#FFCCCC" : isRunning ? "var(--primary)" : "var(--border)"}`,
            boxShadow: isRunning ? "0 0 0 3px rgba(124,58,237,0.08)" : "none",
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: "50%", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16,
              background: isError ? "#FDECEA" : isRunning ? "#EDE9FE" : "var(--surface)",
              border: `1.5px solid ${isError ? "#FCA5A5" : isRunning ? "var(--primary)" : "var(--border)"}`,
            }}>
              {isRunning
                ? <div className="spinner" style={{ width: 14, height: 14, border: "2px solid rgba(124,58,237,0.2)", borderTopColor: "var(--primary)" }} />
                : <span>{icon}</span>}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 3, color: isError ? "var(--err)" : "var(--text)", display: "flex", alignItems: "center", gap: 8 }}>
                {title}
                {isRunning && <span style={{ fontSize: 11, color: "var(--primary)", fontWeight: 500, fontStyle: "italic" }}>en cours…</span>}
                {!isRunning && !isError && <span style={{ fontSize: 11, color: "var(--ok)", fontWeight: 600 }}>✓</span>}
              </div>
              <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6, wordBreak: "break-word" }}>{msg}</div>
              {guide && (
                <div style={{ marginTop: 7, fontSize: 12, color: "var(--primary)", background: "#EDE9FE", borderRadius: 6, padding: "5px 10px", lineHeight: 1.55 }}>
                  🎓 <strong>Pour comprendre :</strong> {guide}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Decision Journal Card ─────────────────────────────────────────────────────

function DecisionJournalCard({ journal, isExpert }: { journal: DecisionJournalEntry[]; isExpert: boolean }) {
  const [open, setOpen] = useState(true);
  const okCount   = journal.filter(e => e.statut === "ok").length;
  const warnCount = journal.filter(e => e.statut === "warn").length;
  const errCount  = journal.filter(e => e.statut === "err").length;
  const infoCount = journal.filter(e => e.statut === "info").length;

  const statusColor = (s: string) =>
    s === "ok" ? "var(--ok)" : s === "warn" ? "var(--warn)" : s === "err" ? "var(--err)" : "var(--primary)";
  const statusLabel = (s: string) =>
    s === "ok" ? "OK" : s === "warn" ? "Attention" : s === "err" ? "Erreur" : "Info";
  const statusBg    = (s: string) =>
    s === "ok" ? "#E6F7F1" : s === "warn" ? "#FEF8E7" : s === "err" ? "#FDECEA" : "#EDE9FE";

  return (
    <div className="card">
      <div className="card-header">
        <h2><span className="card-icon">📋</span>
          {isExpert ? "Rapport de décisions" : "Ce qui s'est passé — étape par étape"}
        </h2>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {okCount   > 0 && <span className="badge" style={{ background:"#DCFCE7", color:"var(--ok)",   borderColor:"#86EFAC" }}>{okCount} OK</span>}
          {warnCount > 0 && <span className="badge" style={{ background:"#FEF8E7", color:"var(--warn)", borderColor:"#FCD34D" }}>{warnCount} Attention</span>}
          {errCount  > 0 && <span className="badge" style={{ background:"#FEE2E2", color:"var(--err)",  borderColor:"#FCA5A5" }}>{errCount} Erreur</span>}
          {infoCount > 0 && <span className="badge badge-running">{infoCount} Info</span>}
        </div>
      </div>

      <p className="muted-text" style={{ marginBottom: 14 }}>
        {isExpert
          ? "Chaque décision prise par l'orchestrateur avec sa justification technique."
          : "Voici toutes les décisions que l'intelligence artificielle a prises pour construire votre modèle, avec une explication pour chacune."}
      </p>

      <button className="toggle-btn" onClick={() => setOpen(o => !o)}>
        {open ? "▲ Réduire" : "▼ Voir le rapport complet"}
      </button>

      {open && (
        <div style={{ marginTop: 4, position: "relative", paddingLeft: 28 }}>
          <div style={{ position: "absolute", left: 11, top: 8, bottom: 24, width: 2, background: "var(--border)", borderRadius: 2 }} />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {journal.map((entry, i) => (
              <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
                <div style={{
                  width: 14, height: 14, borderRadius: "50%", flexShrink: 0, marginTop: 14,
                  background: statusColor(entry.statut), border: "2.5px solid var(--bg)",
                  boxShadow: `0 0 0 2px ${statusColor(entry.statut)}`,
                  position: "relative", zIndex: 1, marginLeft: -5,
                }} />
                <div style={{ flex: 1, padding: "11px 14px 12px", borderRadius: 10, background: "var(--surface)", border: "1px solid var(--border)", marginBottom: 4 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, flexWrap: "wrap" }}>
                    <span style={{ fontWeight: 700, fontSize: 13 }}>{entry.étape}</span>
                    <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 20, fontWeight: 700, background: statusBg(entry.statut), color: statusColor(entry.statut), border: `1px solid ${statusColor(entry.statut)}44`, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      {statusLabel(entry.statut)}
                    </span>
                  </div>
                  <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 6, lineHeight: 1.45 }}>{entry.décision}</div>
                  <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.65, paddingLeft: 10, borderLeft: `2px solid ${statusColor(entry.statut)}44` }}>
                    💡 {entry.justification}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Chat Panel ────────────────────────────────────────────────────────────────

type ChatLoadState = "idle" | "loading" | "ready" | "error";

interface ChatMsg { role: "user" | "assistant"; content: string; }

function ChatPanel({ runId, isExpert }: { runId: string; isExpert: boolean }) {
  const [loadState,   setLoadState]   = useState<ChatLoadState>("idle");
  const [messages,    setMessages]    = useState<ChatMsg[]>([]);
  const [input,       setInput]       = useState("");
  const [generating,  setGenerating]  = useState(false);
  const [device,      setDevice]      = useState<string | null>(null);
  const [freeVramGb,  setFreeVramGb]  = useState<number | null>(null);
  const [error,       setError]       = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, generating]);

  async function onLoad() {
    setLoadState("loading");
    setError("");
    try {
      const res = await loadChatModel(runId);
      setDevice(res.device);
      setFreeVramGb((res as Record<string, unknown>).free_vram_gb as number ?? null);
      setLoadState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoadState("error");
    }
  }

  async function onUnload() {
    await unloadChatModel(runId);
    setLoadState("idle");
    setMessages([]);
    setDevice(null);
    setError("");
  }

  async function onSend() {
    const msg = input.trim();
    if (!msg || generating) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setGenerating(true);
    setError("");
    try {
      const reply = await sendChatMessage(runId, msg);
      setMessages(prev => [...prev, { role: "assistant", content: reply }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
  }

  // ── idle state: show "Tester le modèle" button ──────────────────────────────
  if (loadState === "idle" || loadState === "error") {
    return (
      <div className="card">
        <div className="card-header">
          <h2><span className="card-icon">💬</span>Tester le modèle</h2>
        </div>
        {!isExpert && (
          <div className="novice-guide-box">
            <strong>Discutez avec votre modèle !</strong> Cliquez sur le bouton ci-dessous pour charger
            votre modèle fine-tuné et lui poser des questions directement. C'est le meilleur moyen de
            juger sa qualité par vous-même.
          </div>
        )}
        <p className="muted-text" style={{ marginBottom: 16 }}>
          Le modèle sera chargé en VRAM. Vous pourrez lui poser des questions librement,
          puis le décharger pour libérer la mémoire GPU.
        </p>
        {error && <div className="error-box" style={{ marginBottom: 14 }}>⚠ {error}</div>}
        <button className="action-btn action-accent" onClick={onLoad} style={{ fontSize: 14 }}>
          ▶ Charger le modèle et démarrer le chat
        </button>
      </div>
    );
  }

  // ── loading state ───────────────────────────────────────────────────────────
  if (loadState === "loading") {
    return (
      <div className="card">
        <div className="card-header">
          <h2><span className="card-icon">💬</span>Tester le modèle</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "20px 0" }}>
          <div className="spinner" style={{ width: 24, height: 24, border: "3px solid rgba(124,58,237,0.15)", borderTopColor: "var(--primary)", flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, marginBottom: 4 }}>Chargement du modèle en mémoire…</div>
            <div className="muted-text">Cette opération peut prendre 1 à 3 minutes selon la taille du modèle.</div>
          </div>
        </div>
      </div>
    );
  }

  // ── ready state: chat interface ─────────────────────────────────────────────
  return (
    <div className="card">
      <div className="card-header">
        <h2><span className="card-icon">💬</span>Chat avec le modèle fine-tuné</h2>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            fontSize: 12, padding: "3px 10px", borderRadius: 20, fontWeight: 700,
            background: device === "gpu" ? "#E6F7F1" : "#FEF8E7",
            color: device === "gpu" ? "var(--ok)" : "var(--warn)",
            border: `1px solid ${device === "gpu" ? "#BBF7D0" : "#FCD34D"}`,
          }}>
            {device === "gpu" ? "⚡ GPU" : "🖥 CPU"}
            {freeVramGb !== null && (
              <span style={{ fontWeight: 400, marginLeft: 5, opacity: 0.8 }}>
                ({freeVramGb.toFixed(1)} Go libres au chargement)
              </span>
            )}
          </span>
          <button
            className="action-btn action-secondary"
            style={{ fontSize: 12, padding: "4px 12px" }}
            onClick={onUnload}
          >
            ⏹ Décharger
          </button>
        </div>
      </div>

      {!isExpert && messages.length === 0 && (
        <div className="novice-guide-box" style={{ marginBottom: 14 }}>
          Posez une question à votre modèle. Évaluez la pertinence et la précision de ses réponses
          par rapport à vos documents originaux.
        </div>
      )}

      {/* Messages */}
      <div style={{
        minHeight: 200, maxHeight: 420, overflowY: "auto",
        border: "1px solid var(--border)", borderRadius: 10,
        padding: "12px 14px", marginBottom: 14,
        background: "var(--panel)", display: "flex", flexDirection: "column", gap: 10,
      }}>
        {messages.length === 0 && (
          <div style={{ color: "var(--muted)", fontSize: 13, textAlign: "center", margin: "auto" }}>
            Aucun message pour l'instant. Posez votre première question ci-dessous.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{
            display: "flex",
            justifyContent: m.role === "user" ? "flex-end" : "flex-start",
          }}>
            <div style={{
              maxWidth: "80%", padding: "9px 14px", borderRadius: 12,
              fontSize: 13.5, lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word",
              background: m.role === "user" ? "var(--primary)" : "var(--surface)",
              color: m.role === "user" ? "#fff" : "var(--text)",
              border: m.role === "assistant" ? "1px solid var(--border)" : "none",
              borderBottomRightRadius: m.role === "user" ? 2 : 12,
              borderBottomLeftRadius:  m.role === "assistant" ? 2 : 12,
            }}>
              {m.role === "assistant" && (
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--primary)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Modèle
                </div>
              )}
              {m.content}
            </div>
          </div>
        ))}
        {generating && (
          <div style={{ display: "flex", justifyContent: "flex-start" }}>
            <div style={{
              padding: "9px 16px", borderRadius: 12, borderBottomLeftRadius: 2,
              background: "var(--surface)", border: "1px solid var(--border)",
              display: "flex", alignItems: "center", gap: 8,
            }}>
              <div className="spinner" style={{ width: 14, height: 14, border: "2px solid rgba(124,58,237,0.2)", borderTopColor: "var(--primary)" }} />
              <span className="muted-text" style={{ fontSize: 13 }}>Génération en cours…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error-box" style={{ marginBottom: 10 }}>⚠ {error}</div>}

      {/* Input */}
      <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
        <textarea
          rows={2}
          placeholder="Posez votre question… (Entrée pour envoyer, Shift+Entrée pour sauter une ligne)"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          disabled={generating}
          style={{
            flex: 1, resize: "vertical", minHeight: 52,
            padding: "10px 14px", borderRadius: 10,
            border: "1.5px solid var(--border)", fontSize: 13.5,
            background: "var(--surface)", color: "var(--text)",
            outline: "none", fontFamily: "inherit", lineHeight: 1.5,
          }}
        />
        <button
          className="action-btn action-primary"
          style={{ height: 52, minWidth: 80, fontSize: 14, borderRadius: 10 }}
          onClick={onSend}
          disabled={!input.trim() || generating}
        >
          Envoyer
        </button>
      </div>
    </div>
  );
}

// ── Main Pipeline Page ────────────────────────────────────────────────────────

export default function PipelinePage({
  user, onHistory,
}: {
  user: UserProfile;
  onHistory: () => void;
}) {
  const isExpert = user.isExpert;

  const [files,          setFiles]          = useState<File[]>([]);
  const [goal,           setGoal]           = useState("");
  const [language,       setLanguage]       = useState("fr");
  const [objective,      setObjective]      = useState("balanced");
  const [gpuVram,        setGpuVram]        = useState(4);
  const [gpuInfo,        setGpuInfo]        = useState<GpuInfo | null>(null);
  const [task,           setTask]           = useState("");
  const [logMode,        setLogMode]        = useState<"summary"|"detail">("summary");
  const [status,         setStatus]         = useState<Status>("idle");
  const [logs,           setLogs]           = useState<string[]>([]);
  const [logFilter,      setLogFilter]      = useState<"all"|"warn"|"error">("all");
  const [error,          setError]          = useState<string>("");
  const [runId,          setRunId]          = useState<string>("");
  const [dataset,        setDataset]        = useState<DatasetResponse | null>(null);
  const [trainingReport, setTrainingReport] = useState<TrainingReport | null>(null);
  const [evalReport,     setEvalReport]     = useState<EvalReport | null>(null);
  const [evalState,      setEvalState]      = useState<EvalState>("idle");
  const [lossHistory,    setLossHistory]    = useState<LossHistory | null>(null);
  const [trainingImages, setTrainingImages] = useState<TrainingImage[]>([]);
  const [improvementLog, setImprovementLog] = useState<Array<Record<string, unknown>>>([]);
  const [decisionJournal,setDecisionJournal]= useState<DecisionJournalEntry[]>([]);
  const [sseRetrying,          setSseRetrying]          = useState(false);
  const [confirmPayload,       setConfirmPayload]       = useState<ConfirmPayload | null>(null);
  const [confirmBusy,          setConfirmBusy]          = useState(false);
  const [blockedReason,        setBlockedReason]        = useState<string>("");
  const [showAdditionalUpload, setShowAdditionalUpload] = useState(false);
  const [additionalFiles,      setAdditionalFiles]      = useState<File[]>([]);
  const [uploadBusy,           setUploadBusy]           = useState(false);

  const logsRef              = useRef<HTMLPreElement>(null);
  const fileInputRef         = useRef<HTMLInputElement>(null);
  const additionalFileRef    = useRef<HTMLInputElement>(null);
  const unsubRef = useRef<null | (() => void)>(null);
  const pollRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs]);

  // Auto-detect GPU VRAM on mount — pre-fills the field with the real value
  useEffect(() => {
    fetchGpuInfo().then(info => {
      setGpuInfo(info);
      if (info.detected && info.available_gb) {
        setGpuVram(info.available_gb);
      }
    }).catch(() => {});
  }, []);

  useEffect(() => () => {
    unsubRef.current?.();
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    const savedId = localStorage.getItem(LS_RUN_KEY);
    if (!savedId) return;
    fetchRunResult(savedId).then(result => {
      if (!result || result.status !== "running") { localStorage.removeItem(LS_RUN_KEY); return; }
      setRunId(savedId); setStatus("running");
      setLogs([`── reconnexion au pipeline en cours (run ${savedId}) ──`]);
      unsubRef.current = subscribeToRun(savedId, (e: RunEvent) => {
        if (e.kind === "log") { setSseRetrying(false); setLogs(prev => [...prev, e.data]); }
        else if (e.kind === "confirm") { try { setConfirmPayload(JSON.parse(e.data) as ConfirmPayload); } catch { /* */ } }
        else if (e.kind === "done")  { try { handleDonePayload(JSON.parse(e.data), savedId); } catch { setError("Réponse malformée"); setStatus("error"); } }
        else if (e.kind === "error") { setSseRetrying(false); localStorage.removeItem(LS_RUN_KEY); setError(e.data); setStatus("error"); }
      }, setSseRetrying);
    }).catch(() => localStorage.removeItem(LS_RUN_KEY));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function handleDonePayload(payload: { ok: boolean; error?: string; result?: any }, id: string) {
    setSseRetrying(false); setConfirmPayload(null); setShowAdditionalUpload(false); setAdditionalFiles([]); localStorage.removeItem(LS_RUN_KEY);
    if (!payload.ok) { setError(payload.error ?? "Erreur inconnue"); setStatus("error"); return; }
    const finalOut       = payload.result?.final_output ?? payload.result;
    const pipelineStatus = finalOut?.status ?? "done";
    if (pipelineStatus === "blocked") { setBlockedReason(finalOut?.summary ?? "Fichiers incompatibles."); setStatus("blocked"); return; }
    setStatus("done");
    fetchDataset(id).then(setDataset).catch(() => null);
    fetchTrainingReport(id).then(r => { if (r) setTrainingReport(r); }).catch(() => null);
    fetchLossHistory(id).then(h => { if (h) setLossHistory(h); }).catch(() => null);
    fetchTrainingImages(id).then(setTrainingImages).catch(() => null);
    if (Array.isArray(finalOut?.improvement_log) && finalOut.improvement_log.length)
      setImprovementLog(finalOut.improvement_log as Array<Record<string, unknown>>);
    if (Array.isArray(finalOut?.decision_journal) && finalOut.decision_journal.length)
      setDecisionJournal(finalOut.decision_journal as DecisionJournalEntry[]);
  }

  async function onRun() {
    if (files.length === 0) { setError("Veuillez ajouter au moins un fichier."); return; }
    setStatus("running"); setLogs([]); setError(""); setBlockedReason("");
    setDataset(null); setTrainingReport(null); setEvalReport(null);
    setEvalState("idle"); setLossHistory(null); setTrainingImages([]);
    setImprovementLog([]); setDecisionJournal([]);
    setConfirmPayload(null); setShowAdditionalUpload(false); setAdditionalFiles([]);
    try {
      const id = await startRun({ files, goal, language, objective, gpu_vram_gb: gpuVram, task });
      setRunId(id); localStorage.setItem(LS_RUN_KEY, id);
      unsubRef.current = subscribeToRun(id, (e: RunEvent) => {
        if (e.kind === "log") { setSseRetrying(false); setLogs(prev => [...prev, e.data]); }
        else if (e.kind === "confirm") { try { setConfirmPayload(JSON.parse(e.data) as ConfirmPayload); } catch { /* */ } }
        else if (e.kind === "open")   { setSseRetrying(false); setLogs(prev => [...prev, `── pipeline démarré (run ${id}) ──`]); }
        else if (e.kind === "done")   { try { handleDonePayload(JSON.parse(e.data), id); } catch { setError("Réponse malformée"); setStatus("error"); } }
        else if (e.kind === "error")  { setSseRetrying(false); localStorage.removeItem(LS_RUN_KEY); setError(e.data); setStatus("error"); }
      }, setSseRetrying);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); setStatus("error"); }
  }

  async function onConfirm(decision: "approve" | "refuse" | "cancel-pipeline") {
    if (!runId || confirmBusy) return;
    // "refuse" → show upload zone (no API call yet)
    if (decision === "refuse") {
      setShowAdditionalUpload(true);
      return;
    }
    // "cancel-pipeline" → hard cancel via API
    if (decision === "cancel-pipeline") {
      setConfirmBusy(true);
      try {
        await confirmAction(runId, "refuse");
        setConfirmPayload(null);
        setShowAdditionalUpload(false);
        setLogs(prev => [...prev, "⛔ Pipeline annulé — importez d'autres fichiers et relancez."]);
      } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
      finally { setConfirmBusy(false); }
      return;
    }
    // "approve"
    setConfirmBusy(true);
    try {
      await confirmAction(runId, "approve");
      setConfirmPayload(null);
      setShowAdditionalUpload(false);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setConfirmBusy(false); }
  }

  async function onUploadAdditional() {
    if (!runId || additionalFiles.length === 0 || uploadBusy) return;
    setUploadBusy(true);
    try {
      const uploaded = await uploadAdditionalFiles(runId, additionalFiles);
      setLogs(prev => [...prev,
        `⬆ ${uploaded.length} fichier(s) supplémentaire(s) envoyé(s) — reprise du pipeline…`]);
      setAdditionalFiles([]);
      setShowAdditionalUpload(false);
      setConfirmPayload(null);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setUploadBusy(false); }
  }

  async function onRunEval() {
    if (!runId) return;
    setEvalState("loading");
    try {
      await triggerEval(runId);
      pollRef.current = setInterval(async () => {
        try {
          const report = await fetchEvalReport(runId);
          if (report) { setEvalReport(report); setEvalState("done"); if (pollRef.current) clearInterval(pollRef.current); }
        } catch { /* keep polling */ }
      }, 12_000);
    } catch (err) { setEvalState("error"); setError(err instanceof Error ? err.message : String(err)); }
  }

  const statusLabel: Record<Status, string> = {
    idle: "en attente", running: "en cours", done: "terminé", error: "erreur", blocked: "bloqué",
  };

  const filteredLogs = logs.filter(line => {
    if (logMode === "summary") return line.startsWith("[PIPELINE]") || /^\d{2}:\d{2}:\d{2}\s+(ERROR|CRITICAL)\b/.test(line);
    if (logFilter === "warn")  return /warn/i.test(line);
    if (logFilter === "error") return /error|failed|exception/i.test(line);
    return true;
  });

  const pipelineStep = (() => {
    if (status === "idle")  return 0;
    if (status === "done")  return 5;
    if (status === "error" || status === "blocked") return -1;
    const last = logs[logs.length - 1] ?? "";
    if (/train/i.test(last))                       return 4;
    if (/qa|dataset|paire/i.test(last))            return 3;
    if (/profil|intent|feasib|select/i.test(last)) return 2;
    return 1;
  })();

  const STEPS = isExpert
    ? ["Données", "Analyse", "Préparation", "Entraînement", "Évaluation"]
    : ["Vos fichiers", "Compréhension", "Données QA", "Apprentissage", "Test qualité"];

  return (
    <>
      {/* Hero Banner */}
      <header className="hero-banner">
        <div className="hero-noise" />
        <NeuralNetDecoration />
        <div className="hero-inner">
          <div className="hero-eyebrow">AI-Powered Model Trainer · {isExpert ? "Pipeline Expert" : "Mode guidé"}</div>
          <h1 className="hero-title">
            {isExpert ? "Vos données ont du potentiel. " : "Vos documents ont une valeur cachée. "}
            <span className="hero-title-accent">{isExpert ? "Exploitez-le à fond." : "Révélez-la."}</span>
          </h1>
          <p className="hero-subtitle">
            {isExpert
              ? "Du brut au déployable — fine-tuning PEFT automatisé, évaluation rigoureuse et optimisation itérative autonome en un seul pipeline."
              : "Importez vos fichiers, décrivez votre objectif en quelques mots — notre IA fait le reste, de A à Z, sans aucune compétence technique requise."}
          </p>
          <div className="hero-pills">
            {isExpert ? (
              <>
                <span className="hero-pill">✦ Fine-tuning QLoRA / LoRA / DoRA automatique</span>
                <span className="hero-pill">✦ ROUGE · BLEU · LLM-as-judge</span>
                <span className="hero-pill">✦ Optimisation itérative autonome</span>
              </>
            ) : (
              <>
                <span className="hero-pill">✦ Zéro compétence ML requise</span>
                <span className="hero-pill">✦ Guidage clair à chaque étape</span>
                <span className="hero-pill">✦ Résultats expliqués en français</span>
              </>
            )}
          </div>
        </div>
      </header>

      <div className="page">

        {/* History shortcut — shown when run is done */}
        {status === "done" && (
          <div style={{
            display: "flex", justifyContent: "flex-end", marginBottom: 4,
          }}>
            <button className="action-btn action-secondary" style={{ fontSize: 12 }} onClick={onHistory}>
              📚 Voir ce modèle dans l'historique →
            </button>
          </div>
        )}

        {/* Pipeline steps */}
        <div className="pipeline-steps">
          {STEPS.map((label, i) => (
            <div key={label} className={`step ${pipelineStep === i + 1 ? "active" : pipelineStep > i + 1 ? "done" : ""}`}>
              <span className="step-num">{pipelineStep > i + 1 ? "✓" : i + 1}</span>
              {label}
            </div>
          ))}
        </div>

        {/* Form */}
        <section className="card">
          <div className="card-header">
            <h2><span className="card-icon">📂</span>Configuration du pipeline</h2>
          </div>

          {!isExpert && (
            <div className="novice-guide-box" style={{ marginBottom: 20 }}>
              <strong>Comment ça marche ?</strong> Importez vos documents (PDF, TXT, CSV), décrivez ce que vous voulez
              que votre modèle sache faire, et cliquez sur "Lancer". L'IA s'occupe de tout le reste automatiquement.
            </div>
          )}

          <div className="field">
            <span className="field-label">Documents sources</span>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              style={{ display: "none" }}
              disabled={status === "running"}
              onChange={e => {
                const added = Array.from(e.target.files ?? []);
                setFiles(prev => {
                  const existing = new Set(prev.map(f => f.name));
                  return [...prev, ...added.filter(f => !existing.has(f.name))];
                });
                e.target.value = "";
              }}
            />
            <div
              className={`upload-zone ${status === "running" ? "upload-zone-disabled" : ""}`}
              onClick={() => status !== "running" && fileInputRef.current?.click()}
            >
              <div className="upload-icon">📂</div>
              <div className="upload-label">Cliquez pour ajouter des fichiers</div>
              <div className="upload-sub">PDF · TXT · CSV · Excel — chaque clic ajoute sans remplacer</div>
              <div className="upload-btn">+ Choisir des fichiers</div>
            </div>
            {!isExpert && <div className="field-hint">Plus vous fournissez de texte, meilleur sera le modèle.</div>}
            {files.length > 0 && (
              <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
                {files.map(f => (
                  <span key={f.name} className="file-chip">
                    📄 {f.name}
                    <button
                      className="file-chip-remove"
                      onClick={e => { e.stopPropagation(); setFiles(prev => prev.filter(x => x.name !== f.name)); }}
                      title="Retirer ce fichier"
                    >✕</button>
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="field">
            <span className="field-label">Tâche {!isExpert && "— que voulez-vous faire ?"}</span>
            {!isExpert && (
              <div className="field-hint" style={{ marginBottom: 10 }}>
                Sélectionnez la tâche principale de votre modèle. "Auto-détection" laisse l'IA choisir d'après votre objectif.
              </div>
            )}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {TASKS.map(t => {
                const selected = task === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    disabled={status === "running"}
                    onClick={() => setTask(t.id)}
                    title={t.desc}
                    style={{
                      display:        "flex",
                      alignItems:     "center",
                      gap:            6,
                      padding:        "7px 13px",
                      borderRadius:   20,
                      border:         selected ? "2px solid var(--primary)" : "1.5px solid var(--border)",
                      background:     selected ? "var(--primary)" : "var(--surface)",
                      color:          selected ? "#fff" : "var(--text)",
                      fontWeight:     selected ? 700 : 500,
                      fontSize:       13,
                      cursor:         status === "running" ? "not-allowed" : "pointer",
                      opacity:        status === "running" ? 0.6 : 1,
                      transition:     "all 0.15s",
                      whiteSpace:     "nowrap",
                    }}
                  >
                    <span style={{ fontSize: 15 }}>{t.icon}</span>
                    {t.label}
                  </button>
                );
              })}
            </div>
            {task && (
              <div className="field-hint" style={{ marginTop: 6 }}>
                {TASKS.find(t => t.id === task)?.desc}
              </div>
            )}
          </div>

          <label className="field">
            <span>Objectif {isExpert ? "(description libre)" : "— que doit savoir faire votre modèle ?"}</span>
            <textarea
              rows={3}
              placeholder={isExpert
                ? "Ex. : Fine-tuner un assistant Q&R médical en français pour répondre aux questions des cliniciens."
                : "Ex. : Je veux un assistant qui répond aux questions sur mes documents médicaux en français."}
              value={goal}
              onChange={e => setGoal(e.target.value)}
              disabled={status === "running"}
            />
            {!isExpert && <div className="field-hint">Décrivez en phrases naturelles ce que vous attendez du modèle. Pas besoin de vocabulaire technique.</div>}
          </label>

          <div className="row">
            <label className="field" style={{ maxWidth: 260 }}>
              <span>Langue {!isExpert && "des documents"}</span>
              <select value={language} onChange={e => setLanguage(e.target.value)} disabled={status === "running"}>
                <option value="fr">Français</option>
                <option value="en">English</option>
                <option value="ar">Arabe</option>
              </select>
              {!isExpert && <div className="field-hint">La langue principale de vos documents et des futures questions.</div>}
            </label>

            <label className="field" style={{ maxWidth: 260 }}>
              <span>{isExpert ? "Objectif d'entraînement" : "Priorité"}</span>
              <select value={objective} onChange={e => setObjective(e.target.value)} disabled={status === "running"}>
                <option value="balanced">{isExpert ? "Équilibré — précision & vitesse" : "Équilibré (recommandé)"}</option>
                <option value="speed">{isExpert ? "Rapidité — entraînement minimal" : "Rapide — entraînement court"}</option>
                <option value="performance">{isExpert ? "Performance max — meilleure qualité" : "Meilleure qualité possible"}</option>
              </select>
              {!isExpert && <div className="field-hint">En doute ? Laissez sur "Équilibré".</div>}
            </label>

            {gpuInfo?.detected && (
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "6px 12px", borderRadius: 8,
                background: "#E6F7F1", border: "1px solid #BBF7D0",
                fontSize: 13, color: "#16A34A", fontWeight: 500,
                alignSelf: "flex-end", marginBottom: 4,
              }}>
                <span>⚡</span>
                <span>{gpuInfo.name} — {gpuInfo.total_gb} Go VRAM</span>
              </div>
            )}
          </div>

          <button className="run-btn" onClick={onRun} disabled={status === "running"}>
            {status === "running"
              ? <><div className="spinner" style={{ width: 16, height: 16, border: "2.5px solid rgba(255,255,255,0.4)", borderTopColor: "#fff", display: "inline-block", verticalAlign: "middle", marginRight: 8 }} />Pipeline en cours…</>
              : (isExpert ? "▶ Lancer le pipeline" : "▶ Créer mon modèle")}
          </button>

          {error && <div className="error-box">⚠ {error}</div>}
        </section>

        {/* Confirmation */}
        {confirmPayload && (
          <section className="card confirm-card">
            <div className="confirm-header">
              <span className="confirm-icon">⚠</span>
              <div style={{ flex: 1 }}>
                <div className="confirm-title">Dataset insuffisant — action requise</div>
                <div className="confirm-desc">
                  Le dataset actuel contient{" "}
                  <strong>{confirmPayload.current_pairs} paire(s)</strong>.
                  L'objectif recommandé est de{" "}
                  <strong>{confirmPayload.target_pairs} paires</strong>.
                </div>
                {!isExpert && (
                  <div className="novice-guide-box" style={{ marginTop: 10 }}>
                    Il n'y a pas assez d'exemples pour entraîner un bon modèle. Vous avez deux options :
                    laisser le pipeline générer automatiquement des paires depuis vos documents,
                    ou importer des fichiers supplémentaires pour enrichir le dataset.
                  </div>
                )}
              </div>
            </div>

            {/* QA sample preview */}
            {confirmPayload.sample_pairs && confirmPayload.sample_pairs.length > 0 && (
              <div style={{ marginTop: 14, marginBottom: 4 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: "var(--muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                  Aperçu des paires déjà générées
                </div>
                <div className="qa-list" style={{ maxHeight: 220, overflowY: "auto" }}>
                  {confirmPayload.sample_pairs.map((pair, i) => (
                    <div key={i} className="qa-item">
                      <p className="qa-q"><strong>Q </strong>{pair.q}</p>
                      <p className="qa-a"><strong>R </strong>{pair.a}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Main confirm actions */}
            {!showAdditionalUpload ? (
              <div className="confirm-actions" style={{ marginTop: 16 }}>
                <button
                  className="action-btn action-primary"
                  onClick={() => onConfirm("approve")}
                  disabled={confirmBusy}
                >
                  {confirmBusy ? "…" : "✓ Générer automatiquement les paires manquantes"}
                </button>
                <button
                  className="action-btn action-secondary"
                  onClick={() => onConfirm("refuse")}
                  disabled={confirmBusy}
                >
                  ⬆ Importer d'autres fichiers
                </button>
              </div>
            ) : (
              /* Additional file upload zone */
              <div style={{ marginTop: 16 }}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 10, color: "var(--text)" }}>
                  Importez des fichiers supplémentaires (PDF, TXT, CSV, Excel)
                </div>
                {!isExpert && (
                  <div className="novice-guide-box" style={{ marginBottom: 12 }}>
                    Ajoutez des documents complémentaires pour enrichir votre dataset.
                    Le pipeline les traitera et reprendra automatiquement.
                  </div>
                )}

                <input
                  ref={additionalFileRef}
                  type="file"
                  multiple
                  style={{ display: "none" }}
                  onChange={e => {
                    const added = Array.from(e.target.files ?? []);
                    setAdditionalFiles(prev => {
                      const existing = new Set(prev.map(f => f.name));
                      return [...prev, ...added.filter(f => !existing.has(f.name))];
                    });
                    e.target.value = "";
                  }}
                />
                <div
                  className="upload-zone"
                  onClick={() => additionalFileRef.current?.click()}
                  style={{ marginBottom: additionalFiles.length ? 10 : 0 }}
                >
                  <div className="upload-icon">📂</div>
                  <div className="upload-label">Cliquez pour ajouter des fichiers</div>
                  <div className="upload-sub">PDF · TXT · CSV · Excel</div>
                  <div className="upload-btn">+ Choisir des fichiers</div>
                </div>

                {additionalFiles.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
                    {additionalFiles.map(f => (
                      <span key={f.name} className="file-chip">
                        📄 {f.name}
                        <button
                          className="file-chip-remove"
                          onClick={() => setAdditionalFiles(prev => prev.filter(x => x.name !== f.name))}
                          title="Retirer ce fichier"
                        >✕</button>
                      </span>
                    ))}
                  </div>
                )}

                <div className="confirm-actions">
                  <button
                    className="action-btn action-primary"
                    onClick={onUploadAdditional}
                    disabled={uploadBusy || additionalFiles.length === 0}
                  >
                    {uploadBusy ? "Envoi en cours…" : `⬆ Envoyer ${additionalFiles.length > 0 ? `(${additionalFiles.length} fichier${additionalFiles.length > 1 ? "s" : ""})` : ""} et continuer`}
                  </button>
                  <button
                    className="action-btn action-secondary"
                    onClick={() => { setShowAdditionalUpload(false); setAdditionalFiles([]); }}
                    disabled={uploadBusy}
                  >
                    ← Retour
                  </button>
                  <button
                    className="action-btn"
                    style={{ background: "var(--surface)", color: "var(--muted)", border: "1px solid var(--border)" }}
                    onClick={() => onConfirm("cancel-pipeline")}
                    disabled={uploadBusy || confirmBusy}
                  >
                    ✕ Annuler le pipeline
                  </button>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Blocked */}
        {status === "blocked" && (
          <section className="card blocked-card">
            <div className="blocked-header">
              <span className="blocked-icon">⛔</span>
              <div style={{ flex: 1 }}>
                <div className="blocked-title">Pipeline bloqué — fichiers incompatibles</div>
                <div className="blocked-reason">{blockedReason}</div>
                <div className="blocked-hint">
                  Pour relancer, importez uniquement des fichiers du <strong>même domaine</strong>.
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Live logs */}
        <section className="card">
          <div className="card-header">
            <h2><span className="card-icon">📋</span>Suivi du pipeline</h2>
            <span className={`badge badge-${status}`}>{statusLabel[status]}</span>
          </div>

          {sseRetrying && (
            <div className="sse-retry-banner">
              <div className="spinner" style={{ width: 12, height: 12, border: "2px solid rgba(124,58,237,0.25)", borderTopColor: "var(--primary)", display: "inline-block", verticalAlign: "middle", marginRight: 8 }} />
              Connexion interrompue — reconnexion en cours…
            </div>
          )}

          <div className="log-toolbar">
            <span className="muted-text">{filteredLogs.length}/{logs.length} lignes</span>
            <div className="log-filter">
              <button className={`log-filter-btn ${logMode === "summary" ? "active" : ""}`} onClick={() => setLogMode("summary")}>
                📋 {isExpert ? "Résumé structuré" : "Vue guidée"}
              </button>
              <button className={`log-filter-btn ${logMode === "detail" ? "active" : ""}`} onClick={() => setLogMode("detail")}>
                🔍 {isExpert ? "Logs complets" : "Logs techniques"}
              </button>
              {logMode === "detail" && (
                <>
                  <span style={{ width: 1, height: 18, background: "var(--border)", display: "inline-block", margin: "0 2px", verticalAlign: "middle" }} />
                  {(["all", "warn", "error"] as const).map(f => (
                    <button key={f} className={`log-filter-btn ${logFilter === f ? "active" : ""}`} onClick={() => setLogFilter(f)}>
                      {f === "all" ? "Tout" : f === "warn" ? "⚠ Alertes" : "✕ Erreurs"}
                    </button>
                  ))}
                </>
              )}
            </div>
          </div>

          {logMode === "summary"
            ? <LivePipelineFeed logs={logs} status={status} isExpert={isExpert} />
            : (
              <pre ref={logsRef} className="log-pre">
                {filteredLogs.length === 0
                  ? "(aucune sortie pour l'instant)"
                  : filteredLogs.map((line, i) => <span key={i} className={classifyLog(line)}>{line + "\n"}</span>)}
              </pre>
            )}
        </section>

        {/* Decision journal */}
        {decisionJournal.length > 0 && (
          <DecisionJournalCard journal={decisionJournal} isExpert={isExpert} />
        )}

        {/* Training dashboard */}
        {trainingReport && (
          <TrainingDashboard
            report={trainingReport} runId={runId}
            lossHistory={lossHistory} trainingImages={trainingImages}
            isExpert={isExpert}
          />
        )}

        {/* Improvement log */}
        {improvementLog.length > 0 && (
          <section className="card">
            <div className="card-header">
              <h2><span className="card-icon">🔄</span>{isExpert ? "Boucle d'amélioration automatique" : "Tentatives d'amélioration"}</h2>
              <span className="badge badge-running">{improvementLog.length} itération(s)</span>
            </div>
            {!isExpert && (
              <div className="novice-guide-box" style={{ marginBottom: 12 }}>
                Le pipeline a essayé plusieurs fois d'améliorer le modèle. Chaque ligne montre ce qui a été tenté
                et le score obtenu. Les flèches ↑ indiquent une amélioration, ↓ une dégradation.
              </div>
            )}
            <p className="muted-text" style={{ marginBottom: 12 }}>
              {isExpert
                ? "Hyperparamètres ajustés et métriques obtenues à chaque itération."
                : "Plus le score est proche de 1, meilleur est le modèle."}
            </p>
            <ImprovementLog log={improvementLog} />
          </section>
        )}

        {/* Evaluation */}
        {(trainingReport || status === "done") && (
          <EvalCard evalState={evalState} evalReport={evalReport} onRunEval={onRunEval} isExpert={isExpert} />
        )}

        {/* Chat with model — shown after training completes */}
        {(trainingReport || status === "done") && (runId || trainingReport?.job_id) && (
          <ChatPanel runId={runId || trainingReport?.job_id || ""} isExpert={isExpert} />
        )}

        {/* Dataset */}
        {dataset && (
          <section className="card">
            <div className="card-header">
              <h2><span className="card-icon">🗂</span>Dataset généré — {dataset.count} paires ({dataset.format})</h2>
              <a href={datasetDownloadUrl(runId)} download>
                <button className="action-btn action-ok">⬇ Télécharger .jsonl</button>
              </a>
            </div>
            {!isExpert && (
              <div className="novice-guide-box" style={{ marginBottom: 12 }}>
                Voici les paires question-réponse que le pipeline a générées à partir de vos documents.
                C'est ce que le modèle a appris.
              </div>
            )}
            <div className="qa-list">
              {dataset.pairs.map((raw, i) => {
                const pair = raw as Record<string, unknown>;
                if (dataset.format === "alpaca") {
                  return (
                    <div key={i} className="qa-item">
                      <p className="qa-q"><strong>Q </strong>{String(pair.instruction ?? "")}</p>
                      <p className="qa-a"><strong>R </strong>{String(pair.output ?? "")}</p>
                    </div>
                  );
                }
                const convs = (pair.conversations as { from: string; value: string }[]) ?? [];
                const human = convs.find(c => c.from === "human");
                const gpt   = convs.find(c => c.from === "gpt");
                return (
                  <div key={i} className="qa-item">
                    <p className="qa-q"><strong>Q </strong>{human?.value ?? ""}</p>
                    <p className="qa-a"><strong>R </strong>{gpt?.value ?? ""}</p>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </>
  );
}
