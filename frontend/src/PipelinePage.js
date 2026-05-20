import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { NeuralNetDecoration } from "./MLDecorations";
import { startRun, subscribeToRun, fetchDataset, datasetDownloadUrl, fetchTrainingReport, fetchEvalReport, triggerEval, fetchLossHistory, fetchTrainingImages, fetchRunResult, confirmAction, uploadAdditionalFiles, fetchGpuInfo, reportDownloadUrl, modelDownloadUrl, } from "./api";
const LS_RUN_KEY = "active_run_id";
// ── Constants ─────────────────────────────────────────────────────────────────
const PEFT_LABELS = {
    qlora: { label: "QLoRA", desc: "Quantifié 4-bit — économe en VRAM" },
    lora: { label: "LoRA", desc: "Léger bf16 — meilleure précision" },
    lora_int8: { label: "LoRA int8", desc: "Semi-quantifié 8-bit" },
    dora: { label: "DoRA", desc: "LoRA avec décomposition améliorée" },
    full: { label: "Full", desc: "Entraînement complet — VRAM élevée" },
};
const HPARAM_INFO = {
    lora_rank: { label: "LoRA Rank", desc: "Dimension de l'espace d'adaptation" },
    lora_alpha: { label: "LoRA Alpha", desc: "Facteur de mise à l'échelle" },
    lora_dropout: { label: "Dropout", desc: "Taux de désactivation pour éviter le surapprentissage" },
    learning_rate: { label: "Learning Rate", desc: "Vitesse d'adaptation — trop élevé = instable" },
    num_train_epochs: { label: "Epochs", desc: "Passages complets sur le dataset" },
    per_device_train_batch_size: { label: "Batch Size", desc: "Exemples traités simultanément" },
    gradient_accumulation_steps: { label: "Grad. Accum.", desc: "Simule un batch plus grand" },
    warmup_ratio: { label: "Warmup", desc: "Proportion de montée en régime" },
    weight_decay: { label: "Weight Decay", desc: "Régularisation L2" },
};
const TASKS = [
    { id: "question-answering", icon: "🔍", label: "Question & Réponse", desc: "Répondre à des questions depuis vos documents" },
    { id: "code-generation", icon: "💻", label: "Génération de code", desc: "Produire ou compléter du code source" },
    { id: "summarization", icon: "📝", label: "Résumé de texte", desc: "Condenser des documents longs en résumés clairs" },
    { id: "chatbot", icon: "🤖", label: "Chatbot", desc: "Assistant conversationnel polyvalent" },
];
// ── Helpers ───────────────────────────────────────────────────────────────────
function tierColor(tier) {
    if (tier === "good")
        return "var(--ok)";
    if (tier === "acceptable")
        return "var(--warn)";
    return "var(--err)";
}
function classifyLog(line) {
    const l = line.toLowerCase();
    if (l.includes("error") || l.includes("failed") || l.includes("exception"))
        return "log-line-error";
    if (l.includes("warn"))
        return "log-line-warn";
    if (l.includes("done") || l.includes("finish") || l.includes("success") || l.includes("✓"))
        return "log-line-success";
    return "log-line-info";
}
// ── Pipeline step detection ───────────────────────────────────────────────────
function stepIcon(msg) {
    const m = msg.toLowerCase();
    if (m.includes("profil") || (m.includes("fichier") && m.includes("analysé")))
        return "📂";
    if (m.includes("compatib") || (m.includes("domaine") && m.includes("vérif")))
        return "🔍";
    if (m.includes("stratégie d'évaluation") || m.includes("métrique principal"))
        return "📊";
    if (m.includes("intention") || m.includes("tâche :"))
        return "🎯";
    if (m.includes("faisabilité"))
        return "✅";
    if (m.includes("nouveau modèle sélectionné"))
        return "🔄";
    if (m.includes("sélectionné") && m.includes("modèle"))
        return "🤖";
    if (m.includes("données prêtes") || m.includes("paires qa générées"))
        return "📚";
    if (m.includes("diagnostic itération"))
        return "🔬";
    if (m.includes("dataset insuffisant") || m.includes("générer automatiquement"))
        return "⚠️";
    if (m.includes("entraînem") && (m.includes("lancement") || m.includes("terminé") || m.includes("démarré")))
        return "⚙️";
    if (m.includes("évaluation itération") || m.includes("évaluation terminée"))
        return "📈";
    if (m.includes("incompatibil") || m.includes("bloqué"))
        return "⛔";
    return "▸";
}
function stepTitle(msg) {
    const m = msg.toLowerCase();
    if (m.includes("profil") && (m.includes("terminé") || m.includes("analysé")))
        return "Analyse des fichiers";
    if (m.includes("compatib"))
        return "Compatibilité des domaines";
    if (m.includes("stratégie d'évaluation"))
        return "Stratégie d'évaluation détectée";
    if (m.includes("intention") || (m.includes("tâche") && m.includes(":")))
        return "Extraction de l'intention";
    if (m.includes("faisabilité"))
        return "Vérification de faisabilité";
    if (m.includes("nouveau modèle sélectionné"))
        return "Resélection du modèle";
    if (m.includes("sélectionné") && m.includes("modèle"))
        return "Sélection du modèle";
    if (m.includes("données prêtes") || m.includes("paires qa"))
        return "Préparation des données";
    if (m.includes("diagnostic itération"))
        return "Diagnostic & action corrective";
    if (m.includes("dataset insuffisant") || m.includes("générer automatiquement"))
        return "Augmentation du dataset";
    if (m.includes("entraînem") && m.includes("lancement"))
        return "Lancement de l'entraînement";
    if (m.includes("entraînem") && m.includes("terminé"))
        return "Entraînement terminé";
    if (m.includes("entraînem"))
        return "Entraînement";
    if (m.includes("évaluation itération"))
        return "Évaluation du modèle";
    if (m.includes("incompatibil") || m.includes("bloqué"))
        return "Blocage — domaines incompatibles";
    return "Étape pipeline";
}
// ── Non-expert explanations per step ─────────────────────────────────────────
const STEP_GUIDE = {
    "Analyse des fichiers": "Le pipeline lit vos documents pour comprendre leur format (PDF, CSV, texte), leur langue et leur taille.",
    "Compatibilité des domaines": "On vérifie que tous vos fichiers parlent du même sujet (ex. tous médicaux). Des fichiers de domaines différents produiraient un modèle incohérent.",
    "Extraction de l'intention": "L'IA analyse votre objectif en langage naturel pour déterminer le type de tâche : questions-réponses, classification, extraction d'entités…",
    "Stratégie d'évaluation détectée": "En fonction de votre tâche, le pipeline choisit les bons critères de mesure. Une classification n'est pas évaluée comme une génération de texte.",
    "Vérification de faisabilité": "On s'assure que vos données contiennent assez de texte pour entraîner un modèle. Un dataset trop petit donnerait un modèle inutilisable.",
    "Sélection du modèle": "L'IA choisit le meilleur modèle pré-entraîné selon votre GPU, votre tâche et votre objectif (vitesse vs qualité).",
    "Resélection du modèle": "Le modèle précédent n'a pas donné de bons résultats. L'orchestrateur en essaie un autre mieux adapté.",
    "Préparation des données": "Vos documents sont transformés en paires question-réponse que le modèle va apprendre. C'est l'équivalent de créer un manuel d'exercices.",
    "Augmentation du dataset": "Pas assez d'exemples d'entraînement. Le pipeline génère automatiquement des questions supplémentaires à partir de vos documents.",
    "Lancement de l'entraînement": "Le modèle apprend à partir de vos exemples. C'est l'étape la plus longue — le modèle ajuste ses paramètres pour répondre comme vos documents.",
    "Entraînement terminé": "Le modèle a fini d'apprendre. Un adaptateur (fichier léger) a été créé — il encode tout ce que le modèle a appris de vos données.",
    "Entraînement": "Entraînement en cours — le modèle s'adapte à vos données.",
    "Évaluation du modèle": "On mesure la qualité du modèle sur des exemples qu'il n'a jamais vus pendant l'entraînement. Plus le score est proche de 1, meilleur est le modèle.",
    "Diagnostic & action corrective": "Le score n'est pas assez bon. L'orchestrateur identifie la cause (trop peu de données ? mauvais hyperparamètres ?) et prend une action corrective.",
};
// ── SVG Loss Chart ─────────────────────────────────────────────────────────────
function LossChart({ history }) {
    const W = 900, H = 180, PADL = 52, PADB = 30, PADR = 20, PADT = 16;
    const points = history.points;
    if (points.length < 2)
        return _jsx("p", { className: "muted-text", children: "Pas assez de points pour tracer la courbe." });
    const steps = points.map(p => p.step);
    const minStep = Math.min(...steps), maxStep = Math.max(...steps);
    const trainPts = points.filter(p => p.train_loss != null);
    const evalPts = points.filter(p => p.eval_loss != null);
    const allLoss = [...trainPts.map(p => p.train_loss), ...evalPts.map(p => p.eval_loss)];
    const minLoss = Math.min(...allLoss) * 0.9;
    const maxLoss = Math.max(...allLoss) * 1.05;
    const cx = (s) => PADL + ((s - minStep) / (maxStep - minStep || 1)) * (W - PADL - PADR);
    const cy = (l) => PADT + (1 - (l - minLoss) / (maxLoss - minLoss || 1)) * (H - PADT - PADB);
    const poly = (pts, key) => pts.map(p => `${cx(p.step).toFixed(1)},${cy(p[key]).toFixed(1)}`).join(" ");
    const yTicks = Array.from({ length: 5 }, (_, i) => minLoss + (i / 4) * (maxLoss - minLoss));
    return (_jsxs("svg", { className: "loss-chart-svg", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet", style: { height: 200 }, children: [yTicks.map((v, i) => (_jsxs("g", { children: [_jsx("line", { x1: PADL, y1: cy(v), x2: W - PADR, y2: cy(v), stroke: "#D0D9E8", strokeWidth: "1" }), _jsx("text", { x: PADL - 6, y: cy(v) + 4, textAnchor: "end", fontSize: "10", fill: "#617498", children: v.toFixed(2) })] }, i))), trainPts.length >= 2 && _jsx("polyline", { points: poly(trainPts, "train_loss"), fill: "none", stroke: "#D9467A", strokeWidth: "2", strokeLinejoin: "round" }), evalPts.length >= 2 && _jsx("polyline", { points: poly(evalPts, "eval_loss"), fill: "none", stroke: "#5B7FAE", strokeWidth: "2.5", strokeLinejoin: "round" }), evalPts.map((p, i) => _jsx("circle", { cx: cx(p.step), cy: cy(p.eval_loss), r: "3.5", fill: "#5B7FAE" }, i)), _jsx("line", { x1: PADL, y1: H - PADB, x2: W - PADR, y2: H - PADB, stroke: "#D0D9E8", strokeWidth: "1.5" }), _jsx("text", { x: (W - PADL - PADR) / 2 + PADL, y: H - 4, textAnchor: "middle", fontSize: "10", fill: "#617498", children: "Steps" })] }));
}
// ── Training Images ───────────────────────────────────────────────────────────
function TrainingImagesSection({ images }) {
    const [lightbox, setLightbox] = useState(null);
    if (!images.length)
        return null;
    return (_jsxs("div", { className: "loss-chart-wrap", style: { marginBottom: 0 }, children: [_jsx("div", { className: "loss-chart-title", children: "\uD83D\uDCF8 Graphiques d'entra\u00EEnement" }), _jsx("div", { className: "images-grid", children: images.map(img => (_jsxs("div", { className: "image-card", children: [_jsx("img", { src: img.url, alt: img.name, onClick: () => setLightbox(img.url), loading: "lazy" }), _jsx("div", { className: "image-card-label", children: img.name.replace(/\.(png|jpg|jpeg)$/i, "") })] }, img.name))) }), lightbox && (_jsx("div", { className: "lightbox-overlay", onClick: () => setLightbox(null), children: _jsx("img", { src: lightbox, alt: "zoom" }) }))] }));
}
// ── Improvement Log ───────────────────────────────────────────────────────────
function ImprovementLog({ log }) {
    if (!log.length)
        return null;
    return (_jsxs("div", { children: [_jsx("div", { className: "loss-chart-title", style: { marginBottom: 10 }, children: "\uD83D\uDD04 Historique des it\u00E9rations" }), _jsx("div", { className: "improvement-log", children: log.map((entry, i) => {
                    const tier = String(entry.quality_tier ?? "poor");
                    const primary = String(entry.primary_metric ?? "rouge1").toUpperCase().replace(/_/g, "-");
                    const primaryVal = Number(entry.primary_metric_value ?? entry.rouge1 ?? 0);
                    const prevEntry = i > 0 ? log[i - 1] : null;
                    const prevVal = prevEntry ? Number(prevEntry.primary_metric_value ?? prevEntry.rouge1 ?? 0) : null;
                    const delta = prevVal !== null ? primaryVal - prevVal : null;
                    const overrides = entry.hparam_overrides;
                    const modelShort = String(entry.model_id ?? "—").split("/").pop() ?? "—";
                    return (_jsxs("div", { className: "iter-row", style: { flexDirection: "column", alignItems: "stretch", gap: 10 }, children: [_jsxs("div", { style: { display: "flex", alignItems: "center", gap: 12 }, children: [_jsx("div", { className: "iter-num", children: Number(entry.iteration ?? i) + 1 }), _jsxs("div", { className: "iter-metrics", children: [_jsxs("div", { className: "iter-metric", children: [_jsx("span", { className: "iter-metric-label", children: primary }), _jsxs("span", { className: `iter-metric-value iter-tier-${tier}`, style: { display: "flex", alignItems: "center", gap: 4 }, children: [primaryVal.toFixed(3), delta !== null && (_jsx("span", { style: { fontSize: 11, fontWeight: 700, color: delta > 0.001 ? "var(--ok)" : delta < -0.001 ? "var(--err)" : "var(--muted)" }, children: delta > 0.001 ? `↑+${delta.toFixed(3)}` : delta < -0.001 ? `↓${delta.toFixed(3)}` : "→" }))] })] }), _jsxs("div", { className: "iter-metric", children: [_jsx("span", { className: "iter-metric-label", children: "Paires" }), _jsx("span", { className: "iter-metric-value", style: { color: "var(--muted)", fontSize: 13 }, children: String(entry.n_pairs ?? "—") })] }), _jsxs("div", { className: "iter-metric", children: [_jsx("span", { className: "iter-metric-label", children: "Mod\u00E8le" }), _jsx("span", { className: "iter-metric-value", style: { color: "var(--muted)", fontSize: 11, maxWidth: 130, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, children: modelShort })] })] }), _jsx("div", { style: { marginLeft: "auto" }, children: _jsx("span", { className: "badge", style: {
                                                background: tier === "good" ? "#E6F7F1" : tier === "acceptable" ? "#FEF8E7" : "#FDECEA",
                                                color: tierColor(tier), borderColor: tierColor(tier),
                                            }, children: tier === "good" ? "Bon" : tier === "acceptable" ? "Acceptable" : "Insuffisant" }) })] }), overrides && Object.keys(overrides).filter(k => k !== "_source").length > 0 && (_jsxs("div", { style: {
                                    marginLeft: 40, padding: "7px 12px", borderRadius: 7,
                                    background: "#EDE9FE", border: "1px solid #C4B5FD",
                                    fontSize: 12, color: "#5B21B6", display: "flex", flexWrap: "wrap", gap: "4px 14px", alignItems: "center",
                                }, children: [_jsx("span", { style: { fontWeight: 700, marginRight: 2 }, children: "\uD83D\uDD27 Hyperparam\u00E8tres ajust\u00E9s :" }), Object.entries(overrides).filter(([k]) => k !== "_source").map(([k, v]) => (_jsxs("span", { style: { display: "inline-flex", alignItems: "center", gap: 4 }, children: [_jsx("span", { style: { fontWeight: 600 }, children: HPARAM_INFO[k]?.label ?? k }), _jsx("span", { style: { color: "var(--primary)" }, children: "\u2192" }), _jsx("code", { style: { background: "#fff", padding: "1px 5px", borderRadius: 4, fontSize: 11, color: "var(--primary)" }, children: String(v) })] }, k)))] }))] }, i));
                }) })] }));
}
// ── Metric Row ────────────────────────────────────────────────────────────────
function MetricRow({ label, desc, value, metric, thresholdGood, thresholdAcceptable, noviceHint }) {
    const goodFallback = metric === "bleu4" ? 0.14 : metric === "rouge2" ? 0.22 : metric === "rougeL" ? 0.42 : 0.45;
    const okFallback = metric === "bleu4" ? 0.05 : metric === "rouge2" ? 0.12 : metric === "rougeL" ? 0.25 : 0.28;
    const good = thresholdGood ?? goodFallback;
    const ok = thresholdAcceptable ?? okFallback;
    const max = metric === "bleu4" ? 0.35 : 1.0;
    const color = value >= good ? "var(--ok)" : value >= ok ? "var(--warn)" : "var(--err)";
    const fillW = `${Math.min(100, (value / max) * 100).toFixed(1)}%`;
    const goodW = `${Math.min(100, (good / max) * 100).toFixed(1)}%`;
    return (_jsxs("div", { className: "metric-row", children: [_jsxs("div", { className: "metric-header", children: [_jsx("span", { className: "metric-label", children: label }), _jsxs("div", { children: [_jsx("span", { className: "metric-value", style: { color }, children: value.toFixed(3) }), thresholdGood && _jsxs("span", { className: "metric-threshold", children: ["seuil : ", thresholdGood.toFixed(2)] })] })] }), _jsxs("div", { className: "metric-bar-bg", style: { position: "relative" }, children: [_jsx("div", { className: "metric-bar-fill", style: { width: fillW, background: color } }), _jsx("div", { className: "metric-threshold-line", style: { left: goodW }, title: `Seuil "bon" = ${good.toFixed(2)}` })] }), _jsx("p", { className: "metric-desc", children: desc }), noviceHint && (_jsxs("div", { style: { marginTop: 6, fontSize: 12, color: "var(--primary)", background: "#EDE9FE", borderRadius: 6, padding: "5px 9px" }, children: ["\uD83C\uDF93 ", noviceHint] }))] }));
}
// ── InfoItem ──────────────────────────────────────────────────────────────────
function InfoItem({ icon, label, value, sub, mono }) {
    return (_jsxs("div", { className: "info-item", children: [_jsx("div", { className: "info-icon-wrap", children: icon }), _jsxs("div", { children: [_jsx("div", { className: "info-label", children: label }), _jsx("div", { className: `info-value ${mono ? "mono" : ""}`, children: value }), sub && _jsx("div", { className: "info-sub", children: sub })] })] }));
}
// ── Training Dashboard ────────────────────────────────────────────────────────
function TrainingDashboard({ report, runId, lossHistory, trainingImages, isExpert, }) {
    const [hparamsOpen, setHparamsOpen] = useState(false);
    const peft = PEFT_LABELS[report.peft_method] ?? { label: report.peft_method, desc: "" };
    const success = report.result?.success;
    const evalLoss = report.result?.metrics?.eval_loss;
    const trainLoss = report.result?.metrics?.loss;
    return (_jsxs("div", { className: "card", children: [_jsxs("div", { className: "card-header", children: [_jsxs("h2", { children: [_jsx("span", { className: "card-icon", children: "\uD83E\uDD16" }), "Entra\u00EEnement du mod\u00E8le"] }), _jsx("span", { className: `badge ${success ? "badge-done" : "badge-error"}`, children: success ? "Réussi" : "Échoué" })] }), !isExpert && (_jsxs("div", { className: "novice-guide-box", children: [_jsx("strong", { children: "Qu'est-ce qui s'est pass\u00E9 ?" }), " Le mod\u00E8le a appris de vos donn\u00E9es gr\u00E2ce \u00E0 la m\u00E9thode PEFT. Au lieu de r\u00E9entra\u00EEner tout le mod\u00E8le (tr\u00E8s co\u00FBteux), on ajoute un petit adaptateur qui encode les nouvelles connaissances. C'est ce fichier que vous pouvez t\u00E9l\u00E9charger."] })), _jsxs("div", { className: "info-grid", children: [_jsx(InfoItem, { icon: "\uD83E\uDD16", label: "Mod\u00E8le", value: report.model_id, mono: true }), _jsx(InfoItem, { icon: "\u2699\uFE0F", label: "M\u00E9thode", value: peft.label, sub: peft.desc }), _jsx(InfoItem, { icon: "\uD83D\uDCDA", label: "Paires", value: String(report.n_pairs), sub: !isExpert ? "exemples d'entraînement utilisés" : undefined }), _jsx(InfoItem, { icon: "\uD83D\uDD01", label: "Epochs", value: String(report.hparams?.num_train_epochs ?? "—"), sub: !isExpert ? "passages sur le dataset" : undefined }), evalLoss != null && _jsx(InfoItem, { icon: "\uD83D\uDCC9", label: "Eval Loss", value: evalLoss.toFixed(4), sub: "Plus bas = meilleur apprentissage" }), trainLoss != null && _jsx(InfoItem, { icon: "\uD83D\uDCCA", label: "Train Loss", value: trainLoss.toFixed(4) })] }), report.peft_rationale && (_jsxs("div", { className: "rationale", children: [_jsx("span", { className: "rationale-icon", children: "\uD83D\uDCA1" }), _jsxs("div", { children: [_jsx("div", { style: { fontWeight: 700, marginBottom: 3, fontSize: 12 }, children: "Pourquoi cette m\u00E9thode PEFT ?" }), report.peft_rationale] })] })), report.hparam_rationale && (_jsxs("div", { className: "rationale", style: { borderLeftColor: "var(--ok)", background: "#F0FDF4", border: "1px solid #BBF7D0" }, children: [_jsx("span", { className: "rationale-icon", children: "\uD83C\uDF9B\uFE0F" }), _jsxs("div", { children: [_jsxs("div", { style: { fontWeight: 700, marginBottom: 3, fontSize: 12, color: "var(--ok)" }, children: ["Pourquoi ces hyperparam\u00E8tres ?", report.hparam_source && (_jsxs("span", { style: { marginLeft: 8, fontSize: 10, fontWeight: 400, color: "var(--muted)" }, children: ["source : ", report.hparam_source] }))] }), _jsx("span", { style: { color: "#166534" }, children: report.hparam_rationale })] })] })), lossHistory && lossHistory.points.length >= 2 && (_jsxs("div", { className: "loss-chart-wrap", children: [_jsxs("div", { className: "loss-chart-title", children: ["\uD83D\uDCC8 Courbes de loss", !isExpert && (_jsx("span", { style: { fontWeight: 400, fontSize: 11, color: "var(--muted)", marginLeft: 8 }, children: "\u2014 une courbe qui descend = le mod\u00E8le apprend bien" })), _jsxs("div", { className: "loss-chart-legend", children: [_jsxs("div", { className: "legend-item", children: [_jsx("div", { className: "legend-dot", style: { background: "#D9467A" } }), "Train loss"] }), _jsxs("div", { className: "legend-item", children: [_jsx("div", { className: "legend-dot", style: { background: "#5B7FAE" } }), "Eval loss"] })] })] }), _jsx(LossChart, { history: lossHistory })] })), _jsx(TrainingImagesSection, { images: trainingImages }), _jsx("button", { className: "toggle-btn", onClick: () => setHparamsOpen(o => !o), children: hparamsOpen ? "▲ Masquer les hyperparamètres" : "▼ Voir les hyperparamètres utilisés" }), hparamsOpen && (_jsxs(_Fragment, { children: [!isExpert && (_jsx("div", { className: "novice-guide-box", style: { marginBottom: 12 }, children: "Les hyperparam\u00E8tres sont les r\u00E9glages de l'entra\u00EEnement : \u00E0 quelle vitesse le mod\u00E8le apprend, combien de fois il voit les donn\u00E9es, etc. Ils ont \u00E9t\u00E9 choisis automatiquement selon votre GPU et votre dataset." })), _jsxs("table", { className: "hparam-table", children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "Param\u00E8tre" }), _jsx("th", { children: "Valeur" }), _jsx("th", { children: "Description" })] }) }), _jsx("tbody", { children: Object.entries(report.hparams).map(([k, v]) => {
                                    const info = HPARAM_INFO[k] ?? { label: k, desc: "" };
                                    return (_jsxs("tr", { children: [_jsx("td", { className: "mono", children: info.label }), _jsx("td", { className: "mono val", children: String(v) }), _jsx("td", { className: "muted", children: info.desc })] }, k));
                                }) })] })] })), _jsxs("div", { className: "card-actions", children: [_jsx("a", { href: modelDownloadUrl(runId), download: true, children: _jsx("button", { className: "action-btn action-primary", children: "\u2B07 T\u00E9l\u00E9charger l'adaptateur" }) }), _jsx("a", { href: reportDownloadUrl(runId), download: true, children: _jsx("button", { className: "action-btn action-secondary", children: "\u2B07 Rapport JSON" }) })] })] }));
}
// ── Evaluation Card ───────────────────────────────────────────────────────────
function EvalCard({ evalState, evalReport, onRunEval, isExpert }) {
    const metrics = evalReport?.metrics;
    const tier = metrics?.quality_tier;
    const qualityBadge = (() => {
        if (!metrics)
            return null;
        if (tier === "good")
            return { label: "Bonne qualité", color: "var(--ok)" };
        if (tier === "acceptable")
            return { label: "Qualité acceptable", color: "var(--warn)" };
        return { label: "Qualité insuffisante", color: "var(--err)" };
    })();
    return (_jsxs("div", { className: "card", children: [_jsxs("div", { className: "card-header", children: [_jsxs("h2", { children: [_jsx("span", { className: "card-icon", children: "\uD83D\uDCCA" }), "\u00C9valuation du mod\u00E8le"] }), qualityBadge && (_jsx("span", { className: "quality-badge", style: { color: qualityBadge.color, borderColor: qualityBadge.color }, children: qualityBadge.label }))] }), evalState === "idle" && (_jsxs("div", { className: "eval-prompt", children: [!isExpert ? (_jsxs("div", { className: "novice-guide-box", children: [_jsx("strong", { children: "Qu'est-ce que l'\u00E9valuation ?" }), " On donne au mod\u00E8le des questions qu'il n'a jamais vues pendant l'entra\u00EEnement. On compare ses r\u00E9ponses aux r\u00E9ponses correctes pour mesurer sa qualit\u00E9. Un score de 1.0 = r\u00E9ponses parfaites. En pratique, 0.4+ est un bon r\u00E9sultat."] })) : (_jsx("p", { children: "L'\u00E9valuation mesure la qualit\u00E9 des r\u00E9ponses g\u00E9n\u00E9r\u00E9es par le mod\u00E8le fine-tun\u00E9 par rapport aux r\u00E9ponses de r\u00E9f\u00E9rence (ROUGE, BLEU, LLM-as-judge)." })), _jsx("button", { className: "action-btn action-accent", onClick: onRunEval, children: "\u25B6 Lancer l'\u00E9valuation" })] })), evalState === "loading" && (_jsxs("div", { className: "eval-loading", children: [_jsx("div", { className: "spinner" }), _jsxs("p", { children: ["\u00C9valuation en cours \u2014 g\u00E9n\u00E9ration des r\u00E9ponses sur le jeu de test\u2026", _jsx("br", {}), _jsx("span", { className: "muted-text", children: "Cela peut prendre plusieurs minutes." })] })] })), evalState === "error" && (_jsx("div", { className: "error-box", children: "\u26A0 L'\u00E9valuation a \u00E9chou\u00E9. Consultez les logs du serveur." })), metrics && (_jsxs(_Fragment, { children: [_jsx("div", { className: "quality-summary", children: metrics.quality_summary }), _jsxs("div", { className: "metrics-list", children: [_jsx(MetricRow, { label: "ROUGE-1", metric: "rouge1", value: metrics.rouge1, thresholdGood: metrics.threshold_rouge1_good, thresholdAcceptable: metrics.threshold_rouge1_acceptable, desc: "Proportion des mots de la r\u00E9f\u00E9rence retrouv\u00E9s dans la r\u00E9ponse du mod\u00E8le", noviceHint: !isExpert ? `Score actuel : ${metrics.rouge1.toFixed(2)} — ${metrics.rouge1 >= (metrics.threshold_rouge1_good ?? 0.45) ? "✅ Bon résultat !" : metrics.rouge1 >= (metrics.threshold_rouge1_acceptable ?? 0.28) ? "⚠ Résultat acceptable, peut être amélioré" : "❌ Insuffisant, le modèle a besoin de plus de données ou d'entraînement"}` : undefined }), _jsx(MetricRow, { label: "ROUGE-2", metric: "rouge2", value: metrics.rouge2, thresholdGood: metrics.threshold_rouge2_good, desc: "Proportion des bigrammes (paires de mots cons\u00E9cutifs) en commun avec la r\u00E9f\u00E9rence" }), _jsx(MetricRow, { label: "ROUGE-L", metric: "rougeL", value: metrics.rougeL, thresholdGood: metrics.threshold_rougeL_good, desc: "Qualit\u00E9 structurelle \u2014 les mots apparaissent-ils dans le bon ordre ?" }), _jsx(MetricRow, { label: "BLEU-4", metric: "bleu4", value: metrics.bleu4, thresholdGood: metrics.threshold_bleu4_good, desc: "Pr\u00E9cision des s\u00E9quences de 4 mots \u2014 mesure stricte, souvent bas m\u00EAme pour un bon mod\u00E8le", noviceHint: !isExpert ? "Ce score est naturellement bas. Un BLEU-4 > 0.10 est déjà un bon signe." : undefined })] }), metrics.llm_judge_accuracy != null && (_jsxs("div", { className: "llm-judge", children: [_jsxs("h3", { children: ["\u00C9valuation par LLM-as-judge (", metrics.llm_judge_n, " exemples)"] }), _jsxs("div", { className: "judge-scores", children: [_jsxs("div", { className: "judge-score", children: [_jsx("span", { className: "judge-label", children: "Pr\u00E9cision" }), _jsxs("span", { className: "judge-value", children: [metrics.llm_judge_accuracy.toFixed(1), _jsx("span", { style: { fontSize: 14, color: "var(--muted)" }, children: "/5" })] })] }), _jsxs("div", { className: "judge-score", children: [_jsx("span", { className: "judge-label", children: "Compl\u00E9tude" }), _jsxs("span", { className: "judge-value", children: [metrics.llm_judge_completeness?.toFixed(1), _jsx("span", { style: { fontSize: 14, color: "var(--muted)" }, children: "/5" })] })] })] }), !isExpert && (_jsx("p", { style: { fontSize: 12.5, color: "var(--muted)", marginTop: 6 }, children: "Un autre mod\u00E8le IA a not\u00E9 les r\u00E9ponses de votre mod\u00E8le sur 5. C'est une \u00E9valuation plus proche de ce que ressentirait un humain." }))] })), _jsxs("div", { className: "eval-meta muted-text", children: ["\u00C9valu\u00E9 sur ", metrics.n_test, " exemples du jeu de test"] })] }))] }));
}
// ── Live Pipeline Feed ────────────────────────────────────────────────────────
function LivePipelineFeed({ logs, status, isExpert }) {
    const pipelineLogs = logs.filter(l => l.startsWith("[PIPELINE]") || /^\d{2}:\d{2}:\d{2}\s+(ERROR|CRITICAL)\b/.test(l));
    if (pipelineLogs.length === 0) {
        return (_jsx("div", { style: { padding: "24px 0", color: "var(--muted)", textAlign: "center", fontSize: 14 }, children: status === "running" ? (_jsxs(_Fragment, { children: [_jsx("div", { className: "spinner", style: { width: 18, height: 18, border: "2.5px solid rgba(124,58,237,0.2)", borderTopColor: "var(--primary)", display: "inline-block", marginRight: 10, verticalAlign: "middle" } }), "D\u00E9marrage du pipeline\u2026"] })) : "Aucune étape pipeline pour l'instant." }));
    }
    return (_jsx("div", { style: { display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }, children: pipelineLogs.map((line, i) => {
            const isError = /^\d{2}:\d{2}:\d{2}\s+(ERROR|CRITICAL)\b/.test(line);
            const msg = line.startsWith("[PIPELINE]") ? line.replace(/^\[PIPELINE\]\s*/, "") : line;
            const icon = isError ? "✕" : stepIcon(msg);
            const title = isError ? "Erreur" : stepTitle(msg);
            const guide = !isExpert ? STEP_GUIDE[title] : undefined;
            const isLast = i === pipelineLogs.length - 1;
            const isRunning = isLast && status === "running";
            return (_jsxs("div", { style: {
                    display: "flex", gap: 12, alignItems: "flex-start", padding: "11px 14px", borderRadius: 10,
                    background: isError ? "#FFF5F5" : isRunning ? "#F5F0FF" : "var(--panel)",
                    border: `1px solid ${isError ? "#FFCCCC" : isRunning ? "var(--primary)" : "var(--border)"}`,
                    boxShadow: isRunning ? "0 0 0 3px rgba(124,58,237,0.08)" : "none",
                }, children: [_jsx("div", { style: {
                            width: 36, height: 36, borderRadius: "50%", flexShrink: 0,
                            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16,
                            background: isError ? "#FDECEA" : isRunning ? "#EDE9FE" : "var(--surface)",
                            border: `1.5px solid ${isError ? "#FCA5A5" : isRunning ? "var(--primary)" : "var(--border)"}`,
                        }, children: isRunning
                            ? _jsx("div", { className: "spinner", style: { width: 14, height: 14, border: "2px solid rgba(124,58,237,0.2)", borderTopColor: "var(--primary)" } })
                            : _jsx("span", { children: icon }) }), _jsxs("div", { style: { flex: 1, minWidth: 0 }, children: [_jsxs("div", { style: { fontWeight: 700, fontSize: 13, marginBottom: 3, color: isError ? "var(--err)" : "var(--text)", display: "flex", alignItems: "center", gap: 8 }, children: [title, isRunning && _jsx("span", { style: { fontSize: 11, color: "var(--primary)", fontWeight: 500, fontStyle: "italic" }, children: "en cours\u2026" }), !isRunning && !isError && _jsx("span", { style: { fontSize: 11, color: "var(--ok)", fontWeight: 600 }, children: "\u2713" })] }), _jsx("div", { style: { fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6, wordBreak: "break-word" }, children: msg }), guide && (_jsxs("div", { style: { marginTop: 7, fontSize: 12, color: "var(--primary)", background: "#EDE9FE", borderRadius: 6, padding: "5px 10px", lineHeight: 1.55 }, children: ["\uD83C\uDF93 ", _jsx("strong", { children: "Pour comprendre :" }), " ", guide] }))] })] }, i));
        }) }));
}
// ── Decision Journal Card ─────────────────────────────────────────────────────
function DecisionJournalCard({ journal, isExpert }) {
    const [open, setOpen] = useState(true);
    const okCount = journal.filter(e => e.statut === "ok").length;
    const warnCount = journal.filter(e => e.statut === "warn").length;
    const errCount = journal.filter(e => e.statut === "err").length;
    const infoCount = journal.filter(e => e.statut === "info").length;
    const statusColor = (s) => s === "ok" ? "var(--ok)" : s === "warn" ? "var(--warn)" : s === "err" ? "var(--err)" : "var(--primary)";
    const statusLabel = (s) => s === "ok" ? "OK" : s === "warn" ? "Attention" : s === "err" ? "Erreur" : "Info";
    const statusBg = (s) => s === "ok" ? "#E6F7F1" : s === "warn" ? "#FEF8E7" : s === "err" ? "#FDECEA" : "#EDE9FE";
    return (_jsxs("div", { className: "card", children: [_jsxs("div", { className: "card-header", children: [_jsxs("h2", { children: [_jsx("span", { className: "card-icon", children: "\uD83D\uDCCB" }), isExpert ? "Rapport de décisions" : "Ce qui s'est passé — étape par étape"] }), _jsxs("div", { style: { display: "flex", gap: 6, flexWrap: "wrap" }, children: [okCount > 0 && _jsxs("span", { className: "badge", style: { background: "#DCFCE7", color: "var(--ok)", borderColor: "#86EFAC" }, children: [okCount, " OK"] }), warnCount > 0 && _jsxs("span", { className: "badge", style: { background: "#FEF8E7", color: "var(--warn)", borderColor: "#FCD34D" }, children: [warnCount, " Attention"] }), errCount > 0 && _jsxs("span", { className: "badge", style: { background: "#FEE2E2", color: "var(--err)", borderColor: "#FCA5A5" }, children: [errCount, " Erreur"] }), infoCount > 0 && _jsxs("span", { className: "badge badge-running", children: [infoCount, " Info"] })] })] }), _jsx("p", { className: "muted-text", style: { marginBottom: 14 }, children: isExpert
                    ? "Chaque décision prise par l'orchestrateur avec sa justification technique."
                    : "Voici toutes les décisions que l'intelligence artificielle a prises pour construire votre modèle, avec une explication pour chacune." }), _jsx("button", { className: "toggle-btn", onClick: () => setOpen(o => !o), children: open ? "▲ Réduire" : "▼ Voir le rapport complet" }), open && (_jsxs("div", { style: { marginTop: 4, position: "relative", paddingLeft: 28 }, children: [_jsx("div", { style: { position: "absolute", left: 11, top: 8, bottom: 24, width: 2, background: "var(--border)", borderRadius: 2 } }), _jsx("div", { style: { display: "flex", flexDirection: "column", gap: 6 }, children: journal.map((entry, i) => (_jsxs("div", { style: { display: "flex", alignItems: "flex-start", gap: 12 }, children: [_jsx("div", { style: {
                                        width: 14, height: 14, borderRadius: "50%", flexShrink: 0, marginTop: 14,
                                        background: statusColor(entry.statut), border: "2.5px solid var(--bg)",
                                        boxShadow: `0 0 0 2px ${statusColor(entry.statut)}`,
                                        position: "relative", zIndex: 1, marginLeft: -5,
                                    } }), _jsxs("div", { style: { flex: 1, padding: "11px 14px 12px", borderRadius: 10, background: "var(--surface)", border: "1px solid var(--border)", marginBottom: 4 }, children: [_jsxs("div", { style: { display: "flex", alignItems: "center", gap: 8, marginBottom: 6, flexWrap: "wrap" }, children: [_jsx("span", { style: { fontWeight: 700, fontSize: 13 }, children: entry.étape }), _jsx("span", { style: { fontSize: 10, padding: "2px 8px", borderRadius: 20, fontWeight: 700, background: statusBg(entry.statut), color: statusColor(entry.statut), border: `1px solid ${statusColor(entry.statut)}44`, textTransform: "uppercase", letterSpacing: "0.05em" }, children: statusLabel(entry.statut) })] }), _jsx("div", { style: { fontSize: 13.5, fontWeight: 600, marginBottom: 6, lineHeight: 1.45 }, children: entry.décision }), _jsxs("div", { style: { fontSize: 12.5, color: "var(--muted)", lineHeight: 1.65, paddingLeft: 10, borderLeft: `2px solid ${statusColor(entry.statut)}44` }, children: ["\uD83D\uDCA1 ", entry.justification] })] })] }, i))) })] }))] }));
}
// ── Main Pipeline Page ────────────────────────────────────────────────────────
export default function PipelinePage({ user, onHistory, }) {
    const isExpert = user.isExpert;
    const [files, setFiles] = useState([]);
    const [goal, setGoal] = useState("");
    const [language, setLanguage] = useState("fr");
    const [objective, setObjective] = useState("balanced");
    const [gpuVram, setGpuVram] = useState(4);
    const [gpuInfo, setGpuInfo] = useState(null);
    const [task, setTask] = useState("");
    const [logMode, setLogMode] = useState("summary");
    const [status, setStatus] = useState("idle");
    const [logs, setLogs] = useState([]);
    const [logFilter, setLogFilter] = useState("all");
    const [error, setError] = useState("");
    const [runId, setRunId] = useState("");
    const [dataset, setDataset] = useState(null);
    const [trainingReport, setTrainingReport] = useState(null);
    const [evalReport, setEvalReport] = useState(null);
    const [evalState, setEvalState] = useState("idle");
    const [lossHistory, setLossHistory] = useState(null);
    const [trainingImages, setTrainingImages] = useState([]);
    const [improvementLog, setImprovementLog] = useState([]);
    const [decisionJournal, setDecisionJournal] = useState([]);
    const [sseRetrying, setSseRetrying] = useState(false);
    const [confirmPayload, setConfirmPayload] = useState(null);
    const [confirmBusy, setConfirmBusy] = useState(false);
    const [blockedReason, setBlockedReason] = useState("");
    const [showAdditionalUpload, setShowAdditionalUpload] = useState(false);
    const [additionalFiles, setAdditionalFiles] = useState([]);
    const [uploadBusy, setUploadBusy] = useState(false);
    const logsRef = useRef(null);
    const fileInputRef = useRef(null);
    const additionalFileRef = useRef(null);
    const unsubRef = useRef(null);
    const pollRef = useRef(null);
    useEffect(() => {
        if (logsRef.current)
            logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }, [logs]);
    // Auto-detect GPU VRAM on mount — pre-fills the field with the real value
    useEffect(() => {
        fetchGpuInfo().then(info => {
            setGpuInfo(info);
            if (info.detected && info.available_gb) {
                setGpuVram(info.available_gb);
            }
        }).catch(() => { });
    }, []);
    useEffect(() => () => {
        unsubRef.current?.();
        if (pollRef.current)
            clearInterval(pollRef.current);
    }, []);
    useEffect(() => {
        const savedId = localStorage.getItem(LS_RUN_KEY);
        if (!savedId)
            return;
        fetchRunResult(savedId).then(result => {
            if (!result || result.status !== "running") {
                localStorage.removeItem(LS_RUN_KEY);
                return;
            }
            setRunId(savedId);
            setStatus("running");
            setLogs([`── reconnexion au pipeline en cours (run ${savedId}) ──`]);
            unsubRef.current = subscribeToRun(savedId, (e) => {
                if (e.kind === "log") {
                    setSseRetrying(false);
                    setLogs(prev => [...prev, e.data]);
                }
                else if (e.kind === "confirm") {
                    try {
                        setConfirmPayload(JSON.parse(e.data));
                    }
                    catch { /* */ }
                }
                else if (e.kind === "done") {
                    try {
                        handleDonePayload(JSON.parse(e.data), savedId);
                    }
                    catch {
                        setError("Réponse malformée");
                        setStatus("error");
                    }
                }
                else if (e.kind === "error") {
                    setSseRetrying(false);
                    localStorage.removeItem(LS_RUN_KEY);
                    setError(e.data);
                    setStatus("error");
                }
            }, setSseRetrying);
        }).catch(() => localStorage.removeItem(LS_RUN_KEY));
    }, []); // eslint-disable-line react-hooks/exhaustive-deps
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    function handleDonePayload(payload, id) {
        setSseRetrying(false);
        setConfirmPayload(null);
        setShowAdditionalUpload(false);
        setAdditionalFiles([]);
        localStorage.removeItem(LS_RUN_KEY);
        if (!payload.ok) {
            setError(payload.error ?? "Erreur inconnue");
            setStatus("error");
            return;
        }
        const finalOut = payload.result?.final_output ?? payload.result;
        const pipelineStatus = finalOut?.status ?? "done";
        if (pipelineStatus === "blocked") {
            setBlockedReason(finalOut?.summary ?? "Fichiers incompatibles.");
            setStatus("blocked");
            return;
        }
        setStatus("done");
        fetchDataset(id).then(setDataset).catch(() => null);
        fetchTrainingReport(id).then(r => { if (r)
            setTrainingReport(r); }).catch(() => null);
        fetchLossHistory(id).then(h => { if (h)
            setLossHistory(h); }).catch(() => null);
        fetchTrainingImages(id).then(setTrainingImages).catch(() => null);
        if (Array.isArray(finalOut?.improvement_log) && finalOut.improvement_log.length)
            setImprovementLog(finalOut.improvement_log);
        if (Array.isArray(finalOut?.decision_journal) && finalOut.decision_journal.length)
            setDecisionJournal(finalOut.decision_journal);
    }
    async function onRun() {
        if (files.length === 0) {
            setError("Veuillez ajouter au moins un fichier.");
            return;
        }
        setStatus("running");
        setLogs([]);
        setError("");
        setBlockedReason("");
        setDataset(null);
        setTrainingReport(null);
        setEvalReport(null);
        setEvalState("idle");
        setLossHistory(null);
        setTrainingImages([]);
        setImprovementLog([]);
        setDecisionJournal([]);
        setConfirmPayload(null);
        setShowAdditionalUpload(false);
        setAdditionalFiles([]);
        try {
            const id = await startRun({ files, goal, language, objective, gpu_vram_gb: gpuVram, task });
            setRunId(id);
            localStorage.setItem(LS_RUN_KEY, id);
            unsubRef.current = subscribeToRun(id, (e) => {
                if (e.kind === "log") {
                    setSseRetrying(false);
                    setLogs(prev => [...prev, e.data]);
                }
                else if (e.kind === "confirm") {
                    try {
                        setConfirmPayload(JSON.parse(e.data));
                    }
                    catch { /* */ }
                }
                else if (e.kind === "open") {
                    setSseRetrying(false);
                    setLogs(prev => [...prev, `── pipeline démarré (run ${id}) ──`]);
                }
                else if (e.kind === "done") {
                    try {
                        handleDonePayload(JSON.parse(e.data), id);
                    }
                    catch {
                        setError("Réponse malformée");
                        setStatus("error");
                    }
                }
                else if (e.kind === "error") {
                    setSseRetrying(false);
                    localStorage.removeItem(LS_RUN_KEY);
                    setError(e.data);
                    setStatus("error");
                }
            }, setSseRetrying);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : String(err));
            setStatus("error");
        }
    }
    async function onConfirm(decision) {
        if (!runId || confirmBusy)
            return;
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
            }
            catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            }
            finally {
                setConfirmBusy(false);
            }
            return;
        }
        // "approve"
        setConfirmBusy(true);
        try {
            await confirmAction(runId, "approve");
            setConfirmPayload(null);
            setShowAdditionalUpload(false);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
        finally {
            setConfirmBusy(false);
        }
    }
    async function onUploadAdditional() {
        if (!runId || additionalFiles.length === 0 || uploadBusy)
            return;
        setUploadBusy(true);
        try {
            const uploaded = await uploadAdditionalFiles(runId, additionalFiles);
            setLogs(prev => [...prev,
                `⬆ ${uploaded.length} fichier(s) supplémentaire(s) envoyé(s) — reprise du pipeline…`]);
            setAdditionalFiles([]);
            setShowAdditionalUpload(false);
            setConfirmPayload(null);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
        finally {
            setUploadBusy(false);
        }
    }
    async function onRunEval() {
        if (!runId)
            return;
        setEvalState("loading");
        try {
            await triggerEval(runId);
            pollRef.current = setInterval(async () => {
                try {
                    const report = await fetchEvalReport(runId);
                    if (report) {
                        setEvalReport(report);
                        setEvalState("done");
                        if (pollRef.current)
                            clearInterval(pollRef.current);
                    }
                }
                catch { /* keep polling */ }
            }, 12_000);
        }
        catch (err) {
            setEvalState("error");
            setError(err instanceof Error ? err.message : String(err));
        }
    }
    const statusLabel = {
        idle: "en attente", running: "en cours", done: "terminé", error: "erreur", blocked: "bloqué",
    };
    const filteredLogs = logs.filter(line => {
        if (logMode === "summary")
            return line.startsWith("[PIPELINE]") || /^\d{2}:\d{2}:\d{2}\s+(ERROR|CRITICAL)\b/.test(line);
        if (logFilter === "warn")
            return /warn/i.test(line);
        if (logFilter === "error")
            return /error|failed|exception/i.test(line);
        return true;
    });
    const pipelineStep = (() => {
        if (status === "idle")
            return 0;
        if (status === "done")
            return 5;
        if (status === "error" || status === "blocked")
            return -1;
        const last = logs[logs.length - 1] ?? "";
        if (/train/i.test(last))
            return 4;
        if (/qa|dataset|paire/i.test(last))
            return 3;
        if (/profil|intent|feasib|select/i.test(last))
            return 2;
        return 1;
    })();
    const STEPS = isExpert
        ? ["Données", "Analyse", "Préparation", "Entraînement", "Évaluation"]
        : ["Vos fichiers", "Compréhension", "Données QA", "Apprentissage", "Test qualité"];
    return (_jsxs(_Fragment, { children: [_jsxs("header", { className: "hero-banner", children: [_jsx("div", { className: "hero-noise" }), _jsx(NeuralNetDecoration, {}), _jsxs("div", { className: "hero-inner", children: [_jsxs("div", { className: "hero-eyebrow", children: ["AI-Powered Model Trainer \u00B7 ", isExpert ? "Pipeline Expert" : "Mode guidé"] }), _jsxs("h1", { className: "hero-title", children: [isExpert ? "Vos données ont du potentiel. " : "Vos documents ont une valeur cachée. ", _jsx("span", { className: "hero-title-accent", children: isExpert ? "Exploitez-le à fond." : "Révélez-la." })] }), _jsx("p", { className: "hero-subtitle", children: isExpert
                                    ? "Du brut au déployable — fine-tuning PEFT automatisé, évaluation rigoureuse et optimisation itérative autonome en un seul pipeline."
                                    : "Importez vos fichiers, décrivez votre objectif en quelques mots — notre IA fait le reste, de A à Z, sans aucune compétence technique requise." }), _jsx("div", { className: "hero-pills", children: isExpert ? (_jsxs(_Fragment, { children: [_jsx("span", { className: "hero-pill", children: "\u2726 Fine-tuning QLoRA / LoRA / DoRA automatique" }), _jsx("span", { className: "hero-pill", children: "\u2726 ROUGE \u00B7 BLEU \u00B7 LLM-as-judge" }), _jsx("span", { className: "hero-pill", children: "\u2726 Optimisation it\u00E9rative autonome" })] })) : (_jsxs(_Fragment, { children: [_jsx("span", { className: "hero-pill", children: "\u2726 Z\u00E9ro comp\u00E9tence ML requise" }), _jsx("span", { className: "hero-pill", children: "\u2726 Guidage clair \u00E0 chaque \u00E9tape" }), _jsx("span", { className: "hero-pill", children: "\u2726 R\u00E9sultats expliqu\u00E9s en fran\u00E7ais" })] })) })] })] }), _jsxs("div", { className: "page", children: [status === "done" && (_jsx("div", { style: {
                            display: "flex", justifyContent: "flex-end", marginBottom: 4,
                        }, children: _jsx("button", { className: "action-btn action-secondary", style: { fontSize: 12 }, onClick: onHistory, children: "\uD83D\uDCDA Voir ce mod\u00E8le dans l'historique \u2192" }) })), _jsx("div", { className: "pipeline-steps", children: STEPS.map((label, i) => (_jsxs("div", { className: `step ${pipelineStep === i + 1 ? "active" : pipelineStep > i + 1 ? "done" : ""}`, children: [_jsx("span", { className: "step-num", children: pipelineStep > i + 1 ? "✓" : i + 1 }), label] }, label))) }), _jsxs("section", { className: "card", children: [_jsx("div", { className: "card-header", children: _jsxs("h2", { children: [_jsx("span", { className: "card-icon", children: "\uD83D\uDCC2" }), "Configuration du pipeline"] }) }), !isExpert && (_jsxs("div", { className: "novice-guide-box", style: { marginBottom: 20 }, children: [_jsx("strong", { children: "Comment \u00E7a marche ?" }), " Importez vos documents (PDF, TXT, CSV), d\u00E9crivez ce que vous voulez que votre mod\u00E8le sache faire, et cliquez sur \"Lancer\". L'IA s'occupe de tout le reste automatiquement."] })), _jsxs("div", { className: "field", children: [_jsx("span", { className: "field-label", children: "Documents sources" }), _jsx("input", { ref: fileInputRef, type: "file", multiple: true, style: { display: "none" }, disabled: status === "running", onChange: e => {
                                            const added = Array.from(e.target.files ?? []);
                                            setFiles(prev => {
                                                const existing = new Set(prev.map(f => f.name));
                                                return [...prev, ...added.filter(f => !existing.has(f.name))];
                                            });
                                            e.target.value = "";
                                        } }), _jsxs("div", { className: `upload-zone ${status === "running" ? "upload-zone-disabled" : ""}`, onClick: () => status !== "running" && fileInputRef.current?.click(), children: [_jsx("div", { className: "upload-icon", children: "\uD83D\uDCC2" }), _jsx("div", { className: "upload-label", children: "Cliquez pour ajouter des fichiers" }), _jsx("div", { className: "upload-sub", children: "PDF \u00B7 TXT \u00B7 CSV \u00B7 Excel \u2014 chaque clic ajoute sans remplacer" }), _jsx("div", { className: "upload-btn", children: "+ Choisir des fichiers" })] }), !isExpert && _jsx("div", { className: "field-hint", children: "Plus vous fournissez de texte, meilleur sera le mod\u00E8le." }), files.length > 0 && (_jsx("div", { style: { marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }, children: files.map(f => (_jsxs("span", { className: "file-chip", children: ["\uD83D\uDCC4 ", f.name, _jsx("button", { className: "file-chip-remove", onClick: e => { e.stopPropagation(); setFiles(prev => prev.filter(x => x.name !== f.name)); }, title: "Retirer ce fichier", children: "\u2715" })] }, f.name))) }))] }), _jsxs("div", { className: "field", children: [_jsxs("span", { className: "field-label", children: ["T\u00E2che ", !isExpert && "— que voulez-vous faire ?"] }), !isExpert && (_jsx("div", { className: "field-hint", style: { marginBottom: 10 }, children: "S\u00E9lectionnez la t\u00E2che principale de votre mod\u00E8le. \"Auto-d\u00E9tection\" laisse l'IA choisir d'apr\u00E8s votre objectif." })), _jsx("div", { style: { display: "flex", flexWrap: "wrap", gap: 8 }, children: TASKS.map(t => {
                                            const selected = task === t.id;
                                            return (_jsxs("button", { type: "button", disabled: status === "running", onClick: () => setTask(t.id), title: t.desc, style: {
                                                    display: "flex",
                                                    alignItems: "center",
                                                    gap: 6,
                                                    padding: "7px 13px",
                                                    borderRadius: 20,
                                                    border: selected ? "2px solid var(--primary)" : "1.5px solid var(--border)",
                                                    background: selected ? "var(--primary)" : "var(--surface)",
                                                    color: selected ? "#fff" : "var(--text)",
                                                    fontWeight: selected ? 700 : 500,
                                                    fontSize: 13,
                                                    cursor: status === "running" ? "not-allowed" : "pointer",
                                                    opacity: status === "running" ? 0.6 : 1,
                                                    transition: "all 0.15s",
                                                    whiteSpace: "nowrap",
                                                }, children: [_jsx("span", { style: { fontSize: 15 }, children: t.icon }), t.label] }, t.id));
                                        }) }), task && (_jsx("div", { className: "field-hint", style: { marginTop: 6 }, children: TASKS.find(t => t.id === task)?.desc }))] }), _jsxs("label", { className: "field", children: [_jsxs("span", { children: ["Objectif ", isExpert ? "(description libre)" : "— que doit savoir faire votre modèle ?"] }), _jsx("textarea", { rows: 3, placeholder: isExpert
                                            ? "Ex. : Fine-tuner un assistant Q&R médical en français pour répondre aux questions des cliniciens."
                                            : "Ex. : Je veux un assistant qui répond aux questions sur mes documents médicaux en français.", value: goal, onChange: e => setGoal(e.target.value), disabled: status === "running" }), !isExpert && _jsx("div", { className: "field-hint", children: "D\u00E9crivez en phrases naturelles ce que vous attendez du mod\u00E8le. Pas besoin de vocabulaire technique." })] }), _jsxs("div", { className: "row", children: [_jsxs("label", { className: "field", style: { maxWidth: 260 }, children: [_jsxs("span", { children: ["Langue ", !isExpert && "des documents"] }), _jsxs("select", { value: language, onChange: e => setLanguage(e.target.value), disabled: status === "running", children: [_jsx("option", { value: "fr", children: "Fran\u00E7ais" }), _jsx("option", { value: "en", children: "English" }), _jsx("option", { value: "ar", children: "Arabe" })] }), !isExpert && _jsx("div", { className: "field-hint", children: "La langue principale de vos documents et des futures questions." })] }), _jsxs("label", { className: "field", style: { maxWidth: 260 }, children: [_jsx("span", { children: isExpert ? "Objectif d'entraînement" : "Priorité" }), _jsxs("select", { value: objective, onChange: e => setObjective(e.target.value), disabled: status === "running", children: [_jsx("option", { value: "balanced", children: isExpert ? "Équilibré — précision & vitesse" : "Équilibré (recommandé)" }), _jsx("option", { value: "speed", children: isExpert ? "Rapidité — entraînement minimal" : "Rapide — entraînement court" }), _jsx("option", { value: "performance", children: isExpert ? "Performance max — meilleure qualité" : "Meilleure qualité possible" })] }), !isExpert && _jsx("div", { className: "field-hint", children: "En doute ? Laissez sur \"\u00C9quilibr\u00E9\"." })] }), gpuInfo?.detected && (_jsxs("div", { style: {
                                            display: "flex", alignItems: "center", gap: 8,
                                            padding: "6px 12px", borderRadius: 8,
                                            background: "#E6F7F1", border: "1px solid #BBF7D0",
                                            fontSize: 13, color: "#16A34A", fontWeight: 500,
                                            alignSelf: "flex-end", marginBottom: 4,
                                        }, children: [_jsx("span", { children: "\u26A1" }), _jsxs("span", { children: [gpuInfo.name, " \u2014 ", gpuInfo.total_gb, " Go VRAM"] })] }))] }), _jsx("button", { className: "run-btn", onClick: onRun, disabled: status === "running", children: status === "running"
                                    ? _jsxs(_Fragment, { children: [_jsx("div", { className: "spinner", style: { width: 16, height: 16, border: "2.5px solid rgba(255,255,255,0.4)", borderTopColor: "#fff", display: "inline-block", verticalAlign: "middle", marginRight: 8 } }), "Pipeline en cours\u2026"] })
                                    : (isExpert ? "▶ Lancer le pipeline" : "▶ Créer mon modèle") }), error && _jsxs("div", { className: "error-box", children: ["\u26A0 ", error] })] }), confirmPayload && (_jsxs("section", { className: "card confirm-card", children: [_jsxs("div", { className: "confirm-header", children: [_jsx("span", { className: "confirm-icon", children: "\u26A0" }), _jsxs("div", { style: { flex: 1 }, children: [_jsx("div", { className: "confirm-title", children: "Dataset insuffisant \u2014 action requise" }), _jsxs("div", { className: "confirm-desc", children: ["Le dataset actuel contient", " ", _jsxs("strong", { children: [confirmPayload.current_pairs, " paire(s)"] }), ". L'objectif recommand\u00E9 est de", " ", _jsxs("strong", { children: [confirmPayload.target_pairs, " paires"] }), "."] }), !isExpert && (_jsx("div", { className: "novice-guide-box", style: { marginTop: 10 }, children: "Il n'y a pas assez d'exemples pour entra\u00EEner un bon mod\u00E8le. Vous avez deux options : laisser le pipeline g\u00E9n\u00E9rer automatiquement des paires depuis vos documents, ou importer des fichiers suppl\u00E9mentaires pour enrichir le dataset." }))] })] }), confirmPayload.sample_pairs && confirmPayload.sample_pairs.length > 0 && (_jsxs("div", { style: { marginTop: 14, marginBottom: 4 }, children: [_jsx("div", { style: { fontWeight: 700, fontSize: 12, color: "var(--muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.04em" }, children: "Aper\u00E7u des paires d\u00E9j\u00E0 g\u00E9n\u00E9r\u00E9es" }), _jsx("div", { className: "qa-list", style: { maxHeight: 220, overflowY: "auto" }, children: confirmPayload.sample_pairs.map((pair, i) => (_jsxs("div", { className: "qa-item", children: [_jsxs("p", { className: "qa-q", children: [_jsx("strong", { children: "Q " }), pair.q] }), _jsxs("p", { className: "qa-a", children: [_jsx("strong", { children: "R " }), pair.a] })] }, i))) })] })), !showAdditionalUpload ? (_jsxs("div", { className: "confirm-actions", style: { marginTop: 16 }, children: [_jsx("button", { className: "action-btn action-primary", onClick: () => onConfirm("approve"), disabled: confirmBusy, children: confirmBusy ? "…" : "✓ Générer automatiquement les paires manquantes" }), _jsx("button", { className: "action-btn action-secondary", onClick: () => onConfirm("refuse"), disabled: confirmBusy, children: "\u2B06 Importer d'autres fichiers" })] })) : (
                            /* Additional file upload zone */
                            _jsxs("div", { style: { marginTop: 16 }, children: [_jsx("div", { style: { fontWeight: 700, fontSize: 13, marginBottom: 10, color: "var(--text)" }, children: "Importez des fichiers suppl\u00E9mentaires (PDF, TXT, CSV, Excel)" }), !isExpert && (_jsx("div", { className: "novice-guide-box", style: { marginBottom: 12 }, children: "Ajoutez des documents compl\u00E9mentaires pour enrichir votre dataset. Le pipeline les traitera et reprendra automatiquement." })), _jsx("input", { ref: additionalFileRef, type: "file", multiple: true, style: { display: "none" }, onChange: e => {
                                            const added = Array.from(e.target.files ?? []);
                                            setAdditionalFiles(prev => {
                                                const existing = new Set(prev.map(f => f.name));
                                                return [...prev, ...added.filter(f => !existing.has(f.name))];
                                            });
                                            e.target.value = "";
                                        } }), _jsxs("div", { className: "upload-zone", onClick: () => additionalFileRef.current?.click(), style: { marginBottom: additionalFiles.length ? 10 : 0 }, children: [_jsx("div", { className: "upload-icon", children: "\uD83D\uDCC2" }), _jsx("div", { className: "upload-label", children: "Cliquez pour ajouter des fichiers" }), _jsx("div", { className: "upload-sub", children: "PDF \u00B7 TXT \u00B7 CSV \u00B7 Excel" }), _jsx("div", { className: "upload-btn", children: "+ Choisir des fichiers" })] }), additionalFiles.length > 0 && (_jsx("div", { style: { display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }, children: additionalFiles.map(f => (_jsxs("span", { className: "file-chip", children: ["\uD83D\uDCC4 ", f.name, _jsx("button", { className: "file-chip-remove", onClick: () => setAdditionalFiles(prev => prev.filter(x => x.name !== f.name)), title: "Retirer ce fichier", children: "\u2715" })] }, f.name))) })), _jsxs("div", { className: "confirm-actions", children: [_jsx("button", { className: "action-btn action-primary", onClick: onUploadAdditional, disabled: uploadBusy || additionalFiles.length === 0, children: uploadBusy ? "Envoi en cours…" : `⬆ Envoyer ${additionalFiles.length > 0 ? `(${additionalFiles.length} fichier${additionalFiles.length > 1 ? "s" : ""})` : ""} et continuer` }), _jsx("button", { className: "action-btn action-secondary", onClick: () => { setShowAdditionalUpload(false); setAdditionalFiles([]); }, disabled: uploadBusy, children: "\u2190 Retour" }), _jsx("button", { className: "action-btn", style: { background: "var(--surface)", color: "var(--muted)", border: "1px solid var(--border)" }, onClick: () => onConfirm("cancel-pipeline"), disabled: uploadBusy || confirmBusy, children: "\u2715 Annuler le pipeline" })] })] }))] })), status === "blocked" && (_jsx("section", { className: "card blocked-card", children: _jsxs("div", { className: "blocked-header", children: [_jsx("span", { className: "blocked-icon", children: "\u26D4" }), _jsxs("div", { style: { flex: 1 }, children: [_jsx("div", { className: "blocked-title", children: "Pipeline bloqu\u00E9 \u2014 fichiers incompatibles" }), _jsx("div", { className: "blocked-reason", children: blockedReason }), _jsxs("div", { className: "blocked-hint", children: ["Pour relancer, importez uniquement des fichiers du ", _jsx("strong", { children: "m\u00EAme domaine" }), "."] })] })] }) })), _jsxs("section", { className: "card", children: [_jsxs("div", { className: "card-header", children: [_jsxs("h2", { children: [_jsx("span", { className: "card-icon", children: "\uD83D\uDCCB" }), "Suivi du pipeline"] }), _jsx("span", { className: `badge badge-${status}`, children: statusLabel[status] })] }), sseRetrying && (_jsxs("div", { className: "sse-retry-banner", children: [_jsx("div", { className: "spinner", style: { width: 12, height: 12, border: "2px solid rgba(124,58,237,0.25)", borderTopColor: "var(--primary)", display: "inline-block", verticalAlign: "middle", marginRight: 8 } }), "Connexion interrompue \u2014 reconnexion en cours\u2026"] })), _jsxs("div", { className: "log-toolbar", children: [_jsxs("span", { className: "muted-text", children: [filteredLogs.length, "/", logs.length, " lignes"] }), _jsxs("div", { className: "log-filter", children: [_jsxs("button", { className: `log-filter-btn ${logMode === "summary" ? "active" : ""}`, onClick: () => setLogMode("summary"), children: ["\uD83D\uDCCB ", isExpert ? "Résumé structuré" : "Vue guidée"] }), _jsxs("button", { className: `log-filter-btn ${logMode === "detail" ? "active" : ""}`, onClick: () => setLogMode("detail"), children: ["\uD83D\uDD0D ", isExpert ? "Logs complets" : "Logs techniques"] }), logMode === "detail" && (_jsxs(_Fragment, { children: [_jsx("span", { style: { width: 1, height: 18, background: "var(--border)", display: "inline-block", margin: "0 2px", verticalAlign: "middle" } }), ["all", "warn", "error"].map(f => (_jsx("button", { className: `log-filter-btn ${logFilter === f ? "active" : ""}`, onClick: () => setLogFilter(f), children: f === "all" ? "Tout" : f === "warn" ? "⚠ Alertes" : "✕ Erreurs" }, f)))] }))] })] }), logMode === "summary"
                                ? _jsx(LivePipelineFeed, { logs: logs, status: status, isExpert: isExpert })
                                : (_jsx("pre", { ref: logsRef, className: "log-pre", children: filteredLogs.length === 0
                                        ? "(aucune sortie pour l'instant)"
                                        : filteredLogs.map((line, i) => _jsx("span", { className: classifyLog(line), children: line + "\n" }, i)) }))] }), decisionJournal.length > 0 && (_jsx(DecisionJournalCard, { journal: decisionJournal, isExpert: isExpert })), trainingReport && (_jsx(TrainingDashboard, { report: trainingReport, runId: runId, lossHistory: lossHistory, trainingImages: trainingImages, isExpert: isExpert })), improvementLog.length > 0 && (_jsxs("section", { className: "card", children: [_jsxs("div", { className: "card-header", children: [_jsxs("h2", { children: [_jsx("span", { className: "card-icon", children: "\uD83D\uDD04" }), isExpert ? "Boucle d'amélioration automatique" : "Tentatives d'amélioration"] }), _jsxs("span", { className: "badge badge-running", children: [improvementLog.length, " it\u00E9ration(s)"] })] }), !isExpert && (_jsx("div", { className: "novice-guide-box", style: { marginBottom: 12 }, children: "Le pipeline a essay\u00E9 plusieurs fois d'am\u00E9liorer le mod\u00E8le. Chaque ligne montre ce qui a \u00E9t\u00E9 tent\u00E9 et le score obtenu. Les fl\u00E8ches \u2191 indiquent une am\u00E9lioration, \u2193 une d\u00E9gradation." })), _jsx("p", { className: "muted-text", style: { marginBottom: 12 }, children: isExpert
                                    ? "Hyperparamètres ajustés et métriques obtenues à chaque itération."
                                    : "Plus le score est proche de 1, meilleur est le modèle." }), _jsx(ImprovementLog, { log: improvementLog })] })), (trainingReport || status === "done") && (_jsx(EvalCard, { evalState: evalState, evalReport: evalReport, onRunEval: onRunEval, isExpert: isExpert })), dataset && (_jsxs("section", { className: "card", children: [_jsxs("div", { className: "card-header", children: [_jsxs("h2", { children: [_jsx("span", { className: "card-icon", children: "\uD83D\uDDC2" }), "Dataset g\u00E9n\u00E9r\u00E9 \u2014 ", dataset.count, " paires (", dataset.format, ")"] }), _jsx("a", { href: datasetDownloadUrl(runId), download: true, children: _jsx("button", { className: "action-btn action-ok", children: "\u2B07 T\u00E9l\u00E9charger .jsonl" }) })] }), !isExpert && (_jsx("div", { className: "novice-guide-box", style: { marginBottom: 12 }, children: "Voici les paires question-r\u00E9ponse que le pipeline a g\u00E9n\u00E9r\u00E9es \u00E0 partir de vos documents. C'est ce que le mod\u00E8le a appris." })), _jsx("div", { className: "qa-list", children: dataset.pairs.map((raw, i) => {
                                    const pair = raw;
                                    if (dataset.format === "alpaca") {
                                        return (_jsxs("div", { className: "qa-item", children: [_jsxs("p", { className: "qa-q", children: [_jsx("strong", { children: "Q " }), String(pair.instruction ?? "")] }), _jsxs("p", { className: "qa-a", children: [_jsx("strong", { children: "R " }), String(pair.output ?? "")] })] }, i));
                                    }
                                    const convs = pair.conversations ?? [];
                                    const human = convs.find(c => c.from === "human");
                                    const gpt = convs.find(c => c.from === "gpt");
                                    return (_jsxs("div", { className: "qa-item", children: [_jsxs("p", { className: "qa-q", children: [_jsx("strong", { children: "Q " }), human?.value ?? ""] }), _jsxs("p", { className: "qa-a", children: [_jsx("strong", { children: "R " }), gpt?.value ?? ""] })] }, i));
                                }) })] }))] })] }));
}
