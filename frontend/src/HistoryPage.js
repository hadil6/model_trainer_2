import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { fetchRunHistory, reportDownloadUrl, modelDownloadUrl } from "./api";
function statusInfo(status) {
    if (status === "trained")
        return { label: "Réussi", icon: "✅", cls: "badge-done" };
    if (status === "trained_low_quality")
        return { label: "Qualité limitée", icon: "⚠️", cls: "badge-blocked" };
    if (status === "blocked")
        return { label: "Bloqué", icon: "⛔", cls: "badge-error" };
    if (status === "running")
        return { label: "En cours", icon: "⏳", cls: "badge-running" };
    if (status === "error")
        return { label: "Erreur", icon: "✕", cls: "badge-error" };
    return { label: status, icon: "•", cls: "badge-idle" };
}
function tierColor(tier) {
    if (tier === "good")
        return "var(--ok)";
    if (tier === "acceptable")
        return "var(--warn)";
    return "var(--err)";
}
function formatDate(ts) {
    if (!ts)
        return "—";
    return new Date(ts * 1000).toLocaleDateString("fr-FR", {
        day: "2-digit", month: "short", year: "numeric",
        hour: "2-digit", minute: "2-digit",
    });
}
export default function HistoryPage({ user, onBack, }) {
    const [runs, setRuns] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");
    useEffect(() => {
        fetchRunHistory()
            .then(setRuns)
            .finally(() => setLoading(false));
    }, []);
    const filtered = runs.filter(r => {
        if (!search.trim())
            return true;
        const q = search.toLowerCase();
        return (r.job_id.toLowerCase().includes(q) ||
            (r.model_id ?? "").toLowerCase().includes(q) ||
            (r.task ?? "").toLowerCase().includes(q) ||
            (r.domain ?? "").toLowerCase().includes(q) ||
            (r.status ?? "").toLowerCase().includes(q));
    });
    return (_jsxs("div", { className: "page", children: [_jsxs("div", { className: "history-header", children: [_jsxs("div", { children: [_jsx("h1", { className: "history-title", children: "\uD83D\uDCDA Historique des mod\u00E8les" }), _jsxs("p", { className: "muted-text", children: ["Bonjour ", user.prenom, " \u2014 retrouvez ici tous vos fine-tunings r\u00E9alis\u00E9s."] })] }), _jsx("button", { className: "action-btn action-secondary", onClick: onBack, children: "\u2190 Nouveau pipeline" })] }), !loading && runs.length > 0 && (_jsxs("div", { className: "history-stats", children: [_jsxs("div", { className: "history-stat", children: [_jsx("span", { className: "history-stat-value", children: runs.length }), _jsx("span", { className: "history-stat-label", children: "Total" })] }), _jsxs("div", { className: "history-stat", children: [_jsx("span", { className: "history-stat-value", style: { color: "var(--ok)" }, children: runs.filter(r => r.status === "trained").length }), _jsx("span", { className: "history-stat-label", children: "R\u00E9ussis" })] }), _jsxs("div", { className: "history-stat", children: [_jsx("span", { className: "history-stat-value", style: { color: "var(--warn)" }, children: runs.filter(r => r.status === "trained_low_quality").length }), _jsx("span", { className: "history-stat-label", children: "Qualit\u00E9 limit\u00E9e" })] }), _jsxs("div", { className: "history-stat", children: [_jsx("span", { className: "history-stat-value", style: { color: "var(--err)" }, children: runs.filter(r => r.status === "blocked" || r.status === "error").length }), _jsx("span", { className: "history-stat-label", children: "\u00C9chou\u00E9s" })] })] })), runs.length > 3 && (_jsx("input", { type: "text", placeholder: "\uD83D\uDD0D Rechercher par mod\u00E8le, t\u00E2che, domaine, ID\u2026", value: search, onChange: e => setSearch(e.target.value), style: {
                    width: "100%", padding: "10px 14px",
                    borderRadius: 9, border: "1.5px solid var(--border)",
                    fontSize: 14, background: "var(--surface)", color: "var(--text)",
                    fontFamily: "inherit",
                } })), loading && (_jsxs("div", { style: { textAlign: "center", padding: "40px 0", color: "var(--muted)" }, children: [_jsx("div", { className: "spinner", style: { margin: "0 auto 12px" } }), "Chargement de l'historique\u2026"] })), !loading && runs.length === 0 && (_jsxs("div", { className: "card", style: { textAlign: "center", padding: "48px 24px" }, children: [_jsx("div", { style: { fontSize: 48, marginBottom: 16 }, children: "\uD83D\uDDC2" }), _jsx("h3", { style: { margin: "0 0 8px", color: "var(--primary)" }, children: "Aucun mod\u00E8le r\u00E9alis\u00E9" }), _jsx("p", { className: "muted-text", children: "Lancez votre premier pipeline pour voir vos mod\u00E8les ici." }), _jsx("button", { className: "action-btn action-primary", style: { marginTop: 20 }, onClick: onBack, children: "\u25B6 Lancer un pipeline" })] })), !loading && runs.length > 0 && filtered.length === 0 && (_jsx("div", { className: "card", style: { textAlign: "center", padding: "32px 24px" }, children: _jsxs("p", { className: "muted-text", children: ["Aucun r\u00E9sultat pour \u00AB ", search, " \u00BB."] }) })), !loading && filtered.map(run => {
                const st = statusInfo(run.status);
                const metric = run.primary_metric?.toUpperCase().replace(/_/g, "-") ?? "—";
                const metricVal = run.primary_metric_value;
                return (_jsxs("div", { className: "card history-run-card", children: [_jsxs("div", { className: "history-run-top", children: [_jsxs("div", { style: { display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }, children: [_jsx("span", { style: { fontSize: 20 }, children: st.icon }), _jsx("span", { className: "mono", style: { fontSize: 13, color: "var(--muted)" }, children: run.job_id }), _jsx("span", { className: `badge ${st.cls}`, children: st.label }), run.quality_tier && (_jsx("span", { className: "badge", style: {
                                                background: run.quality_tier === "good" ? "#DCFCE7" : run.quality_tier === "acceptable" ? "#FEF8E7" : "#FEE2E2",
                                                color: tierColor(run.quality_tier),
                                                borderColor: tierColor(run.quality_tier),
                                            }, children: run.quality_tier === "good" ? "Bonne qualité" : run.quality_tier === "acceptable" ? "Acceptable" : "Insuffisant" }))] }), _jsx("span", { className: "muted-text", style: { fontSize: 12 }, children: formatDate(run.created_at) })] }), _jsxs("div", { className: "history-run-grid", children: [run.model_id && (_jsxs("div", { className: "history-run-item", children: [_jsx("span", { className: "history-run-label", children: "Mod\u00E8le" }), _jsx("span", { className: "history-run-value mono", children: run.model_id.split("/").pop() })] })), run.peft_method && (_jsxs("div", { className: "history-run-item", children: [_jsx("span", { className: "history-run-label", children: "M\u00E9thode" }), _jsx("span", { className: "history-run-value", children: run.peft_method.toUpperCase() })] })), run.task && (_jsxs("div", { className: "history-run-item", children: [_jsx("span", { className: "history-run-label", children: "T\u00E2che" }), _jsx("span", { className: "history-run-value", children: run.task })] })), run.domain && (_jsxs("div", { className: "history-run-item", children: [_jsx("span", { className: "history-run-label", children: "Domaine" }), _jsx("span", { className: "history-run-value", children: run.domain })] })), run.n_pairs != null && (_jsxs("div", { className: "history-run-item", children: [_jsx("span", { className: "history-run-label", children: "Paires QA" }), _jsx("span", { className: "history-run-value", children: run.n_pairs })] })), metricVal != null && (_jsxs("div", { className: "history-run-item", children: [_jsx("span", { className: "history-run-label", children: metric }), _jsx("span", { className: "history-run-value", style: { color: tierColor(run.quality_tier), fontFamily: "var(--mono)", fontWeight: 800 }, children: metricVal.toFixed(3) })] }))] }), run.summary && (_jsx("div", { style: {
                                fontSize: 12.5, color: "var(--muted)", marginTop: 10,
                                padding: "7px 11px", borderRadius: 6,
                                background: "var(--panel)", borderLeft: "3px solid var(--border)",
                            }, children: run.summary })), (run.status === "trained" || run.status === "trained_low_quality") && (_jsxs("div", { style: { display: "flex", gap: 10, marginTop: 14, paddingTop: 12, borderTop: "1px solid var(--border)" }, children: [_jsx("a", { href: modelDownloadUrl(run.job_id), download: true, children: _jsx("button", { className: "action-btn action-primary", style: { fontSize: 12 }, children: "\u2B07 Adaptateur LoRA" }) }), _jsx("a", { href: reportDownloadUrl(run.job_id), download: true, children: _jsx("button", { className: "action-btn action-secondary", style: { fontSize: 12 }, children: "\u2B07 Rapport JSON" }) })] }))] }, run.job_id));
            })] }));
}
