import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { NeuralNetDecoration, FloatingMLBackground, NeuralNetPageBackground } from "./MLDecorations";
export default function AuthPage({ onLogin }) {
    const [prenom, setPrenom] = useState("");
    const [nom, setNom] = useState("");
    const [isExpert, setIsExpert] = useState(null);
    const [error, setError] = useState("");
    function handleSubmit() {
        if (!prenom.trim()) {
            setError("Veuillez entrer votre prénom.");
            return;
        }
        if (!nom.trim()) {
            setError("Veuillez entrer votre nom.");
            return;
        }
        if (isExpert === null) {
            setError("Veuillez sélectionner votre profil.");
            return;
        }
        setError("");
        onLogin({ prenom: prenom.trim(), nom: nom.trim(), isExpert });
    }
    return (_jsxs("div", { className: "auth-page", children: [_jsx(FloatingMLBackground, {}), _jsx(NeuralNetPageBackground, {}), _jsxs("div", { className: "auth-banner", children: [_jsx("div", { className: "hero-noise" }), _jsx(NeuralNetDecoration, {}), _jsxs("div", { className: "auth-banner-inner", children: [_jsx("div", { className: "hero-eyebrow", children: "AI-Powered Model Trainer \u00B7 DATA&AI-2025" }), _jsxs("h1", { className: "hero-title", style: { fontSize: 32 }, children: ["Vos donn\u00E9es ont du potentiel.", " ", _jsx("span", { className: "hero-title-accent", children: "Exploitez-le." })] }), _jsx("p", { className: "hero-subtitle", style: { fontSize: 14 }, children: "Du brut au d\u00E9ployable \u2014 fine-tuning IA enti\u00E8rement automatis\u00E9, sans expertise technique requise." })] })] }), _jsxs("div", { className: "auth-card", children: [_jsx("h2", { className: "auth-title", children: "Cr\u00E9er votre profil" }), _jsx("p", { className: "auth-subtitle", children: "Ces informations permettent de personnaliser votre exp\u00E9rience." }), _jsxs("div", { className: "auth-fields", children: [_jsxs("label", { className: "field", children: [_jsx("span", { children: "Pr\u00E9nom" }), _jsx("input", { type: "text", placeholder: "Ex. : Hadil", value: prenom, onChange: e => setPrenom(e.target.value), onKeyDown: e => e.key === "Enter" && handleSubmit() })] }), _jsxs("label", { className: "field", children: [_jsx("span", { children: "Nom" }), _jsx("input", { type: "text", placeholder: "Ex. : Souilem", value: nom, onChange: e => setNom(e.target.value), onKeyDown: e => e.key === "Enter" && handleSubmit() })] })] }), _jsx("div", { className: "auth-profile-title", children: "Quel est votre profil ?" }), _jsxs("div", { className: "auth-profiles", children: [_jsxs("div", { className: `auth-profile-card ${isExpert === true ? "selected" : ""}`, onClick: () => setIsExpert(true), children: [_jsx("div", { className: "auth-profile-icon", children: "\uD83C\uDF93" }), _jsx("div", { className: "auth-profile-name", children: "Expert" }), _jsx("div", { className: "auth-profile-desc", children: "Je connais le machine learning, les m\u00E9triques ROUGE/BLEU et le fine-tuning. Je veux l'interface compl\u00E8te sans explications suppl\u00E9mentaires." }), _jsxs("div", { className: "auth-profile-features", children: [_jsx("span", { children: "\u2713 Interface compl\u00E8te" }), _jsx("span", { children: "\u2713 M\u00E9triques techniques" }), _jsx("span", { children: "\u2713 Logs d\u00E9taill\u00E9s" })] })] }), _jsxs("div", { className: `auth-profile-card ${isExpert === false ? "selected" : ""}`, onClick: () => setIsExpert(false), children: [_jsx("div", { className: "auth-profile-icon", children: "\uD83C\uDF31" }), _jsx("div", { className: "auth-profile-name", children: "Technicien" }), _jsx("div", { className: "auth-profile-desc", children: "Je d\u00E9couvre le fine-tuning et j'ai besoin d'explications \u00E0 chaque \u00E9tape pour comprendre ce que fait le pipeline." }), _jsxs("div", { className: "auth-profile-features", children: [_jsx("span", { children: "\u2713 Guides \u00E9tape par \u00E9tape" }), _jsx("span", { children: "\u2713 Explications simplifi\u00E9es" }), _jsx("span", { children: "\u2713 Conseils contextuels" })] })] })] }), error && (_jsxs("div", { className: "error-box", style: { marginBottom: 0 }, children: ["\u26A0 ", error] })), _jsx("button", { className: "run-btn", style: { width: "100%", marginTop: 20, fontSize: 16 }, onClick: handleSubmit, disabled: !prenom || !nom || isExpert === null, children: "Commencer \u2192" })] })] }));
}
