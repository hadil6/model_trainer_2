import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState } from "react";
import AuthPage from "./AuthPage";
import HistoryPage from "./HistoryPage";
import PipelinePage from "./PipelinePage";
import { FloatingMLBackground, NeuralNetPageBackground } from "./MLDecorations";
export default function App() {
    const [page, setPage] = useState("auth");
    const [user, setUser] = useState(null);
    function handleLogin(u) {
        setUser(u);
        setPage("pipeline");
    }
    function handleLogout() {
        setUser(null);
        setPage("auth");
    }
    if (!user || page === "auth") {
        return _jsx(AuthPage, { onLogin: handleLogin });
    }
    return (_jsxs(_Fragment, { children: [_jsx(FloatingMLBackground, {}), _jsx(NeuralNetPageBackground, {}), _jsxs("nav", { className: "app-navbar", children: [_jsxs("div", { className: "app-navbar-brand", children: [_jsx("span", { className: "app-navbar-logo", children: "\u2726" }), _jsx("span", { className: "app-navbar-title", children: "ModelTrainer" })] }), _jsxs("div", { className: "app-navbar-links", children: [_jsx("button", { className: `app-nav-link ${page === "pipeline" ? "active" : ""}`, onClick: () => setPage("pipeline"), children: "\u25B6 Pipeline" }), _jsx("button", { className: `app-nav-link ${page === "history" ? "active" : ""}`, onClick: () => setPage("history"), children: "\uD83D\uDCDA Historique" })] }), _jsxs("div", { className: "app-navbar-user", children: [_jsxs("span", { className: "app-navbar-name", children: [user.prenom, " ", user.nom] }), _jsx("span", { className: "app-navbar-badge", children: user.isExpert ? "Expert" : "Technicien" }), _jsx("button", { className: "app-navbar-logout", onClick: handleLogout, children: "Changer de profil" })] })] }), page === "history"
                ? _jsx(HistoryPage, { user: user, onBack: () => setPage("pipeline") })
                : _jsx(PipelinePage, { user: user, onHistory: () => setPage("history") })] }));
}
