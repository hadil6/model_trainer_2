"""SMTP email notifications — login alerts + pipeline completion.

The module is intentionally a no-op when SMTP env vars are missing, so the
rest of the app keeps working even if email is not configured.

Required environment variables (set in `.env`)
-----------------------------------------------
SMTP_HOST          e.g. smtp.gmail.com
SMTP_PORT          e.g. 587 (TLS)
SMTP_USER          full sender address, e.g. notifications@your-domain.com
SMTP_PASSWORD      app password (Gmail/Outlook) or API key
SMTP_FROM_NAME     human-readable sender name (default "AI Model Trainer")

Public API
----------
send_email(to, subject, body_text)             → low-level sender (returns bool)
send_login_notification(user)                  → "Vous venez de vous connecter"
send_pipeline_completion(user, run_summary)    → "Votre fine-tuning est terminé"
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime

logger = logging.getLogger(__name__)


def _smtp_configured() -> bool:
    """True only if all required SMTP env vars are present."""
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"))


def send_email(to: str, subject: str, body_text: str) -> bool:
    """Send a plain-text email. Returns True on success, False on any failure.

    Never raises — failures are logged but never break the calling flow.
    """
    if not _smtp_configured():
        logger.info("email_skipped (SMTP not configured): would send '%s' to %s", subject, to)
        return False
    if not to or "@" not in to:
        logger.warning("email_skipped (invalid recipient): %r", to)
        return False

    host       = os.environ["SMTP_HOST"]
    port       = int(os.environ["SMTP_PORT"])
    user       = os.environ["SMTP_USER"]
    password   = os.environ["SMTP_PASSWORD"]
    from_name  = os.environ.get("SMTP_FROM_NAME", "AI Model Trainer")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = f"{from_name} <{user}>"
    msg["To"]      = to
    msg.set_content(body_text)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.ehlo()
            server.login(user, password)
            server.send_message(msg)
        logger.info("email_sent: '%s' to %s", subject, to)
        return True
    except Exception as exc:
        logger.warning("email_failed: %s — %s: %s", to, type(exc).__name__, exc)
        return False


# ── High-level helpers ───────────────────────────────────────────────────────

def send_login_notification(
    email: str,
    prenom: str,
    nom: str,
    is_expert: bool,
    ip: str | None = None,
) -> bool:
    """Send a 'you just logged in' alert to the user."""
    if not email:
        return False
    profile = "Expert" if is_expert else "Technicien"
    now     = datetime.now().strftime("%d/%m/%Y à %H:%M")
    ip_line = f"  - Adresse IP : {ip}\n" if ip else ""
    body = (
        f"Bonjour {prenom},\n\n"
        f"Une connexion à votre compte AI Model Trainer vient d'être effectuée :\n\n"
        f"  - Email      : {email}\n"
        f"  - Nom        : {prenom} {nom}\n"
        f"  - Profil     : {profile}\n"
        f"  - Date/heure : {now}\n"
        f"{ip_line}"
        f"\n"
        f"Si vous n'êtes pas à l'origine de cette connexion, ignorez ce message "
        f"ou contactez l'administrateur de la plateforme.\n\n"
        f"Cordialement,\n"
        f"L'équipe AI Model Trainer\n"
    )
    return send_email(email, "Connexion à AI Model Trainer", body)


def send_pipeline_completion(
    email: str,
    prenom: str,
    job_id: str,
    status: str,
    model_id: str,
    primary_metric: str,
    primary_value: float | None,
    n_pairs: int | None,
    duration_minutes: int | None = None,
    history_url: str = "http://localhost:5173",
) -> bool:
    """Send 'your fine-tuning is done' notification with key metrics."""
    if not email:
        return False
    status_label = {
        "trained":             "✓ Réussi",
        "trained_low_quality": "⚠ Qualité limitée",
        "blocked":             "✕ Bloqué",
        "error":               "✕ Erreur",
    }.get(status, status)
    metric_line = (
        f"  - Métrique {primary_metric.upper()} : {primary_value:.3f}\n"
        if primary_value is not None else ""
    )
    pairs_line  = f"  - Paires QA générées  : {n_pairs}\n" if n_pairs else ""
    dur_line    = f"  - Durée totale        : {duration_minutes} minutes\n" if duration_minutes else ""
    body = (
        f"Bonjour {prenom},\n\n"
        f"Votre fine-tuning vient de se terminer.\n\n"
        f"  - Job ID              : {job_id}\n"
        f"  - Modèle de base      : {model_id}\n"
        f"  - Statut              : {status_label}\n"
        f"{metric_line}"
        f"{pairs_line}"
        f"{dur_line}"
        f"\n"
        f"Consultez les détails et téléchargez votre adaptateur LoRA ici :\n"
        f"  {history_url}\n\n"
        f"Cordialement,\n"
        f"L'équipe AI Model Trainer\n"
    )
    return send_email(email, f"Pipeline terminé — {status_label}", body)
