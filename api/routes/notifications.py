"""User-facing notification endpoints.

Currently exposes:
  POST /api/notify/login   → triggers a login-alert email to the user.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

from services.email_notifier import send_login_notification

logger = logging.getLogger(__name__)
router  = APIRouter(prefix="/api/notify")


class LoginPayload(BaseModel):
    email:     EmailStr
    prenom:    str       = Field(..., min_length=1, max_length=64)
    nom:       str       = Field(..., min_length=1, max_length=64)
    is_expert: bool      = False


@router.post("/login")
async def notify_login(
    payload: LoginPayload,
    background: BackgroundTasks,
    request: Request,
) -> JSONResponse:
    """Send a 'someone just logged into your account' email — non-blocking.

    The actual SMTP send happens in a background task so the login flow is
    never delayed. If SMTP is unconfigured, the function silently no-ops
    (we don't want to expose that publicly in the response).
    """
    ip = request.client.host if request.client else None
    background.add_task(
        send_login_notification,
        email=str(payload.email).strip().lower(),
        prenom=payload.prenom.strip(),
        nom=payload.nom.strip(),
        is_expert=payload.is_expert,
        ip=ip,
    )
    logger.info("notify_login queued: email=%s ip=%s", payload.email, ip)
    return JSONResponse({"queued": True})
