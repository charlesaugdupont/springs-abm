"""
Single shared-password gate for the whole app (see the plan's Auth &
deployment section) - no per-user accounts, no database. A correct
password sets `session["authenticated"] = True` on Starlette's signed
session cookie (itsdangerous-signed, tamper-proof but not encrypted -
fine, since the cookie holds nothing but a boolean).

require_login raises a plain 401 rather than the old 303-redirect-to-
/login: a redirect Location header is meaningless to a fetch() call from
the SPA, which would just silently receive login-page HTML back as "data".
The SPA's fetch wrapper (frontend/src/api/client.ts) is what turns a 401
into an actual client-side redirect to /login.
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from webapp.settings import SHARED_PASSWORD

router = APIRouter()


class LoginRequest(BaseModel):
    password: str


def require_login(request: Request) -> None:
    """FastAPI dependency - routes needing auth depend on this."""
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated.")


@router.get("/session")
def session_status(request: Request):
    return {"authenticated": bool(request.session.get("authenticated"))}


@router.post("/login")
def login_submit(request: Request, body: LoginRequest):
    if SHARED_PASSWORD and secrets.compare_digest(body.password, SHARED_PASSWORD):
        request.session["authenticated"] = True
        return {"authenticated": True}
    raise HTTPException(status_code=401, detail="Incorrect password.")


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"authenticated": False}
