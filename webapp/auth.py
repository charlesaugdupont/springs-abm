"""
Single shared-password gate for the whole app (see the plan's Auth &
deployment section) - no per-user accounts, no database. A correct
password sets `session["authenticated"] = True` on Starlette's signed
session cookie (itsdangerous-signed, tamper-proof but not encrypted -
fine, since the cookie holds nothing but a boolean).
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from webapp.settings import SHARED_PASSWORD
from webapp.templating import templates

router = APIRouter()


def require_login(request: Request) -> None:
    """FastAPI dependency - routes needing auth depend on this. Redirects to /login (a
    plain 303 + Location header, which browsers follow natively) rather than a raw 401."""
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=303, headers={"Location": "/login"})


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse("/scenario", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login_submit(request: Request, password: str = Form(...)):
    if SHARED_PASSWORD and secrets.compare_digest(password, SHARED_PASSWORD):
        request.session["authenticated"] = True
        return RedirectResponse("/scenario", status_code=303)
    return templates.TemplateResponse(
        request, "login.html", {"error": "Incorrect password."}, status_code=401,
    )


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
