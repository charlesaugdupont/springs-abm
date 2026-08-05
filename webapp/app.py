"""FastAPI application factory - wires routers, session middleware, and
static files. See webapp/settings.py for environment-driven config."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from webapp import auth, executor
from webapp.routers import about, runs, scenario
from webapp.settings import DEV_INSECURE_SECRET_KEY, IS_DEV, SESSION_SECRET_KEY, SHARED_PASSWORD

app = FastAPI(title="SPRINGS-ABM")

if not IS_DEV:
    if not SHARED_PASSWORD:
        raise RuntimeError(
            "WEBAPP_SHARED_PASSWORD must be set outside of local dev (WEBAPP_ENV=dev)."
        )
    if SESSION_SECRET_KEY == DEV_INSECURE_SECRET_KEY:
        raise RuntimeError(
            "WEBAPP_SECRET_KEY must be set to a real random value outside of local dev "
            "(WEBAPP_ENV=dev) - left at its insecure default, session cookies would be "
            "forgeable. Generate one with: python3 -c \"import secrets; print(secrets.token_hex(32))\""
        )

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, same_site="lax")

app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

app.include_router(auth.router)
app.include_router(scenario.router)
app.include_router(runs.router)
app.include_router(about.router)


@app.get("/")
def index():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/scenario", status_code=303)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.on_event("startup")
def _warm_up():
    executor.warm_up()
