"""FastAPI application factory - wires routers, session middleware, and
static files. See webapp/settings.py for environment-driven config.

All application routers are mounted under /api (a clean JSON API - see the
plan's Architecture section). The built SPA (frontend/dist/, via Vite's
outDir - see frontend/vite.config.ts) is served from webapp/frontend_dist/,
mounted at "/" - but as an explicit catch-all route, not a StaticFiles
mount: a Mount at "/" would claim every path unconditionally as a prefix
match regardless of registration order, which would either shadow /api
(if registered first) or never be reached (if registered last, since a
"/" prefix mount swallows requests before any later route sees them).
Registering /api and /assets first, then a plain @app.get("/{full_path}")
last, sidesteps that entirely: it only runs for paths nothing above it
matched, and explicitly serves index.html for anything that isn't a real
built file - the mechanism that keeps client-side routes (e.g. /about,
/runs/<id>) working on a hard refresh, not just client-side navigation."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from webapp import auth, executor
from webapp.routers import parameters, runs, scenario
from webapp.settings import DEV_INSECURE_SECRET_KEY, IS_DEV, SESSION_SECRET_KEY, SHARED_PASSWORD

FRONTEND_DIST = Path("webapp/frontend_dist")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    executor.warm_up()
    yield


app = FastAPI(title="SPRINGS-ABM", lifespan=_lifespan)

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

app.include_router(auth.router, prefix="/api")
app.include_router(parameters.router, prefix="/api")
app.include_router(scenario.router, prefix="/api")
app.include_router(runs.router, prefix="/api")


# Not /healthz: Cloud Run reserves that exact bare path (no trailing slash)
# for its own internal default health-check convention and intercepts it
# at the infrastructure level before it ever reaches this container,
# returning Google's own 404 page instead of routing here - confirmed
# empirically (every other path, including /healthz/ with a trailing
# slash, routes through correctly).
@app.get("/health")
def health():
    return {"status": "ok"}


# Vite's own content-hashed build output (webapp/frontend_dist/assets/*) -
# a real static-file mount is fine here since /assets never needs the
# SPA-fallback behavior below (a missing hashed asset is a real 404, not a
# client-side route).
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend_assets")


@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    """Catch-all, registered last so /api, /static, and /assets above all
    take priority. Serves a root-level built file if the path matches one
    (favicon.svg, etc.) - otherwise falls back to index.html, the mechanism
    that keeps a client-side route like /runs/<id> working on a hard
    refresh instead of 404ing."""
    candidate = FRONTEND_DIST / full_path
    if full_path and candidate.is_file():
        return FileResponse(candidate)

    index_path = FRONTEND_DIST / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend build not found - run `npm run build` in frontend/.")
    return FileResponse(index_path)
