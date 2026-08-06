# SPRINGS-ABM web UI - deployed image. Deliberately excludes the GDAL/PROJ
# geo stack (geopandas/rasterio/osmnx/contextily) and matplotlib/seaborn:
# see requirements-web.txt's header comment for why neither is needed at
# webapp runtime.

# ---- Stage 1: build the frontend/ SPA ----------------------------------
# Only frontend/ is needed here - Vite's own outDir (see
# frontend/vite.config.ts) writes straight to ../webapp/frontend_dist,
# which this stage creates fresh; the Python webapp/ tree isn't copied
# into this stage at all.
FROM node:22-slim AS frontend-builder
WORKDIR /repo/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: the actual deployed image --------------------------------
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEBAPP_ENV=production

COPY requirements-web.txt .
# CPU-only torch wheel, explicitly - the default PyPI wheel bundles CUDA
# (multi-GB); this project never uses a GPU (config.py's device defaults
# to "cpu" everywhere, and webapp/executor.py never overrides it).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 \
 && pip install -r requirements-web.txt

# useradd before the app-code COPYs (not a trailing `chown -R /app`
# afterward) so each COPY's --chown lands the right ownership in the same
# layer it's created in, rather than adding a whole extra layer that
# duplicates every previously-copied file's size just to re-own them. The
# plain `chown /app` (non-recursive - /app is still empty at this point,
# so this is nearly free) is still needed on top of that: WORKDIR created
# /app as root, and abm/model/initialize_model.py's SVEIRModel creates a
# fresh top-level outputs/ directory at runtime
# (root_path="outputs/webapp" in webapp/simulation_runner.py) - webapp
# needs write access to /app *itself* to create that new directory, not
# just to the specific subdirectories each COPY already owns.
RUN useradd -m webapp && chown webapp:webapp /app

# App code, in dependency order. main.py/run_viz.py/sensitivity.py are the
# legacy CLI workflow - not used by the webapp, but abm/ imports config.py
# which sits at repo root, so the root Python path layout is preserved
# rather than restructured for this one deployment target.
COPY --chown=webapp:webapp config.py ./
COPY --chown=webapp:webapp abm/ ./abm/
COPY --chown=webapp:webapp experiments/__init__.py experiments/orchestrator.py experiments/metrics.py ./experiments/
COPY --chown=webapp:webapp webapp/ ./webapp/
COPY --chown=webapp:webapp grids/7d9ce7c720a6/ ./grids/7d9ce7c720a6/
COPY --chown=webapp:webapp --from=frontend-builder /repo/webapp/frontend_dist ./webapp/frontend_dist

USER webapp

EXPOSE 8000
# --workers 1 is REQUIRED, not a default left un-tuned: webapp/jobs.py's
# SQLite-backed job store and webapp/executor.py's single-slot execution
# queue both assume exactly one process. On Cloud Run this is enforced at
# the *service* level too (--max-instances=1 at deploy time, not something
# this file can express) - see the plan's Phase 6: Cloud Run's default
# autoscaling would otherwise let multiple instances run concurrently,
# each with its own disconnected ephemeral filesystem, silently breaking
# job polling.
#
# Shell form (not exec-form array) so $PORT expands - Cloud Run (like
# Render and most PaaS Docker hosts) injects its own PORT env var at
# runtime (default 8080) and expects the container to listen on it, not on
# a value baked in at build time. Falls back to 8000 for local `docker run`
# with no PORT set. `exec` makes the shell replace itself with uvicorn
# (rather than fork it as a child) so uvicorn runs as PID 1 and actually
# receives SIGTERM directly for a clean shutdown - Cloud Run sends SIGTERM
# on every scale-down/redeploy, and without this an in-flight request could
# get cut off by the harder SIGKILL that follows an unacknowledged SIGTERM.
CMD exec uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
