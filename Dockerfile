# SPRINGS-ABM web UI - deployed image. Deliberately excludes the GDAL/PROJ
# geo stack (geopandas/rasterio/osmnx/contextily) and matplotlib/seaborn:
# see requirements-web.txt's header comment for why neither is needed at
# webapp runtime.
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

# App code, in dependency order. main.py/run_viz.py/sensitivity.py are the
# legacy CLI workflow - not used by the webapp, but abm/ imports config.py
# which sits at repo root, so the root Python path layout is preserved
# rather than restructured for this one deployment target.
COPY config.py ./
COPY abm/ ./abm/
COPY experiments/__init__.py experiments/orchestrator.py experiments/metrics.py ./experiments/
COPY webapp/ ./webapp/
COPY grids/7d9ce7c720a6/ ./grids/7d9ce7c720a6/

RUN useradd -m webapp && chown -R webapp:webapp /app
USER webapp

EXPOSE 8000
# --workers 1 is REQUIRED, not a default left un-tuned: webapp/jobs.py's
# in-memory job store and webapp/executor.py's single-slot execution queue
# are both process-local singletons - see the plan's §2/§6.
#
# Shell form (not exec-form array) so $PORT expands - Render (like most
# PaaS Docker hosts) injects its own PORT env var at runtime and expects
# the container to listen on it, not on a value baked in at build time.
# Falls back to 8000 for local `docker run` with no PORT set.
CMD uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
