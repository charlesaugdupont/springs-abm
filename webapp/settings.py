"""Deployment-facing settings, sourced from environment variables where
they need to differ between local dev and the hosted deployment."""
import os

DEFAULT_GRID_ID = "7d9ce7c720a6"  # the only grid that exists - see Finding #6, never derived from user input

# Session/auth (see webapp/auth.py). No defaults for secrets in production:
# app.py's startup check refuses to boot outside dev if either is missing/
# left at its insecure default.
DEV_INSECURE_SECRET_KEY = "dev-only-insecure-secret-change-me"
SESSION_SECRET_KEY = os.environ.get("WEBAPP_SECRET_KEY", DEV_INSECURE_SECRET_KEY)
SHARED_PASSWORD = os.environ.get("WEBAPP_SHARED_PASSWORD")
IS_DEV = os.environ.get("WEBAPP_ENV", "dev") == "dev"

# Hosting-driven hard caps on the two scale-affecting scenario fields - see
# the plan's §2: tighter than what parameter_registry.py's ui_min/ui_max
# would otherwise allow, specifically to bound worst-case wall-clock time
# on Render's free (CPU-throttled) tier. A separate concern from the
# registry's ui bounds (which reflect what's scientifically reasonable,
# not what's currently affordable to compute) - loosen here, not there, if
# hosting ever changes.
MIN_AGENTS, MAX_AGENTS = 500, 5_000
MIN_STEPS, MAX_STEPS = 30, 200
