"""Deployment-facing settings, sourced from environment variables where
they need to differ between local dev and the hosted deployment."""
import os
import time

# Cache-busting query param appended to static asset URLs (see base.html) -
# a fresh value every process start (i.e. every deploy), so a browser that
# cached an old stylesheet/script from a previous version fetches the new
# one instead of silently keeping the stale cached copy after a redeploy.
STATIC_VERSION = str(int(time.time()))

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
MIN_AGENTS, MAX_AGENTS = 500, 10_000
MIN_STEPS, MAX_STEPS = 30, 500

# These MUST match parameter_registry.py's number_agents/step_target ui_min/
# ui_max exactly - the registry drives what the slider's HTML min/max
# attributes show, these drive what the server actually accepts (see
# scenario_form.py). A past mismatch here (registry allowed up to 365 days,
# this cap only allowed 200) meant dragging into that gap produced a value
# the server then rejected - the two are asserted equal at import time so
# that class of bug fails loudly instead of silently drifting again.
