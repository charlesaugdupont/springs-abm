"""Shared fixtures for the webapp API test suite."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp import db
from webapp.app import app
from webapp.auth import require_login


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Every test gets its own throwaway SQLite file - no shared state with
    a real deployment's data, and no bleed-over between tests."""
    db.reset_for_tests(str(tmp_path / "test_runs.db"))
    yield


@pytest.fixture(autouse=True)
def drain_simulation_worker():
    """Wait, after each test, for any simulation it submitted to finish.

    POST /api/scenario/run is fire-and-forget: the job runs on the executor's
    worker thread after the request returns. Left running, it keeps drawing
    from torch's process-global RNG while later tests - including the
    model-core suite, which runs in the same process - seed and step their own
    models, so their random streams (and any marginal assertion) depended on
    thread timing. The executor has one worker, so a no-op submitted behind
    the queue completes only once everything before it has.
    """
    yield
    from webapp.executor import _EXECUTOR
    _EXECUTOR.submit(lambda: None).result(timeout=600)


@pytest.fixture
def client():
    """An authenticated TestClient - bypasses require_login via FastAPI's
    dependency_overrides, since most API tests aren't testing auth itself
    (see test_api_auth.py for that)."""
    app.dependency_overrides[require_login] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(require_login, None)


@pytest.fixture
def anon_client():
    """A TestClient with no auth override - for exercising the real
    login/session/401 behavior."""
    with TestClient(app) as c:
        yield c
