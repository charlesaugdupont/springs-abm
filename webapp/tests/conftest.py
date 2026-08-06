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
