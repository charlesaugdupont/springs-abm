"""Tests for the auth flow: GET /api/session, POST /api/login, POST
/api/logout, and require_login's 401 behavior (replacing the old
303-redirect-to-/login, which was meaningless to a fetch() call).

Monkeypatches webapp.auth.SHARED_PASSWORD directly, not
webapp.settings.SHARED_PASSWORD - auth.py imports the value with `from
webapp.settings import SHARED_PASSWORD`, a one-time value copy into its own
module namespace, not a live reference back into settings.py."""
from __future__ import annotations

from webapp import auth


def test_session_reports_unauthenticated_by_default(anon_client):
    resp = anon_client.get("/api/session")
    assert resp.status_code == 200
    assert resp.json() == {"authenticated": False}


def test_protected_route_returns_401_when_not_logged_in(anon_client):
    resp = anon_client.get("/api/parameters")
    assert resp.status_code == 401


def test_login_wrong_password_returns_401(anon_client, monkeypatch):
    monkeypatch.setattr(auth, "SHARED_PASSWORD", "correct-horse")
    resp = anon_client.post("/api/login", json={"password": "wrong"})
    assert resp.status_code == 401


def test_login_correct_password_sets_session(anon_client, monkeypatch):
    monkeypatch.setattr(auth, "SHARED_PASSWORD", "correct-horse")
    resp = anon_client.post("/api/login", json={"password": "correct-horse"})
    assert resp.status_code == 200
    assert resp.json() == {"authenticated": True}

    session_resp = anon_client.get("/api/session")
    assert session_resp.json() == {"authenticated": True}

    protected_resp = anon_client.get("/api/parameters")
    assert protected_resp.status_code == 200


def test_logout_clears_session(anon_client, monkeypatch):
    monkeypatch.setattr(auth, "SHARED_PASSWORD", "correct-horse")
    anon_client.post("/api/login", json={"password": "correct-horse"})
    resp = anon_client.post("/api/logout")
    assert resp.status_code == 200
    assert resp.json() == {"authenticated": False}

    session_resp = anon_client.get("/api/session")
    assert session_resp.json() == {"authenticated": False}


def test_login_with_no_shared_password_configured_always_fails(anon_client, monkeypatch):
    monkeypatch.setattr(auth, "SHARED_PASSWORD", None)
    resp = anon_client.post("/api/login", json={"password": "anything"})
    assert resp.status_code == 401
