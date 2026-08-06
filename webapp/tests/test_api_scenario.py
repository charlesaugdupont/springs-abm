"""Tests for POST /api/scenario/run. The end-to-end test
(test_submit_and_poll_to_completion_returns_config_form) is the actual
regression test for this phase's core fix: the results page used to never
show what parameters produced a run, even though the server captured them
all along - it just never sent them back to any client."""
from __future__ import annotations

import time

from webapp import executor
from webapp.scenario_form import ScenarioFormInput


def _minimal_payload(**overrides) -> dict:
    """The smallest/fastest legal scenario (hosting-driven MIN_AGENTS/
    MIN_STEPS, see webapp/settings.py) as a plain JSON-able dict."""
    payload = ScenarioFormInput().model_dump()
    payload.update(overrides)
    return payload


def test_submit_and_poll_to_completion_returns_config_form(client):
    payload = _minimal_payload()
    resp = client.post("/api/scenario/run", json=payload)
    assert resp.status_code == 202
    body = resp.json()
    job_id = body["job_id"]
    assert body["status"] == "queued"

    deadline = time.time() + 120
    result = None
    while time.time() < deadline:
        poll = client.get(f"/api/runs/{job_id}")
        assert poll.status_code == 200
        poll_body = poll.json()
        # The core regression test: config_form must be present at every
        # stage of the poll, not just once the run finishes.
        assert poll_body["config_form"]["number_agents"] == payload["number_agents"]
        if poll_body["status"] == "done":
            result = poll_body
            break
        if poll_body["status"] == "error":
            raise AssertionError(f"Run failed: {poll_body['error']}")
        time.sleep(1)

    assert result is not None, "Simulation did not finish within the test timeout"
    assert result["result"]["n_u5"] > 0
    assert set(result["result"]["pathogen_names"]) <= {"rota", "campy"}


def test_rejects_both_pathogens_disabled(client):
    resp = client.post("/api/scenario/run", json=_minimal_payload(rota_enabled=False, campy_enabled=False))
    assert resp.status_code == 422
    assert "pathogen" in resp.json()["detail"].lower()


def test_rejects_out_of_bounds_value(client):
    resp = client.post("/api/scenario/run", json=_minimal_payload(number_agents=999_999))
    assert resp.status_code == 422


def test_rejects_unknown_field(client):
    resp = client.post("/api/scenario/run", json=_minimal_payload(not_a_real_field=1))
    assert resp.status_code == 422


def test_queue_full_returns_429(client, monkeypatch):
    monkeypatch.setattr(executor, "MAX_ACCEPTED_JOBS", 0)
    resp = client.post("/api/scenario/run", json=_minimal_payload())
    assert resp.status_code == 429


def test_requires_auth(anon_client):
    resp = anon_client.post("/api/scenario/run", json=_minimal_payload())
    assert resp.status_code == 401
