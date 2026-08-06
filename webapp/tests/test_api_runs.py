"""Tests for GET /api/runs, GET /api/runs/{id}, GET /api/runs/{id}/data.json
that don't require actually executing a simulation - they exercise the
routes directly against jobs.py, using its real (SQLite-backed, isolated
per test via the isolated_db fixture) storage rather than mocking it."""
from __future__ import annotations

from webapp import jobs
from webapp.jobs import SimResultBundle


def _fake_result(**overrides) -> SimResultBundle:
    defaults = dict(
        config_snapshot={"seed": 1},
        pathogen_names=["rota"],
        days=[0, 1],
        u5_prevalence={"rota": [0.1, 0.2]},
        all_ages_prevalence={"rota": [0.05, 0.1]},
        cumulative_u5_illness_days={"rota": [1.0, 2.0]},
        mean_household_wealth=[0.5, 0.5],
        cumulative_care_seeking_events=[0.0, 1.0],
        spatial_grid_size=25,
        spatial_daily_grids=[[[0.0] * 25 for _ in range(25)]] * 2,
        summary_metrics={"peak_prevalence": 0.2},
        proportion_infected_at_least_once=0.3,
        n_u5=42,
        runtime_seconds=1.23,
    )
    defaults.update(overrides)
    return SimResultBundle(**defaults)


def test_run_detail_not_found(client):
    resp = client.get("/api/runs/does-not-exist")
    assert resp.status_code == 404


def test_run_detail_queued(client):
    record = jobs.new_job({"seed": 7, "number_agents": 500})
    resp = client.get(f"/api/runs/{record.job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert body["config_form"] == {"seed": 7, "number_agents": 500}
    assert body["result"] is None


def test_run_detail_done_includes_result(client):
    record = jobs.new_job({"seed": 7})
    jobs.set_done(record.job_id, _fake_result())
    resp = client.get(f"/api/runs/{record.job_id}")
    body = resp.json()
    assert body["status"] == "done"
    assert body["result"]["n_u5"] == 42
    assert body["config_form"] == {"seed": 7}


def test_run_detail_error(client):
    record = jobs.new_job({})
    jobs.set_error(record.job_id, "simulation blew up")
    resp = client.get(f"/api/runs/{record.job_id}")
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"] == "simulation blew up"


def test_list_runs_returns_summaries_not_full_results(client):
    record = jobs.new_job({"seed": 1})
    jobs.set_done(record.job_id, _fake_result())
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["runs"]) == 1
    run = body["runs"][0]
    assert run["job_id"] == record.job_id
    assert run["summary"]["n_u5"] == 42
    # The list view must not carry the heavy per-day series/spatial grids.
    assert "days" not in run["summary"]
    assert "spatial_daily_grids" not in run["summary"]


def test_list_runs_respects_limit(client):
    for i in range(3):
        jobs.new_job({"seed": i})
    resp = client.get("/api/runs?limit=2")
    assert len(resp.json()["runs"]) == 2


def test_run_data_json_not_found_before_done(client):
    record = jobs.new_job({})
    resp = client.get(f"/api/runs/{record.job_id}/data.json")
    assert resp.status_code == 404


def test_run_data_json_matches_result_once_done(client):
    record = jobs.new_job({})
    jobs.set_done(record.job_id, _fake_result())
    resp = client.get(f"/api/runs/{record.job_id}/data.json")
    assert resp.status_code == 200
    assert resp.json()["n_u5"] == 42


def test_requires_auth(anon_client):
    resp = anon_client.get("/api/runs")
    assert resp.status_code == 401
