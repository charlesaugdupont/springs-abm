"""Tests for GET /api/parameters - the JSON view over parameter_registry.py
that the frontend's scenario form and About page both render from."""
from __future__ import annotations

from webapp.parameter_registry import REGISTRY, editable_fields
from webapp.scenario_form import ScenarioFormInput


def test_parameters_response_shape(client):
    resp = client.get("/api/parameters")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"category_order", "by_category", "by_evidence_tier"}


def test_editable_count_matches_registry(client):
    resp = client.get("/api/parameters")
    body = resp.json()
    total_editable = sum(len(cat["editable"]) for cat in body["by_category"])
    assert total_editable == len(editable_fields())


def test_internal_category_is_excluded(client):
    resp = client.get("/api/parameters")
    body = resp.json()
    categories = {cat["category"] for cat in body["by_category"]}
    assert "internal" not in categories


def test_by_evidence_tier_covers_every_non_internal_entry(client):
    resp = client.get("/api/parameters")
    body = resp.json()
    total_in_tiers = sum(len(tier["params"]) for tier in body["by_evidence_tier"])
    non_internal = [m for m in REGISTRY if m.category != "internal"]
    assert total_in_tiers == len(non_internal)


def test_editable_entries_carry_form_name_and_default(client):
    resp = client.get("/api/parameters")
    body = resp.json()
    pop_cat = next(c for c in body["by_category"] if c["category"] == "Population & Demographics")
    number_agents = next(p for p in pop_cat["editable"] if p["path"] == "number_agents")
    assert number_agents["form_name"] == "number_agents"
    assert number_agents["default"] == ScenarioFormInput().number_agents
    assert number_agents["ui_min"] == 500
    assert number_agents["ui_max"] == 10_000


def test_corrected_recovery_rate_bounds_are_reflected(client):
    resp = client.get("/api/parameters")
    body = resp.json()
    rota_cat = next(c for c in body["by_category"] if c["category"] == "Rotavirus")
    recovery = next(p for p in rota_cat["editable"] if p["path"] == "pathogens[rota].recovery_rate")
    assert recovery["ui_min"] == 0.14
    assert recovery["ui_max"] == 0.33


def test_requires_auth(anon_client):
    resp = anon_client.get("/api/parameters")
    assert resp.status_code == 401
