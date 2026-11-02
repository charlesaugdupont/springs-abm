"""
Rotavirus vaccination by schedule (RotavirusConfig.vaccination_model = "schedule").

The default stays "hazard" (the original constant daily rate; checked byte for
byte by experiments/perf/regression_check.py). These tests cover the schedule:
coverage and ages from the DHS-based tables, take, entry into V, waning, and
a renewed slot drawing a new schedule.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from config import SVEIRCONFIG, VACCINE_AGE_PERCENTILES
from abm.constants import AgentPropertyKeys as K, Compartment
from experiments.metrics import vaccination_metrics

SCHEDULE = {"pathogens[rota].vaccination_model": "schedule"}


def _rota(model):
    return next(p for p in model.pathogens if p.name == "rota")


def test_hazard_is_the_default():
    rota = next(p for p in SVEIRCONFIG.pathogens if p.name == "rota")
    assert rota.vaccination_model == "hazard"
    assert len(rota.vaccine_dose1_age_days) == len(VACCINE_AGE_PERCENTILES) == 21


def test_schedule_draws_follow_the_tables(make_model):
    model = make_model(run=False, agents=3000, overrides=SCHEDULE)
    rota = _rota(model)
    rota._vax_rng = np.random.default_rng(5)
    d = rota._draw_schedules(20000)
    cfg = rota.config
    planned = d["planned"].numpy()
    assert (planned >= 1).mean() == pytest.approx(cfg.vaccine_dose_continuation[0], abs=0.01)
    assert (planned >= 2).mean() == pytest.approx(cfg.vaccine_dose_continuation[0] * cfg.vaccine_dose_continuation[1], abs=0.01)
    assert np.median(d["ages"][:, 0].numpy()) == pytest.approx(48.0, abs=1.0)
    gaps = np.diff(d["ages"].numpy(), axis=1)
    assert np.median(gaps) == pytest.approx(30.0, abs=1.5) and gaps.min() >= cfg.vaccine_dose_interval_days[0] - 1e-9
    assert d["takes"][:, 0].double().mean().item() == pytest.approx(cfg.vaccine_take_by_dose[0], abs=0.01)
    assert d["lasts"].mean().item() == pytest.approx(cfg.vaccine_protection_days, rel=0.03)


def test_children_enter_v_only_when_a_dose_takes(make_model):
    model = make_model(run=False, agents=3000, overrides=SCHEDULE)
    rota, nd = _rota(model), model.graph.ndata
    rota._susceptible_to_vaccinated(model.graph)          # day 0: doses the children's ages have passed
    v = rota._vax
    status, ever = nd[K.status("rota")], nd[K.ever_vaccinated("rota")]
    child = nd[K.IS_CHILD].bool()
    assert torch.all(ever == v["protected"]), "only children whose vaccine took are ever_vaccinated"
    assert not torch.any((status == Compartment.VACCINATED) & ~v["protected"])
    assert torch.all(v["given"][~child] == 0), "adults are not vaccinated"
    age_days = nd[K.AGE].double() * 365.25 / 12
    older = child & (age_days > 400)
    assert v["given"][older].ge(1).double().mean().item() == pytest.approx(rota.config.vaccine_dose_continuation[0], abs=0.04)
    assert torch.all(v["given"][child & (age_days < 20)] == 0)


def test_protection_wanes(make_model):
    model = make_model(run=False, agents=3000, overrides={**SCHEDULE, "pathogens[rota].vaccine_protection_days": 5.0})
    rota, nd = _rota(model), model.graph.ndata
    for _ in range(80):
        rota._susceptible_to_vaccinated(model.graph)
        nd[K.AGE] += 12.0 / 365.25
    status = nd[K.status("rota")]
    v = rota._vax
    # everyone protected more than ~40 mean durations ago has left V
    assert not torch.any((status == Compartment.VACCINATED) & (v["wane_day"] <= rota._vax_day - 1))
    assert v["protected"].any()


def test_a_renewed_slot_draws_a_new_schedule(make_model):
    model = make_model(run=False, agents=3000, overrides=SCHEDULE)
    rota, nd = _rota(model), model.graph.ndata
    rota._susceptible_to_vaccinated(model.graph)
    child = nd[K.IS_CHILD].bool()
    i = int(((rota._vax["given"] > 0) & child).nonzero()[0])
    nd[K.AGE][i] = 0.0
    nd[K.BIRTH_DAY][i] = 5
    nd[K.ever_vaccinated("rota")][i] = False
    nd[K.status("rota")][i] = Compartment.SUSCEPTIBLE
    rota._susceptible_to_vaccinated(model.graph)
    assert int(rota._vax["given"][i]) == 0 and not bool(rota._vax["protected"][i])


def test_vaccination_metrics(make_model):
    model = make_model(steps=30, agents=3000, overrides=SCHEDULE)
    m = vaccination_metrics(model)
    assert m["rota_vaccine_dose1_12_23m"] >= m["rota_vaccine_dose2_12_23m"] >= m["rota_vaccine_dose3_12_23m"]
    assert 0.0 <= m["rota_vaccine_in_V_12_23m"] <= m["rota_vaccine_protected_ever_12_23m"] <= 1.0
    assert vaccination_metrics(make_model(steps=3, agents=600)) == {}


def test_a_vaccine_response_can_count_as_a_prior_infection(make_model):
    from abm.systems.child_illness import symptomatic_probability
    model = make_model(run=False, agents=600, overrides={**SCHEDULE, "pathogens[rota].vaccine_counts_as_prior_infection": True})
    cfg = _rota(model).config
    ages = torch.tensor([8.0, 8.0])
    p = symptomatic_probability(cfg, ages, torch.tensor([0, 1]))
    assert float(p[1] / p[0]) == pytest.approx(np.exp(-cfg.symptomatic_reduction_per_prior_infection), rel=1e-6)
    assert SVEIRCONFIG.pathogens[0].vaccine_counts_as_prior_infection is False


def test_rota_severity_metrics(make_model):
    from experiments.metrics import rota_severity_metrics, SEVERITY_THRESHOLDS
    model = make_model(steps=60, agents=2000)
    m = rota_severity_metrics(model, burn_in_days=0)
    for band in ("0_12m", "12_24m", "24_60m"):
        counts = [m[f"rota_episodes_sev{int(round(t * 10)):02d}_{band}"] for t in SEVERITY_THRESHOLDS]
        assert counts == sorted(counts, reverse=True), "fewer episodes at higher severity"


def test_the_akuse_vaccination_preset_is_valid(make_model):
    from experiments.presets import AKUSE_VACCINATION_2026
    model = make_model(run=False, agents=600, overrides=AKUSE_VACCINATION_2026)
    cfg = _rota(model).config
    assert cfg.vaccination_model == "schedule" and cfg.vaccine_counts_as_prior_infection is True
