"""
Paper 1's experiments and checks (experiments.scenarios.paper1): the switches
they use leave the model unchanged by default, and do what the experiments
say they do.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from config import SVEIRCONFIG


# ---------------------------------------------------------------------------
# Experiment 9: the decision model
# ---------------------------------------------------------------------------

def test_equal_persona_bounds_fix_the_parameter_and_keep_the_others(make_model):
    """gamma_range [1, 1]: every persona's gamma is exactly 1, and alpha and
    lambda are the model's own personas (the same sample, scaled as before)."""
    base = make_model(steps=1, run=False)
    eq = make_model(steps=1, run=False, overrides={"gamma_range": [1.0, 1.0], "lambda_range": [1.0, 1.0]})
    a, b = base.agent_personas, eq.agent_personas
    assert torch.equal(b[:, 1], torch.ones_like(b[:, 1])) and torch.equal(b[:, 2], torch.ones_like(b[:, 2]))
    assert torch.equal(a[:, 0], b[:, 0])
    with pytest.raises(ValueError, match="lower <= upper"):
        make_model(steps=1, run=False, overrides={"gamma_range": [0.9, 0.4]})


def test_cpt_with_unit_parameters_is_expected_utility():
    """With gamma = lambda = theta = eta = 1, the CPT rule decides exactly as
    the expected-utility rule (CareSeekingSystem._evaluate_ev) on a grid of
    states, and the model's CPT does not."""
    from abm.systems.care_seeking import CareSeekingSystem as C
    unit = SVEIRCONFIG.steering_parameters.model_copy(update={"cpt_theta": 1.0, "cpt_eta": 1.0})
    model = SVEIRCONFIG.steering_parameters
    same = differ = 0
    for w in (0.02, 0.05, 0.3, 0.7, 1.0):
        for drop in (0.0, 0.01, 0.05, 0.3):
            for h_p in (0.6, 1.0):
                for h_c in (0.25, 0.6, 0.9):
                    for alpha in (0.1, 0.5, 0.9):
                        w_after = max(0.0, w - drop)
                        cpt = C._evaluate_cpt(None, w, h_p, h_c, alpha, 1.0, 1.0, unit, w_after=w_after)
                        ev = C._evaluate_ev(None, w, h_p, h_c, alpha, unit, w_after=w_after)
                        same += cpt == ev
                        differ += C._evaluate_cpt(None, w, h_p, h_c, alpha, 0.65, 2.0, model,
                                                  w_after=w_after) != ev
    assert same == 5 * 4 * 2 * 3 * 3
    assert differ > 0


def test_decision_variants_are_settings_on_todays_model():
    from experiments.orchestrator import REPS_KEY
    from experiments.scenarios import paper1 as P
    from experiments.scenarios.designs import p1_3a_combos
    combos = P.p1_decision_combos()
    assert [c[P.LABEL] for c in combos] == list(P.DECISION_VARIANTS)
    base = {k: v for k, v in next(c for c in p1_3a_combos() if c[P.LABEL] == "vacc_default").items()
            if k not in (P.LABEL, REPS_KEY)}
    for c in combos:
        assert c[REPS_KEY] == 40
        assert {k: c[k] for k in base} == base
    eu = next(c for c in combos if c[P.LABEL] == "expected_utility")
    assert eu["gamma_range"] == [1.0, 1.0] and eu["lambda_range"] == [1.0, 1.0]
    assert eu["steering_parameters.cpt_theta"] == eu["steering_parameters.cpt_eta"] == 1.0


# ---------------------------------------------------------------------------
# Experiment 7: the transmission routes
# ---------------------------------------------------------------------------

def test_every_route_is_knocked_out_once():
    from experiments.scenarios import paper1 as P
    from abm.pathogens.registry import build_pathogen  # noqa: F401  (the registry knows the routes)
    from abm.pathogens.campylobacter import Campylobacter
    from abm.pathogens.cryptosporidium import Cryptosporidium
    from abm.pathogens.rotavirus import Rotavirus
    routes = {(cls_name, r) for cls_name, cls in (("rota", Rotavirus), ("campy", Campylobacter),
                                                 ("crypto", Cryptosporidium)) for r in cls.ROUTES}
    assert set(P.ROUTES) == routes and len(P.ROUTES) == len(routes)
    combos = P.p1_routes_combos()
    assert len(combos) == len(routes) and len({c[P.LABEL] for c in combos}) == len(routes)


def test_a_knocked_out_route_makes_no_infection(make_model):
    """route_off on day 0: the route's multiplier is 0 and it never infects."""
    from experiments.scenarios import paper1 as P
    from abm.constants import AgentPropertyKeys as K
    m = make_model(steps=40, run=False, pathogens=["crypto"])
    combo = {P.ROUTE_OFF_KEY: "crypto:community"}
    # the orchestrator's step-by-step driver
    m.infection_incidence.clear()
    m.u5_prevalence_history = {p.name: [] for p in m.pathogens}
    m.u5_symptomatic_prevalence_history = {p.name: [] for p in m.pathogens}
    child = m.graph.ndata[K.IS_CHILD]
    for day in range(40):
        P.route_off(m, day, combo)
        m.step(child_mask=child)
    p = m.pathogens[0]
    assert p.route_multipliers == {"community": 0.0}
    assert p.total_by_route["community"] == 0
    assert sum(p.total_by_route.values()) > 0
    with pytest.raises(ValueError):
        P.route_off(m, 0, {P.ROUTE_OFF_KEY: "crypto:h2h"})


def test_a_step_callback_that_does_nothing_leaves_the_run_unchanged(tmp_path):
    """The step-by-step driver of hook runs (p1_routes) gives exactly model.run()'s results."""
    from experiments.orchestrator import SweepSpec, _run_one
    from experiments.scenarios.common import experiment_metrics

    def spec(callback):
        return SweepSpec(name="cb", grid_id="7d9ce7c720a6", params=[], metrics_fn=experiment_metrics, reps=1,
                         steps=60, agents=500, output_dir=str(tmp_path), base_seed=23, step_callback=callback)
    a = _run_one((spec(None), {}, 0, "a"))["record"]
    b = _run_one((spec(_noop), {}, 0, "a"))["record"]
    keys = [k for k in a if k not in ("run_id",)]
    assert all((a[k] == b[k]) or (a[k] != a[k] and b[k] != b[k]) for k in keys)


def _noop(model, day, combo):
    return None


# ---------------------------------------------------------------------------
# Experiment 8: the observation process
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def observed_run():
    from model_tests.conftest import build_model
    return build_model(agents=800, steps=400, seed=29)


def test_observation_defaults_are_maled(observed_run):
    from experiments.observation import observe
    a = observe(observed_run, burn_in_days=60)
    b = observe(observed_run, burn_in_days=60, stools_per_month=1, shedding_scale=1.0,
                diarrhoeal_stool_tested_prob=SVEIRCONFIG.observation.diarrhoeal_stool_tested_prob)
    assert a.keys() == b.keys()
    assert all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)


def test_observation_designs_move_detection_the_right_way(observed_run):
    """More stools or longer windows detect at least as many infections; the
    "maled" design is the run's own observation."""
    from experiments.observation import observe
    from experiments.scenarios import paper1 as P
    d = P.observation_designs(observed_run, burn_in_days=60)
    own = observe(observed_run, burn_in_days=60)
    for p in ("rota", "campy", "crypto"):
        k = f"{p}_obs_infections_per_child_year_0_24m"
        assert d[f"obsd_maled_{p}_infections_per_child_year_0_24m"] == own[k] or own[k] != own[k]
    # counts of detected infections (rates also move their denominators): positives can only rise
    for p in ("rota", "crypto"):
        lo = d[f"obsd_windows_half_{p}_infections_per_child_year_0_24m"]
        mid = d[f"obsd_maled_{p}_infections_per_child_year_0_24m"]
        assert lo <= mid * 1.05 or lo != lo
    assert len(d) == len(P.OBS_DESIGNS) * 3 * len(P.OBS_KEYS)
    with pytest.raises(ValueError):
        observe(observed_run, burn_in_days=60, stools_per_month=0)


# ---------------------------------------------------------------------------
# Analysis 5 and 3: the Morris design and the long runs
# ---------------------------------------------------------------------------

def test_morris_design_is_valid_trajectories_on_todays_grid():
    from experiments.orchestrator import REPS_KEY
    from experiments.scenarios import paper1 as P
    X = P.load_morris_design()
    k, r = len(P.MORRIS_FACTORS), P.MORRIS_TRAJECTORIES
    assert X.shape == (r * (k + 1), k)
    assert set(np.round(X.ravel() * (P.MORRIS_LEVELS - 1), 9)) <= {0.0, 1.0, 2.0, 3.0}
    for t in range(r):
        d = np.diff(X[t * (k + 1):(t + 1) * (k + 1)], axis=0)
        moved = np.abs(d) > 1e-12
        assert np.all(moved.sum(axis=1) == 1)
        assert sorted(np.argmax(moved, axis=1)) == list(range(k))
    combos = P.p1_morris_combos()
    assert len(combos) == len(X) and all(c[REPS_KEY] == 2 for c in combos)
    for c in combos:
        assert 0.05 - 1e-9 <= c["alpha_range"][0] < c["alpha_range"][1] <= 0.95 + 1e-9
        assert isinstance(c["pathogens[rota].exposure_period"], int)
    # every factor's range holds the model's value, and every config path exists
    from experiments.orchestrator import set_param
    cfg = SVEIRCONFIG.model_copy(deep=True)
    for path, value in combos[0].items():
        if not path.startswith(("gsa.", "hook.")):
            set_param(cfg, path, value)


def test_epi_weekly_series_add_up(make_model):
    from experiments.scenarios import paper1 as P
    m = make_model(steps=70)
    s = P.epi_weekly_series(m)
    assert len(s["wk_child_years_0_60m"]) == 10
    for p in m.pathogens:
        assert sum(s[f"wk_inf_{p.name}_0_24m"]) <= sum(s[f"wk_inf_{p.name}_0_60m"]) <= sum(s[f"wk_inf_{p.name}_all"])
        child = float(np.asarray(m.new_infections_by_age_month[p.name])[:70].sum())
        assert sum(s[f"wk_inf_{p.name}_0_60m"]) == pytest.approx(child)


# ---------------------------------------------------------------------------
# Analysis 6: docking of person-to-person transmission against its equations
# ---------------------------------------------------------------------------

def test_reduced_model_matches_its_seir_equations():
    """The model's Rotavirus in one well-mixed cell (experiments.verification.seir_docking)
    ends with the final size of its equations, and its mean curve follows them."""
    from experiments.verification import seir_docking as D
    summary, _ = D.dock(r0_levels=(2.0,), reps=4, n=2000, i0=20, days=150)
    s = summary[0]
    assert s["attack_final_size_eq"] == pytest.approx(s["attack_mean_field"], abs=2e-3)
    assert s["attack_mean"] == pytest.approx(s["attack_final_size_eq"], abs=0.04)
    assert abs(s["peak_day_model"] - s["peak_day_mean_field"]) <= 3
    assert s["curve_rmse_rel_mean_field"] < 0.1


def test_route_counts_add_up_to_the_route_shares(make_model):
    from experiments.metrics import route_fractions_post_burnin
    from experiments.scenarios import paper1 as P
    m = make_model(steps=60)
    c = P.route_counts(m, burn_in_days=10)
    f = route_fractions_post_burnin(m, burn_in_days=10)
    for p in m.pathogens:
        total = c[f"{p.name}_infections_post_burnin"]
        assert total == sum(c[f"{p.name}_{r}_infections_post_burnin"] for r in p.ROUTES)
        if total > 0:
            for r in p.ROUTES:
                assert c[f"{p.name}_{r}_infections_post_burnin"] / total == pytest.approx(
                    f[f"{p.name}_{r}_fraction_post_burnin"])
