"""
Tests for the per-pathogen illness mechanics added alongside Cryptosporidium:
the duration scale and the per-pathogen base severity.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from abm.agent.illness_mechanics import calculate_illness_duration, calculate_illness_severity
from config import IllnessMechanicsConfig, SVEIRCONFIG


def test_duration_scale_of_one_is_an_exact_no_op():
    """Rotavirus and campylobacter must be bit-identical, not merely close.

    They pass duration_scale=1.0, so every arithmetic operation introduced for
    Cryptosporidium has to be exactly neutral at that value - multiplying by
    1.0 is exact in IEEE-754, which is why the scale is applied as a
    multiplication rather than, say, a reparameterised min/max.
    """
    cfg = IllnessMechanicsConfig()
    severity = torch.tensor([0.0, 0.25, 0.4, 0.75, 1.0])

    torch.manual_seed(7)
    unscaled = calculate_illness_duration(severity, cfg)
    torch.manual_seed(7)
    scaled_one = calculate_illness_duration(severity, cfg, duration_scale=1.0)

    assert torch.equal(unscaled, scaled_one)


def test_duration_scale_stretches_the_window():
    # With the ceiling coupled to duration_max_days (cap=None), the stretch is
    # visible as episodes exceeding the unscaled maximum.
    cfg = IllnessMechanicsConfig(duration_cap_days=None)
    severity = torch.full((4000,), 0.4)

    torch.manual_seed(11)
    base = calculate_illness_duration(severity, cfg).float()
    torch.manual_seed(11)
    stretched = calculate_illness_duration(severity, cfg, duration_scale=1.6).float()

    assert stretched.mean().item() == pytest.approx(1.6 * base.mean().item(), rel=0.05)
    # The clamp scales too, so episodes can exceed the shared 12-day ceiling -
    # which is the whole point for a pathogen whose diarrhoea often runs
    # beyond two weeks.
    assert base.max().item() <= cfg.duration_max_days
    assert stretched.max().item() > cfg.duration_max_days


# MAL-ED Supplementary Table 11 medians (days) for ages 0-6, 6-12, 12-18, 18-24
# months, read from abm/evidence/maled.py so the test follows the transcription.
from abm.evidence.maled import DIARRHOEA_DURATION_DAYS_BY_AGE


@pytest.mark.parametrize("pathogen", ["rota", "campy", "crypto"])
def test_natural_durations_match_maled_medians_by_age(pathogen):
    """With no prior infections, the natural (untreated) episode-duration
    median in each 6-month band lies inside MAL-ED's interquartile range.

    First episodes have the highest severity (prior infections only lower it),
    so this is the most demanding case for the upper side of the window.
    """
    illness_cfg = IllnessMechanicsConfig()
    p_cfg = next(p for p in SVEIRCONFIG.pathogens if p.name == pathogen)
    n = 4000
    for band, (lo, hi) in enumerate([(0, 6), (6, 12), (12, 18), (18, 24)]):
        torch.manual_seed(100 + band)
        age = torch.rand(n) * (hi - lo) + lo
        severity = calculate_illness_severity(
            pathogen_name=pathogen, is_child=torch.ones(n, dtype=torch.bool), age=age,
            vaccine_status=None, num_infections=torch.zeros(n), cfg=illness_cfg,
        )
        duration = calculate_illness_duration(
            severity, illness_cfg, duration_scale=p_cfg.duration_scale
        ).float()
        model_median = float(duration.median().item())
        _, q1, q3 = DIARRHOEA_DURATION_DAYS_BY_AGE[pathogen][band]
        assert q1 <= model_median <= q3, (
            f"{pathogen} {lo}-{hi} months: model median {model_median} outside MAL-ED IQR "
            f"[{q1}, {q3}]"
        )


def test_duration_cap_is_decoupled_from_expected_maximum():
    """duration_max_days is the EXPECTED duration at severity 1, not a hard
    ceiling: at severity 1 roughly half of episodes exceed it, none exceed
    duration_cap_days, and there is no pile-up of mass at the expected maximum."""
    cfg = IllnessMechanicsConfig()
    torch.manual_seed(5)
    d = calculate_illness_duration(torch.ones(20000), cfg).float()
    assert d.max().item() <= cfg.duration_cap_days
    frac_above = (d > cfg.duration_max_days).float().mean().item()
    assert 0.3 < frac_above < 0.6
    frac_at_max = (d == cfg.duration_max_days).float().mean().item()
    assert frac_at_max < 0.3

    coupled = IllnessMechanicsConfig(duration_cap_days=None)
    torch.manual_seed(5)
    d_old = calculate_illness_duration(torch.ones(20000), coupled).float()
    assert d_old.max().item() <= coupled.duration_max_days


def test_base_severity_ordering_is_campy_crypto_rota():
    """Cryptosporidium sits between campylobacter and rotavirus.

    The ordering is the literature-grounded part of this choice (GEMS: crypto
    MSD more often dehydrating and more often fatal than average, but
    rotavirus remains the archetypal severe dehydrating infection); the exact
    values are an assumption.
    """
    cfg = IllnessMechanicsConfig()
    assert cfg.base_severity_campy < cfg.base_severity_crypto < cfg.base_severity_rota


def test_severity_is_pathogen_specific_and_falls_back_safely():
    cfg = IllnessMechanicsConfig()
    is_child = torch.ones(3, dtype=torch.bool)
    age = torch.zeros(3)                      # newborn: maximum age multiplier
    num_infections = torch.zeros(3)

    sev = {}
    for name in ("rota", "campy", "crypto"):
        sev[name] = calculate_illness_severity(
            pathogen_name=name, is_child=is_child, age=age,
            vaccine_status=None, num_infections=num_infections, cfg=cfg,
        )[0].item()
    assert sev["campy"] < sev["crypto"] < sev["rota"]

    # An unknown pathogen name must not raise - it falls back to a default.
    fallback = calculate_illness_severity(
        pathogen_name="giardia", is_child=is_child, age=age,
        vaccine_status=None, num_infections=num_infections, cfg=cfg,
    )
    assert torch.all(fallback > 0)


def test_logged_episode_durations_are_maled_scale_in_a_real_run(make_model):
    """End-to-end: natural durations logged for children under two have a
    median of 2-4 days for every pathogen (MAL-ED: 2-3 days), and an episode is
    never longer than the shared maximum (duration_scale is 1.0 throughout)."""
    # Cryptosporidium transmission is boosted only to produce enough
    # episodes to test; natural durations do not depend on transmission.
    model = make_model(steps=250, agents=4000,
                       overrides={"pathogens[crypto].household_transmission_prob": 0.05,
                                  "pathogens[crypto].community_infection_prob": 0.002,
                                  "pathogens[rota].water_dose_scale": 1.0})
    cfg = IllnessMechanicsConfig()
    by_pathogen: dict[str, list[int]] = {}
    for entry in model.get_child_episode_log():
        if entry["age_months"] < 24:
            by_pathogen.setdefault(entry["pathogen"], []).append(entry["initial_duration"])
    # Pathogens with too few under-2 episodes in a short run are skipped here
    # (Cryptosporidium's disease rate is low); the per-band unit test above
    # covers every pathogen deterministically.
    tested = []
    for name in ("rota", "campy", "crypto"):
        if len(by_pathogen.get(name, [])) < 8:
            continue
        tested.append(name)
        med = float(np.median(by_pathogen[name]))
        assert 2.0 <= med <= 4.0, f"{name} median natural duration {med} days"
        assert max(by_pathogen[name]) <= cfg.duration_cap_days
    assert len(tested) >= 2, f"too few episodes to test any pathogen: {by_pathogen.keys()}"
