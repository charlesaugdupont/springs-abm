# experiments/presets.py
"""
Named sets of configuration overrides (orchestrator dot-paths) for settings that
are not the model's defaults but are the recommended way to run a particular
setting. Kept separate from experiments/legacy.py, which pins EARLIER model
generations.
"""

#: Rotavirus vaccination as in Akuse today (docs/vaccination_schedule.md,
#: adopted 2026-11-02): doses by age with Ghana DHS 2022 coverage and timing,
#: ROTAVAC take and waning (Asare et al. 2026), and a vaccine response
#: protecting against illness like one natural infection. The model's default
#: stays the constant-hazard vaccination of campaigns v1-v4.
AKUSE_VACCINATION_2026: dict = {
    "pathogens[rota].vaccination_model": "schedule",
    "pathogens[rota].vaccine_counts_as_prior_infection": True,
}
