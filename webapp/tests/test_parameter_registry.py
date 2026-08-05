"""
Completeness check for webapp/parameter_registry.py - the anti-drift
enforcement mechanism described in that module's docstring. If a new field
is added to SVEIRConfig without a matching registry entry, this test fails
loudly instead of the scenario builder / About page silently missing it.

Run with: pytest  (see repo-root pytest.ini for why --pyargs is the default)
"""
from webapp.parameter_registry import (
    REGISTRY, REGISTRY_BY_PATH, iter_config_paths, CATEGORY_ORDER, EVIDENCE_TIER_ORDER,
)


def test_every_registry_path_is_unique():
    paths = [m.path for m in REGISTRY]
    duplicates = {p for p in paths if paths.count(p) > 1}
    assert not duplicates, f"Duplicate registry entries for: {duplicates}"


def test_every_config_field_has_a_registry_entry():
    config_paths = set(iter_config_paths())
    registry_paths = set(REGISTRY_BY_PATH.keys())

    missing = config_paths - registry_paths
    assert not missing, (
        f"SVEIRConfig field(s) with no parameter_registry.py entry: {sorted(missing)}. "
        "Add a ParamMeta for each - this is the mechanism that keeps the scenario "
        "builder and About page in sync with the model (see module docstring)."
    )


def test_every_registry_entry_points_at_a_real_config_field():
    config_paths = set(iter_config_paths())
    registry_paths = set(REGISTRY_BY_PATH.keys())

    stale = registry_paths - config_paths
    assert not stale, (
        f"parameter_registry.py entries with no matching SVEIRConfig field (stale, "
        f"likely renamed/removed): {sorted(stale)}"
    )


def test_every_entry_has_a_known_category_or_is_internal():
    valid = set(CATEGORY_ORDER) | {"internal"}
    bad = [m.path for m in REGISTRY if m.category not in valid]
    assert not bad, f"Registry entries with an unrecognized category: {bad}"


def test_every_entry_has_a_valid_evidence_tier():
    bad = [m.path for m in REGISTRY if m.evidence_tier not in EVIDENCE_TIER_ORDER]
    assert not bad, f"Registry entries with an unrecognized evidence_tier: {bad}"


def test_editable_fields_have_rationale_text():
    thin = [m.path for m in REGISTRY if m.editable and len(m.rationale) < 20]
    assert not thin, f"Editable fields need a real rationale (info-box content): {thin}"


def test_editable_numeric_fields_have_ui_bounds():
    missing_bounds = [
        m.path for m in REGISTRY
        if m.editable and m.ui_widget in ("slider", "number", "number+randomize-button")
        and (m.ui_min is None or m.ui_max is None)
    ]
    assert not missing_bounds, f"Editable numeric fields missing ui_min/ui_max: {missing_bounds}"
