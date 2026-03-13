"""Unit tests for ship_layout.validator.PlacementValidator."""

from __future__ import annotations

import pytest

from common.geometry import load_vanilla_part_geometry
from ship_layout.validator import PlacementValidator, ValidationResult


@pytest.fixture(scope="module")
def geometry_cache():
    return load_vanilla_part_geometry()


def _make_validator(geometry_cache, *, mirror_mode=False, allowlist=None, requirements=None):
    return PlacementValidator(
        geometry_cache,
        min_x=-20,
        max_x=20,
        min_y=-20,
        max_y=20,
        mirror_mode=mirror_mode,
        part_allowlist=allowlist,
        part_requirements=requirements,
    )


# ---------------------------------------------------------------------------
# validate_candidate — geometry_unknown
# ---------------------------------------------------------------------------


def test_validate_candidate_geometry_unknown_rejects_missing_part(geometry_cache):
    """validate_candidate should reject an unknown part_id."""
    validator = _make_validator(geometry_cache)
    anchor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 0}
    candidate = {"part_id": "cosmoteer.does_not_exist", "rotation": 0, "x": 1, "y": 0}
    result = validator.validate_candidate(candidate, anchor, set())
    assert not result.accepted
    assert result.rejection == "geometry_unknown"
    assert result.primary_cells is None
    assert result.mirror_companion is None



# ---------------------------------------------------------------------------
# validate_candidate — allowlist
# ---------------------------------------------------------------------------


def test_validate_candidate_allowlist_rejects_unlisted_part(geometry_cache):
    """validate_candidate should reject parts absent from a configured allowlist."""
    validator = _make_validator(geometry_cache, allowlist={"cosmoteer.armor_2x1"})
    anchor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 0}
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 1, "y": 0}
    result = validator.validate_candidate(candidate, anchor, set())
    assert not result.accepted
    assert result.rejection == "allowlist"


def test_validate_candidate_allowlist_accepts_listed_part(geometry_cache):
    """validate_candidate should pass the allowlist for a listed part."""
    validator = _make_validator(geometry_cache, allowlist={"cosmoteer.armor"})
    anchor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": -1, "y": 0}
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 0}
    # Note: connectivity may still fail — we only test the allowlist is passed
    result = validator.validate_candidate(candidate, anchor, set())
    assert result.rejection != "allowlist"


def test_validate_candidate_no_allowlist_passes_any_part(geometry_cache):
    """When no allowlist is configured, all known parts should pass that check."""
    validator = _make_validator(geometry_cache, allowlist=None)
    anchor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": -1, "y": 0}
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 0}
    result = validator.validate_candidate(candidate, anchor, set())
    assert result.rejection != "allowlist"


# ---------------------------------------------------------------------------
# validate_candidate — connectivity
# ---------------------------------------------------------------------------


def test_validate_candidate_connectivity_rejects_non_touching_pair(geometry_cache):
    """validate_candidate should reject placements with no structural hull contact."""
    validator = _make_validator(geometry_cache)
    # Anchor at (0, 0), candidate far away at (5, 5): no shared hull side
    anchor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 0}
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 5, "y": 5}
    result = validator.validate_candidate(candidate, anchor, set())
    assert not result.accepted
    assert result.rejection == "connectivity"


def test_validate_candidate_connectivity_accepts_adjacent_parts(geometry_cache):
    """validate_candidate should pass connectivity for side-adjacent 1×1 parts."""
    validator = _make_validator(geometry_cache)
    anchor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": -1, "y": 0}
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 0}
    result = validator.validate_candidate(candidate, anchor, set())
    # This test specifically targets connectivity; note the assertion tests that
    # connectivity was not the rejection reason rather than full acceptance.
    # The placement is within bounds and non-overlapping, so acceptance is expected.
    assert result.accepted, f"Expected accepted but got rejection: {result.rejection}"


# ---------------------------------------------------------------------------
# validate_candidate — mirror mode
# ---------------------------------------------------------------------------


def test_validate_candidate_mirror_mode_returns_companion_with_distinct_cells(geometry_cache):
    """Mirror mode: accepted placement should carry a non-None companion with different cells."""
    validator = _make_validator(geometry_cache, mirror_mode=True)
    # Anchor flush left at x=-1, candidate one step left at x=-2
    anchor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": -1, "y": 0}
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": -2, "y": 0}
    result = validator.validate_candidate(candidate, anchor, set())
    assert result.accepted, f"Expected accepted but got rejection: {result.rejection}"
    assert result.mirror_companion is not None
    assert result.primary_cells is not None
    assert result.companion_cells is not None
    assert result.companion_cells != result.primary_cells


def test_validate_candidate_mirror_mode_self_mirroring_companion_is_none(geometry_cache):
    """Mirror mode: a 2-wide part centered on the axis should have companion=None."""
    validator = _make_validator(geometry_cache, mirror_mode=True)
    # cosmoteer.armor_2x1 at x=-1 straddles the axis: cells {(-1,0),(0,0)}
    anchor = {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": -1, "y": 0}
    candidate = {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": -1, "y": 1}
    result = validator.validate_candidate(candidate, anchor, set())
    assert result.accepted, f"Expected accepted but got rejection: {result.rejection}"
    # Self-mirroring: both cells straddle the axis so no separate companion is needed
    assert result.mirror_companion is None
    assert result.companion_cells is None


# ---------------------------------------------------------------------------
# validate_seed_part
# ---------------------------------------------------------------------------


def test_validate_seed_part_skips_connectivity_check(geometry_cache):
    """validate_seed_part should not reject for missing structural contact."""
    validator = _make_validator(geometry_cache)
    # Isolated part with no anchor — validate_seed_part must not check connectivity
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 7, "y": 7}
    result = validator.validate_seed_part(candidate, set())
    assert result.accepted
    assert result.rejection is None


def test_validate_seed_part_rejects_geometry_unknown(geometry_cache):
    """validate_seed_part should reject unknown part_ids."""
    validator = _make_validator(geometry_cache)
    candidate = {"part_id": "cosmoteer.nonexistent", "rotation": 0, "x": 0, "y": 0}
    result = validator.validate_seed_part(candidate, set())
    assert not result.accepted
    assert result.rejection == "geometry_unknown"


def test_validate_seed_part_rejects_overlap(geometry_cache):
    """validate_seed_part should reject parts that overlap already-occupied cells."""
    validator = _make_validator(geometry_cache)
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 0}
    occupied = {(0, 0)}
    result = validator.validate_seed_part(candidate, occupied)
    assert not result.accepted
    assert result.rejection == "overlap"


def test_validate_seed_part_returns_primary_cells(geometry_cache):
    """validate_seed_part should return the pre-computed primary_cells on success."""
    validator = _make_validator(geometry_cache)
    candidate = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 3, "y": 3}
    result = validator.validate_seed_part(candidate, set())
    assert result.accepted
    assert result.primary_cells == frozenset({(3, 3)})


# ---------------------------------------------------------------------------
# requirements_satisfied
# ---------------------------------------------------------------------------


def test_requirements_satisfied_returns_true_when_none(geometry_cache):
    """requirements_satisfied should return True when no requirements are configured."""
    validator = _make_validator(geometry_cache, requirements=None)
    assert validator.requirements_satisfied({}) is True
    assert validator.requirements_satisfied({"cosmoteer.armor": 999}) is True


def test_requirements_satisfied_returns_false_when_short(geometry_cache):
    """requirements_satisfied should return False when a requirement is unmet."""
    validator = _make_validator(geometry_cache, requirements={"cosmoteer.armor": 3})
    assert validator.requirements_satisfied({"cosmoteer.armor": 2}) is False


def test_requirements_satisfied_returns_true_when_met(geometry_cache):
    """requirements_satisfied should return True when all requirements are met."""
    validator = _make_validator(geometry_cache, requirements={"cosmoteer.armor": 3})
    assert validator.requirements_satisfied({"cosmoteer.armor": 3}) is True
    assert validator.requirements_satisfied({"cosmoteer.armor": 5}) is True


# ---------------------------------------------------------------------------
# seed_state_mirror_valid
# ---------------------------------------------------------------------------


def test_seed_state_mirror_valid_non_mirror_always_true(geometry_cache):
    """seed_state_mirror_valid should always return True when mirror_mode is off."""
    validator = _make_validator(geometry_cache, mirror_mode=False)
    assert validator.seed_state_mirror_valid({(0, 0), (1, 0)}) is True


def test_seed_state_mirror_valid_rejects_asymmetric(geometry_cache):
    """seed_state_mirror_valid should reject unbalanced cells in mirror mode."""
    validator = _make_validator(geometry_cache, mirror_mode=True)
    asymmetric = {(0, 0), (1, 0)}  # only right-side cells
    assert validator.seed_state_mirror_valid(asymmetric) is False


def test_seed_state_mirror_valid_accepts_balanced(geometry_cache):
    """seed_state_mirror_valid should accept mirror-balanced cells in mirror mode."""
    validator = _make_validator(geometry_cache, mirror_mode=True)
    balanced = {(-1, 0), (0, 0)}  # straddles axis at x=-0.5
    assert validator.seed_state_mirror_valid(balanced) is True
