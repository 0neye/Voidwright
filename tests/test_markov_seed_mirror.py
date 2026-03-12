"""Mirror-mode seeded generation regressions for footprint symmetry handling."""

from __future__ import annotations

import pytest

from markov.generation import WeightedSampler
from markov.model import GenerationConfig, RelativeMarkovModel, RelativePlacementToken
from markov.state import history_symbol


def _build_seed_payload() -> dict:
    """Build a tiny payload suitable for seed-only mirror generation tests."""

    root_token = RelativePlacementToken(
        part_id="cosmoteer.armor_2x1",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    return {
        "schema_version": 2,
        "config": {"markov_order": 1},
        "start_counts": {root_token: 1},
        "transition_counts": {},
        "part_frequency": {"cosmoteer.armor_2x1": 1, "cosmoteer.armor": 2},
    }


def _build_seed_growth_payload() -> tuple[dict, str, str]:
    """Build a tiny mirror payload where one seeded transition can grow the ship.

    Returns:
        Tuple of `(payload, root_token, growth_token)` keys so tests can drive
        deterministic weighted sampling
    """

    root_token = RelativePlacementToken(
        part_id="cosmoteer.armor_2x1",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    growth_token = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_2x1",
        anchor_rotation=0,
        dx=-1,
        dy=0,
    ).as_key()
    payload = {
        "schema_version": 2,
        "config": {"markov_order": 1},
        "start_counts": {root_token: 1},
        "transition_counts": {
            history_symbol(root_token): {growth_token: 1},
        },
        "part_frequency": {"cosmoteer.armor_2x1": 1, "cosmoteer.armor": 1},
    }
    return payload, root_token, growth_token


def test_seed_mirror_accepts_asymmetric_part_layout_with_symmetric_footprint() -> None:
    """Mirror mode should accept seeds with asymmetric parts when occupied cells match."""

    model = RelativeMarkovModel(_build_seed_payload())
    config = GenerationConfig(max_parts=3, mirror_symmetry=True, rng_seed=7)
    seed_parts = [
        # Left side uses one 2x1 part...
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": -3, "y": 0},
        # ...while right side uses two 1x1 parts with the same reflected footprint
        {"part_id": "cosmoteer.armor", "rotation": 0, "x": 1, "y": 0},
        {"part_id": "cosmoteer.armor", "rotation": 0, "x": 2, "y": 0},
    ]

    payload = model.generate(config, seed_parts=seed_parts)
    assert payload["stats"]["seed"]["seed_parts_placed"] == 3
    assert payload["stats"]["parts_generated"] == 3
    assert payload["stats"]["stop_reason"] == "max_parts"


def test_seed_mirror_accepts_centerline_straddling_anchor_without_left_only_parts() -> None:
    """Mirror mode should allow a centered self-mirroring seed to anchor sampling."""

    model = RelativeMarkovModel(_build_seed_payload())
    config = GenerationConfig(max_parts=1, mirror_symmetry=True, rng_seed=9)
    seed_parts = [
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": -1, "y": 0},
    ]

    payload = model.generate(config, seed_parts=seed_parts)
    assert payload["stats"]["seed"]["seed_parts_placed"] == 1
    assert payload["stats"]["parts_generated"] == 1


def test_seed_mirror_rejects_asymmetric_occupied_footprint() -> None:
    """Mirror mode should reject seeded footprints that do not reflect exactly."""

    model = RelativeMarkovModel(_build_seed_payload())
    config = GenerationConfig(max_parts=2, mirror_symmetry=True, rng_seed=11)
    seed_parts = [
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": -3, "y": 0},
        {"part_id": "cosmoteer.armor", "rotation": 0, "x": 2, "y": 0},
    ]

    with pytest.raises(RuntimeError, match="occupied footprint is not mirror-symmetric"):
        model.generate(config, seed_parts=seed_parts)


def test_seed_mirror_grows_beyond_seed_when_transition_is_valid(monkeypatch) -> None:
    """Mirror seeded generation should add parts when a legal transition exists."""

    payload, root_token, growth_token = _build_seed_growth_payload()
    model = RelativeMarkovModel(payload)
    config = GenerationConfig(max_parts=3, mirror_symmetry=True, rng_seed=21, max_resample_per_step=2)
    seed_parts = [
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": -1, "y": 0},
    ]

    def _sample_known_sequence(counter: dict, _rng) -> str:
        """Return deterministic root and growth tokens for this seeded run."""

        if root_token in counter:
            return root_token
        if growth_token in counter:
            return growth_token
        return next(iter(counter))

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_sample_known_sequence))
    generated_payload = model.generate(config, seed_parts=seed_parts)

    assert generated_payload["stats"]["seed"]["seed_parts_placed"] == 1
    assert generated_payload["stats"]["parts_generated"] == 3
    assert generated_payload["stats"]["parts_generated"] > generated_payload["stats"]["seed"]["seed_parts_placed"]
    assert generated_payload["stats"]["stop_reason"] == "max_parts"
