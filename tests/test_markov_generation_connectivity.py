"""Regression tests for structural connectivity checks during Markov generation."""

from __future__ import annotations

from markov.generation import WeightedSampler
from markov.model import END_TOKEN, GenerationConfig, RelativeMarkovModel, RelativePlacementToken
from markov.state import history_symbol


def _build_minimal_model_payload() -> dict:
    """Build a tiny model payload with one root and one transition state.

    Returns:
        Minimal schema-v2 payload suitable for deterministic generation tests
    """

    root_token = RelativePlacementToken(
        part_id="cosmoteer.armor_wedge",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    invalid_air_contact = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_wedge",
        anchor_rotation=0,
        dx=0,
        dy=-1,
    ).as_key()
    valid_structural_contact = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_wedge",
        anchor_rotation=0,
        dx=1,
        dy=0,
    ).as_key()

    root_state = history_symbol(root_token)
    second_state = history_symbol(valid_structural_contact)

    return {
        "schema_version": 2,
        "config": {"markov_order": 1},
        "start_counts": {root_token: 1},
        "transition_counts": {
            root_state: {
                invalid_air_contact: 1,
                valid_structural_contact: 1,
            },
            second_state: {END_TOKEN: 1},
        },
        "part_frequency": {
            "cosmoteer.armor_wedge": 1,
            "cosmoteer.armor": 2,
        },
    }


def _build_root_only_payload(*, part_id: str, rotation: int = 0) -> dict:
    """Build a minimal payload that only emits one root token.

    Args:
        part_id: Root part ID used for the only start token
        rotation: Saved rotation for the root token

    Returns:
        Payload with no transitions so generation stops after root placement
    """

    root_token = RelativePlacementToken(
        part_id=part_id,
        rotation=rotation,
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
        "part_frequency": {part_id: 1},
    }


def _build_seeded_virtual_root_rotation_payload() -> tuple[dict, str, str, str]:
    """Build a payload where only one root rotation matches seeded anchors.

    Returns:
        Tuple of `(payload, compatible_root, incompatible_root, valid_transition)`
        keys for deterministic monkeypatch sampling in seeded-generation tests
    """

    compatible_root = RelativePlacementToken(
        part_id="cosmoteer.armor_2x1",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    incompatible_root = RelativePlacementToken(
        part_id="cosmoteer.armor_2x1",
        rotation=1,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    valid_transition = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_2x1",
        anchor_rotation=0,
        dx=2,
        dy=0,
    ).as_key()
    incompatible_anchor_transition = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_2x1",
        anchor_rotation=1,
        dx=2,
        dy=0,
    ).as_key()

    payload = {
        "schema_version": 2,
        "config": {"markov_order": 1},
        "start_counts": {
            compatible_root: 1,
            incompatible_root: 1,
        },
        "transition_counts": {
            history_symbol(compatible_root): {valid_transition: 1},
            history_symbol(incompatible_root): {incompatible_anchor_transition: 1},
            history_symbol(valid_transition): {END_TOKEN: 1},
        },
        "part_frequency": {
            "cosmoteer.armor_2x1": 2,
            "cosmoteer.armor": 2,
        },
    }
    return payload, compatible_root, incompatible_root, valid_transition


def _build_seeded_overlap_retry_payload() -> tuple[dict, str, str, str]:
    """Build a payload where one candidate overlaps and another is valid.

    Returns:
        Tuple of `(payload, root_token, overlap_token, valid_token)` keys
    """

    root_token = RelativePlacementToken(
        part_id="cosmoteer.armor_2x1",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    overlap_token = RelativePlacementToken(
        part_id="cosmoteer.armor_2x1",
        rotation=0,
        anchor_part_id="cosmoteer.armor_2x1",
        anchor_rotation=0,
        dx=1,
        dy=0,
    ).as_key()
    valid_token = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_2x1",
        anchor_rotation=0,
        dx=2,
        dy=0,
    ).as_key()

    payload = {
        "schema_version": 2,
        "config": {"markov_order": 1},
        "start_counts": {root_token: 1},
        "transition_counts": {
            history_symbol(root_token): {overlap_token: 1, valid_token: 1},
            history_symbol(valid_token): {END_TOKEN: 1},
        },
        "part_frequency": {"cosmoteer.armor_2x1": 2, "cosmoteer.armor": 1},
    }
    return payload, root_token, overlap_token, valid_token


def _build_seeded_prefers_viable_virtual_root_payload() -> tuple[dict, str, str, str]:
    """Build a payload where only the non-matching root can grow from the seed.

    Returns:
        Tuple of `(payload, compatible_but_dead_root, viable_only_root, valid_transition)` keys
    """

    compatible_but_dead_root = RelativePlacementToken(
        part_id="cosmoteer.armor_2x1",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    viable_only_root = RelativePlacementToken(
        part_id="cosmoteer.corridor",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    valid_transition = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_2x1",
        anchor_rotation=0,
        dx=2,
        dy=0,
    ).as_key()

    payload = {
        "schema_version": 2,
        "config": {"markov_order": 1},
        "start_counts": {
            compatible_but_dead_root: 1,
            viable_only_root: 1,
        },
        "transition_counts": {
            history_symbol(compatible_but_dead_root): {},
            history_symbol(viable_only_root): {valid_transition: 1},
            history_symbol(valid_transition): {END_TOKEN: 1},
        },
        "part_frequency": {
            "cosmoteer.armor_2x1": 1,
            "cosmoteer.corridor": 1,
            "cosmoteer.armor": 1,
        },
    }
    return payload, compatible_but_dead_root, viable_only_root, valid_transition


def test_generation_resamples_when_candidate_fails_structural_connectivity(monkeypatch) -> None:
    """Generation should reject air-contact candidates and resample the step."""

    model = RelativeMarkovModel(_build_minimal_model_payload())
    config = GenerationConfig(
        max_parts=10,
        max_attempts=10,
        max_resample_per_step=4,
        rng_seed=123,
    )

    root_token = next(iter(model.start_counts))
    invalid_air_contact = next(
        token_key
        for token_key in model.transition_counts[history_symbol(root_token)]
        if token_key != END_TOKEN and RelativePlacementToken.from_key(token_key).dx == 0
    )
    valid_structural_contact = next(
        token_key
        for token_key in model.transition_counts[history_symbol(root_token)]
        if token_key != END_TOKEN and RelativePlacementToken.from_key(token_key).dx == 1
    )

    sampled_tokens = iter([root_token, invalid_air_contact, valid_structural_contact, END_TOKEN])

    def _sample_in_order(counter: dict, _rng) -> str:
        """Return pre-seeded token samples for deterministic rejection flow."""

        sampled_token = next(sampled_tokens)
        assert sampled_token in counter
        return sampled_token

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_sample_in_order))
    payload = model.generate(config)

    assert payload["stats"]["stop_reason"] == "end_token"
    assert payload["stats"]["parts_generated"] == 2
    assert payload["stats"]["rejections"]["structural"] == 1
    assert payload["parts"] == [
        {"part_id": "cosmoteer.armor_wedge", "rotation": 0, "x": 0, "y": 0, "flip_x": False, "flip_y": False},
        {"part_id": "cosmoteer.armor", "rotation": 0, "x": 1, "y": 0, "flip_x": False, "flip_y": False},
    ]


def test_mirror_generation_keeps_wedge_handedness_swap(monkeypatch) -> None:
    """Mirror mode should still rotate mirrored wedges to the expected saved orientation."""

    model = RelativeMarkovModel(_build_minimal_model_payload())
    config = GenerationConfig(
        max_parts=2,
        max_attempts=4,
        max_resample_per_step=2,
        rng_seed=123,
        mirror_symmetry=True,
    )

    root_token = next(iter(model.start_counts))
    sampled_tokens = iter([root_token])

    def _sample_in_order(counter: dict, _rng) -> str:
        sampled_token = next(sampled_tokens)
        assert sampled_token in counter
        return sampled_token

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_sample_in_order))
    payload = model.generate(config)

    assert payload["parts"] == [
        {"part_id": "cosmoteer.armor_wedge", "rotation": 0, "x": -1, "y": 0, "flip_x": False, "flip_y": False},
        {"part_id": "cosmoteer.armor_wedge", "rotation": 1, "x": 0, "y": 0, "flip_x": True, "flip_y": False},
    ]


def test_mirror_generation_keeps_self_mirroring_centerline_root_once(monkeypatch) -> None:
    """Mirror mode should keep one part when reflected cells are identical."""

    model = RelativeMarkovModel(_build_root_only_payload(part_id="cosmoteer.armor_2x1", rotation=0))
    config = GenerationConfig(
        max_parts=4,
        max_attempts=4,
        max_resample_per_step=2,
        rng_seed=123,
        mirror_symmetry=True,
    )

    root_token = next(iter(model.start_counts))
    sampled_tokens = iter([root_token])

    def _sample_in_order(counter: dict, _rng) -> str:
        """Return pre-seeded token samples for deterministic root placement."""

        sampled_token = next(sampled_tokens)
        assert sampled_token in counter
        return sampled_token

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_sample_in_order))
    payload = model.generate(config)

    assert payload["parts"] == [
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": -1, "y": 0, "flip_x": False, "flip_y": False},
    ]
    assert payload["stats"]["mirror"]["primary_parts"] == 1
    assert payload["stats"]["mirror"]["mirror_parts"] == 0


def test_seeded_generation_uses_rotation_compatible_virtual_root(monkeypatch) -> None:
    """Seeded generation should avoid virtual roots with incompatible anchor rotation."""

    payload, compatible_root, incompatible_root, valid_transition = (
        _build_seeded_virtual_root_rotation_payload()
    )
    model = RelativeMarkovModel(payload)
    config = GenerationConfig(
        max_parts=2,
        max_attempts=8,
        max_resample_per_step=2,
        rng_seed=3,
    )
    seed_parts = [
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": 0, "y": 0},
    ]

    def _prefer_incompatible_root(counter: dict, _rng) -> str:
        """Prefer bad root only when selection does not pre-filter it out."""

        if incompatible_root in counter:
            return incompatible_root
        if compatible_root in counter:
            return compatible_root
        if valid_transition in counter:
            return valid_transition
        return next(iter(counter))

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_prefer_incompatible_root))
    generated_payload = model.generate(config, seed_parts=seed_parts)

    assert generated_payload["stats"]["parts_generated"] == 2
    assert generated_payload["stats"]["rejections"]["missing_anchor"] == 0
    assert generated_payload["stats"]["seed"]["virtual_root"]["rotation"] == 0


def test_seeded_generation_avoids_retrying_same_overlap_token(monkeypatch) -> None:
    """Seeded retries should move past an overlap token within the same step."""

    payload, root_token, overlap_token, valid_token = _build_seeded_overlap_retry_payload()
    model = RelativeMarkovModel(payload)
    config = GenerationConfig(
        max_parts=2,
        max_attempts=8,
        max_resample_per_step=2,
        rng_seed=5,
    )
    seed_parts = [
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": 0, "y": 0},
    ]

    def _prefer_overlap(counter: dict, _rng) -> str:
        """Always choose overlap first unless it has been excluded already."""

        if overlap_token in counter:
            return overlap_token
        if valid_token in counter:
            return valid_token
        if root_token in counter:
            return root_token
        return next(iter(counter))

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_prefer_overlap))
    monkeypatch.setattr("markov.generation._is_structurally_connected_to_anchor", lambda *_args: True)

    generated_payload = model.generate(config, seed_parts=seed_parts)

    assert generated_payload["stats"]["parts_generated"] == 2
    assert generated_payload["stats"]["rejections"]["overlap"] >= 1
    assert generated_payload["stats"]["stop_reason"] == "max_parts"


def test_seeded_generation_prefers_viable_virtual_root_over_signature_only_match(monkeypatch) -> None:
    """Seeded startup should prefer roots that can actually grow from the seed."""

    payload, compatible_but_dead_root, viable_only_root, valid_transition = (
        _build_seeded_prefers_viable_virtual_root_payload()
    )
    model = RelativeMarkovModel(payload)
    config = GenerationConfig(
        max_parts=2,
        max_attempts=8,
        max_resample_per_step=2,
        rng_seed=17,
    )
    seed_parts = [
        {"part_id": "cosmoteer.armor_2x1", "rotation": 0, "x": 0, "y": 0},
    ]

    def _prefer_dead_root_when_available(counter: dict, _rng) -> str:
        """Choose the dead root only if selection fails to pre-filter it out."""

        if compatible_but_dead_root in counter:
            return compatible_but_dead_root
        if viable_only_root in counter:
            return viable_only_root
        if valid_transition in counter:
            return valid_transition
        return next(iter(counter))

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_prefer_dead_root_when_available))
    generated_payload = model.generate(config, seed_parts=seed_parts)

    assert generated_payload["stats"]["parts_generated"] == 2
    assert generated_payload["stats"]["stop_reason"] == "max_parts"
    assert generated_payload["stats"]["seed"]["virtual_root"]["selection"] == "viable_transition_only"
    assert generated_payload["stats"]["seed"]["virtual_root"]["part_id"] == "cosmoteer.corridor"
