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
