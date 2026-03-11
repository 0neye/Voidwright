"""Shared Markov backend compatibility surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from common.geometry import load_vanilla_part_geometry

from .corpus import iter_vanilla_parts_from_graph, iter_vanilla_parts_from_ship
from .generation import WeightedSampler, generate_ship_layout
from .order import choose_root, order_ship_parts, parts_touch
from .state import history_symbol, state_key
from .training import build_payload_from_corpus, build_payload_from_graph_corpus
from .types import (
    END_TOKEN,
    ROOT_ANCHOR,
    GenerationConfig,
    RelativePlacementToken,
    ShipPart,
    TrainingConfig,
    TrainingStats,
)
from .validation import validate_relative_placement_assumptions

__all__ = [
    "END_TOKEN",
    "ROOT_ANCHOR",
    "GenerationConfig",
    "RelativeMarkovModel",
    "RelativePlacementToken",
    "ShipPart",
    "TrainingConfig",
    "TrainingStats",
    "WeightedSampler",
    "build_model_from_corpus",
    "build_model_from_graph_corpus",
    "choose_root",
    "history_symbol",
    "iter_vanilla_parts_from_graph",
    "iter_vanilla_parts_from_ship",
    "order_ship_parts",
    "parts_touch",
    "state_key",
    "validate_relative_placement_assumptions",
]


class RelativeMarkovModel:
    """Load, save, and sample the relative-placement Markov model."""

    def __init__(self, payload: dict):
        """Initialize an in-memory model from a serialized payload"""

        self.payload = payload
        self.schema_version = int(payload.get("schema_version", 1))
        self.order = int(payload["config"]["markov_order"])
        self.start_counts: Dict[str, int] = payload["start_counts"]
        self.transition_counts: Dict[str, Dict[str, int]] = payload["transition_counts"]
        self.part_frequency: Dict[str, int] = payload["part_frequency"]
        self.geometry_cache = load_vanilla_part_geometry()

    def _state_key(self, history: Sequence[str], order: int) -> str:
        """Return the serialized state key for a token history tail"""

        if self.schema_version >= 2:
            return state_key(history, order)
        tail = list(history[-order:]) if order > 0 else []
        return " || ".join(tail)

    @classmethod
    def load(cls, path: str | Path) -> "RelativeMarkovModel":
        """Load a serialized Markov model from disk"""

        with Path(path).open(encoding="utf-8") as file_handle:
            return cls(json.load(file_handle))

    def save(self, path: str | Path) -> None:
        """Write this Markov model payload to disk"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(self.payload, file_handle, separators=(",", ":"), sort_keys=True)
            file_handle.write("\n")

    def _placement_within_bounds(self, part: ShipPart, config: GenerationConfig) -> bool:
        """Return True when all footprint cells fit within configured bounds"""

        for x, y in part.footprint_cells(self.geometry_cache):
            if not (
                config.bounds_min_x <= x <= config.bounds_max_x
                and config.bounds_min_y <= y <= config.bounds_max_y
            ):
                return False
        return True

    def _within_primary_bounds(self, part: ShipPart, config: GenerationConfig) -> bool:
        """Return True when a primary part stays on the left side in mirror mode"""

        for x, y in part.footprint_cells(self.geometry_cache):
            if not (config.bounds_min_x <= x <= -1 and config.bounds_min_y <= y <= config.bounds_max_y):
                return False
        return True

    def _within_mirror_bounds(self, part: ShipPart, config: GenerationConfig) -> bool:
        """Return True when a mirrored part stays on the right side in mirror mode"""

        for x, y in part.footprint_cells(self.geometry_cache):
            if not (0 <= x <= config.bounds_max_x and config.bounds_min_y <= y <= config.bounds_max_y):
                return False
        return True

    def generate(self, config: GenerationConfig, *, seed_parts=None) -> dict:
        """Generate a ship layout from this model

        Args:
            config: Generation configuration
            seed_parts: Optional list of seed parts to pre-place before generation

        Returns:
            Generation payload with emitted parts, trace, and stats
        """

        return generate_ship_layout(self, config, seed_parts=seed_parts)


def build_model_from_corpus(input_dir: Path, config: TrainingConfig) -> RelativeMarkovModel:
    """Build a Markov model from canonical extracted ship JSON files"""

    payload = build_payload_from_corpus(input_dir, config)
    return RelativeMarkovModel(payload)


def build_model_from_graph_corpus(graph_dir: Path, config: TrainingConfig) -> RelativeMarkovModel:
    """Build a Markov model from pre-generated structural graph JSON artifacts"""

    payload = build_payload_from_graph_corpus(graph_dir, config)
    return RelativeMarkovModel(payload)
