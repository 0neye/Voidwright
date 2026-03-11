"""Shared Markov domain types and constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

from ship_layout.geometry import footprint_cells as shared_footprint_cells
from ship_layout.types import PlacedPart

__all__ = [
    "Coord",
    "END_TOKEN",
    "ROOT_ANCHOR",
    "RelativePlacementToken",
    "ShipPart",
    "TrainingConfig",
    "GenerationConfig",
    "TrainingStats",
    "_config_as_dict",
]

Coord = Tuple[int, int]
END_TOKEN = "__END__"
ROOT_ANCHOR = "__ROOT__"


def _config_as_dict(config_obj) -> dict:
    """Convert a dataclass config into a JSON-safe dictionary

    Args:
        config_obj: Dataclass config instance to serialize

    Returns:
        Dictionary with set-like fields converted to sorted lists
    """

    config_dict = asdict(config_obj)
    for field_name, field_value in config_dict.items():
        if isinstance(field_value, (frozenset, set)):
            # Keep set-like values deterministic for model artifact diffs
            config_dict[field_name] = sorted(field_value)
    return config_dict


@dataclass(frozen=True)
class RelativePlacementToken:
    part_id: str
    rotation: int
    anchor_part_id: str
    anchor_rotation: int
    dx: int
    dy: int

    def as_key(self) -> str:
        """Return the compact serialized key for this placement token"""

        return "|".join(
            [
                self.part_id,
                str(self.rotation),
                self.anchor_part_id,
                str(self.anchor_rotation),
                str(self.dx),
                str(self.dy),
            ]
        )

    def to_dict(self) -> dict:
        """Return this token as a plain JSON-serializable dictionary"""

        return asdict(self)

    @classmethod
    def from_key(cls, key: str) -> "RelativePlacementToken":
        """Decode a serialized placement token key back into a dataclass"""

        part_id, rotation, anchor_part_id, anchor_rotation, dx, dy = key.split("|")
        return cls(
            part_id=part_id,
            rotation=int(rotation),
            anchor_part_id=anchor_part_id,
            anchor_rotation=int(anchor_rotation),
            dx=int(dx),
            dy=int(dy),
        )


@dataclass(frozen=True)
class ShipPart:
    part_id: str
    rotation: int
    x: int
    y: int
    flip_x: bool = False
    flip_y: bool = False

    def footprint_cells(self, geometry_cache: Dict[str, object]) -> frozenset[Coord]:
        """Return the occupied world cells for this placed part"""

        return shared_footprint_cells(
            PlacedPart(
                part_id=self.part_id,
                rotation=self.rotation,
                x=self.x,
                y=self.y,
                flip_x=self.flip_x,
                flip_y=self.flip_y,
            ),
            geometry_cache,
        )

    def bbox(self, geometry_cache: Dict[str, object]) -> Tuple[int, int, int, int]:
        """Return the inclusive bounding box for this part"""

        cells = self.footprint_cells(geometry_cache)
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class TrainingConfig:
    markov_order: int = 2
    min_parts_per_ship: int = 2
    max_parts_per_ship: int = 5000
    anchor_window: int = 128
    part_allowlist: Optional[frozenset] = None


@dataclass
class GenerationConfig:
    max_parts: int = 250
    max_attempts: int = 3000
    max_resample_per_step: int = 32
    bounds_min_x: int = -64
    bounds_max_x: int = 64
    bounds_min_y: int = -64
    bounds_max_y: int = 64
    rng_seed: Optional[int] = None
    part_allowlist: Optional[frozenset] = None
    mirror_symmetry: bool = False
    part_requirements: Optional[dict] = None


@dataclass
class TrainingStats:
    ships_seen: int = 0
    ships_used: int = 0
    ships_skipped_too_small: int = 0
    ships_skipped_too_large: int = 0
    vanilla_parts_seen: int = 0
    vanilla_parts_used: int = 0
    non_vanilla_parts_excluded: int = 0
    unknown_vanilla_geometry_excluded: int = 0
    root_tokens: int = 0
    transition_tokens: int = 0
    end_tokens: int = 0
    touching_transitions: int = 0
    non_touching_transitions: int = 0
