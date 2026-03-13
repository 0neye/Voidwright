from __future__ import annotations

import orjson
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from common.geometry import (
    Coord,
    VANILLA_NAMESPACE,
    infer_meta,
    is_vanilla_part_id,
    iter_ship_files,
    normalize_part_id,
)
from .layout_helpers import door_adjacent_cells as shared_door_adjacent_cells

__all__ = [
    "SIDE_BY_ORIENTATION",
    "ROTATION_NORMALIZATION_OFFSETS",
    "FALLBACK_ALLOWED_LOCATION_ANCHOR_DELTAS",
    "REJECT_CLASS_HINTS",
    "SideSignature",
    "PlacementObservation",
    "CandidatePart",
    "Thresholds",
    "ValidationResult",
    "DoorValidationSummary",
    "DoorPlacementRules",
    "build_cell_map",
    "choose_boundary_part",
    "resolve_observation_from_cells",
    "door_adjacent_cells",
    "match_allowed_door_cell_with_anchor_delta",
    "resolve_vanilla_allowed_location_fallback",
    "resolve_door_observation",
    "iter_potential_boundaries",
    "default_overrides",
    "classify_override_reject",
    "crew_rule_for_part",
    "signature_matches_allowed",
    "match_crew_override",
    "infer_rules_from_corpus",
    "validate_corpus_against_rules",
]

# Cosmoteer ship Door.Cell is not the low-coordinate occupied cell of the pair.
# Instead it names the right/bottom cell of the two-cell doorway span:
# - orientation 0 joins (x, y-1) <-> (x, y)
# - orientation 1 joins (x-1, y) <-> (x, y)
# Vanilla game-data allowed_door_locations use the same doorway frame, but the
# exported rotated footprint is normalized back to a top-left local bbox. The
# normalization shift is rotation-dependent:
#   rot0=(0,0), rot1=(1,0), rot2=(1,1), rot3=(0,1)
# so local doorway coordinates must add that offset before matching the exported
# allowed_door_locations.
SIDE_BY_ORIENTATION = {0: ("S", "N"), 1: ("E", "W")}
ROTATION_NORMALIZATION_OFFSETS = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
# Residual resolver_none cases are dominated by doors where one vanilla part
# appears to span both doorway cells while a second adjacent vanilla part shares
# only one side cell. In practice these are recoverable by matching the ship door
# back to game-data allowed_door_locations with a tiny anchor drift allowance.
FALLBACK_ALLOWED_LOCATION_ANCHOR_DELTAS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (0, -1),
    (-1, 0),
    (-1, -1),
    (1, -1),
    (-1, 1),
)
REJECT_CLASS_HINTS = ("armor", "structure", "wedge")
@dataclass(frozen=True)
class SideSignature:
    part_id: str
    rotation: int
    side: str
    offset: int
    width: int
    height: int
    traversable: bool

    def key(self) -> str:
        return "|".join([
            self.part_id,
            str(self.rotation),
            self.side,
            str(self.offset),
            str(self.width),
            str(self.height),
            "1" if self.traversable else "0",
        ])

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "rotation": self.rotation,
            "side": self.side,
            "offset": self.offset,
            "rotated_width": self.width,
            "rotated_height": self.height,
            "traversable": self.traversable,
        }


@dataclass(frozen=True)
class PlacementObservation:
    orientation: int
    door_cell: Coord
    a: SideSignature
    b: SideSignature
    a_index: int
    b_index: int
    a_part: "CandidatePart"
    b_part: "CandidatePart"

    def pair_key(self) -> str:
        left = self.a.key()
        right = self.b.key()
        if right < left:
            left, right = right, left
        return f"{self.orientation}||{left}||{right}"


@dataclass
class CandidatePart:
    index: int
    part_id: str
    rotation: int
    x0: int
    y0: int
    width: int
    height: int
    traversable: bool
    footprint_tiles: frozenset[Coord]
    unblocked_tiles: frozenset[Coord]
    blocked_travel_cells: frozenset[Coord]
    allowed_door_locations: Tuple[Coord, ...]

    def contains_world_cell(self, cell: Coord) -> bool:
        return (cell[0] - self.x0, cell[1] - self.y0) in self.footprint_tiles

    def local_cell(self, cell: Coord) -> Coord:
        return (cell[0] - self.x0, cell[1] - self.y0)

    def cell_is_walkable(self, cell: Coord) -> bool:
        local = self.local_cell(cell)
        return local in self.unblocked_tiles and local not in self.blocked_travel_cells

    def boundary_offset(self, cell: Coord, side: str) -> Optional[int]:
        local_x, local_y = self.local_cell(cell)
        if (local_x, local_y) not in self.footprint_tiles:
            return None
        if side == "E":
            return local_y if (local_x + 1, local_y) not in self.footprint_tiles else None
        if side == "W":
            return local_y if (local_x - 1, local_y) not in self.footprint_tiles else None
        if side == "N":
            return local_x if (local_x, local_y - 1) not in self.footprint_tiles else None
        if side == "S":
            return local_x if (local_x, local_y + 1) not in self.footprint_tiles else None
        return None

    def signature_for_boundary(self, cell: Coord, side: str) -> Optional[SideSignature]:
        offset = self.boundary_offset(cell, side)
        if offset is None:
            return None
        return SideSignature(
            part_id=self.part_id,
            rotation=self.rotation,
            side=side,
            offset=offset,
            width=self.width,
            height=self.height,
            traversable=self.traversable,
        )

    def matches_allowed_door_cell(self, door_cell: Coord) -> bool:
        if not self.allowed_door_locations:
            return False
        ox, oy = ROTATION_NORMALIZATION_OFFSETS[self.rotation % 4]
        local = (door_cell[0] - self.x0 + ox, door_cell[1] - self.y0 + oy)
        return local in self.allowed_door_locations


@dataclass
class Thresholds:
    min_side_observations: int = 2
    min_side_ratio: float = 0.02
    min_pair_observations: int = 2
    min_pair_ratio: float = 0.02


@dataclass
class ValidationResult:
    allowed: bool
    confidence: str
    reason: str
    observation: Optional[PlacementObservation]
    pair_stats: Optional[dict]
    side_stats: List[Optional[dict]]
    decision: str = "reject"
    source: str = "inferred"
    details: dict = field(default_factory=dict)


@dataclass
class DoorValidationSummary:
    per_door: List[ValidationResult]
    counts: Dict[str, int]


class DoorPlacementRules:
    """Reusable validator for single doors and whole-ship door sets.

    Decision tiers:
    - allow: safe positive match
    - reject: safe negative match
    - unresolved: insufficient evidence / intentionally deferred

    Single-door validation prefers curated overrides first, then falls back to
    the inferred side/pair corpus rules. Whole-ship validation additionally
    enforces override-level per-part door-count caps.
    """

    def __init__(self, payload: dict):
        self.payload = payload
        t = payload.get("thresholds", {})
        self.thresholds = Thresholds(
            min_side_observations=int(t.get("min_side_observations", 2)),
            min_side_ratio=float(t.get("min_side_ratio", 0.02)),
            min_pair_observations=int(t.get("min_pair_observations", 2)),
            min_pair_ratio=float(t.get("min_pair_ratio", 0.02)),
        )
        self.side_rules = payload.get("rules", {}).get("side_rules", {})
        self.pair_rules = payload.get("rules", {}).get("pair_rules", {})
        self.overrides = payload.get("overrides", default_overrides())
        self.vanilla_prefix = payload.get("vanilla_filter", {}).get("namespace_prefix", VANILLA_NAMESPACE)

    @classmethod
    def load(cls, path: str | Path) -> "DoorPlacementRules":
        with Path(path).open("rb") as fh:
            return cls(orjson.loads(fh.read()))

    def _validate_with_overrides(self, observation: Optional[PlacementObservation]) -> Optional[ValidationResult]:
        if observation is None:
            return ValidationResult(
                False,
                "none",
                "door does not resolve to two distinct adjacent part boundaries",
                None,
                None,
                [],
                decision="unresolved",
                source="resolver",
            )

        for sig in (observation.a, observation.b):
            if not is_vanilla_part_id(sig.part_id):
                return ValidationResult(
                    False,
                    "none",
                    f"non-vanilla part excluded for this phase: {sig.part_id}",
                    observation,
                    None,
                    [],
                    decision="unresolved",
                    source="vanilla_filter",
                    details={"signature": sig.to_dict()},
                )
            reject_reason = classify_override_reject(sig)
            if reject_reason:
                return ValidationResult(
                    False,
                    "high",
                    reject_reason,
                    observation,
                    None,
                    [],
                    decision="reject",
                    source="override",
                    details={"signature": sig.to_dict()},
                )

        crew_match = match_crew_override(observation)
        if crew_match:
            return crew_match

        geometry_checks = []
        for sig, part in ((observation.a, observation.a_part), (observation.b, observation.b_part)):
            if crew_rule_for_part(sig.part_id) is not None:
                continue
            if not part.allowed_door_locations:
                continue
            geometry_checks.append((sig, part, part.matches_allowed_door_cell(observation.door_cell)))

        if geometry_checks and any(matches for _, _, matches in geometry_checks):
            matched_sig, matched_part, _ = next(item for item in geometry_checks if item[2])
            return ValidationResult(
                True,
                "high",
                "vanilla game-data allowed_door_locations matched after rotation normalization",
                observation,
                None,
                [],
                decision="allow",
                source="game_data",
                details={
                    "signature": matched_sig.to_dict(),
                    "door_cell": list(observation.door_cell),
                    "part_location": [matched_part.x0, matched_part.y0],
                    "rotation_normalization_offset": list(ROTATION_NORMALIZATION_OFFSETS[matched_part.rotation % 4]),
                },
            )
        return None

    def validate_observation(self, observation: Optional[PlacementObservation]) -> ValidationResult:
        override = self._validate_with_overrides(observation)
        if override is not None:
            return override

        pair_stats = self.pair_rules.get(observation.pair_key()) if observation else None
        side_keys = [observation.a.key(), observation.b.key()] if observation else []
        side_stats = [self.side_rules.get(side_keys[0]), self.side_rules.get(side_keys[1])] if side_keys else []

        if pair_stats and pair_stats.get("allow"):
            return ValidationResult(True, "high", "exact observed pair rule passed", observation, pair_stats, side_stats, decision="allow", source="inferred")

        if side_stats and all(stat and stat.get("allow") for stat in side_stats):
            return ValidationResult(True, "medium", "both part-side rules passed", observation, pair_stats, side_stats, decision="allow", source="inferred")

        return ValidationResult(
            False,
            "low",
            "missing or too-rare side/pair evidence in canonical corpus",
            observation,
            pair_stats,
            side_stats,
            decision="unresolved",
            source="inferred",
        )

    def validate_candidate(self, parts: Sequence[dict], door: dict) -> ValidationResult:
        if not isinstance(door, dict) or "Cell" not in door or "Orientation" not in door:
            return self.validate_observation(None)
        cell_to_parts, _ = build_cell_map(parts)
        cell = tuple(map(int, door["Cell"]))
        orientation = int(door["Orientation"])
        obs = resolve_observation_from_cells(cell_to_parts, cell, orientation)
        if obs is None:
            fallback = resolve_vanilla_allowed_location_fallback(cell_to_parts, cell, orientation)
            if fallback is not None:
                return fallback
        return self.validate_observation(obs)

    def validate_doors(self, parts: Sequence[dict], doors: Sequence[dict]) -> DoorValidationSummary:
        cell_to_parts, _ = build_cell_map(parts)
        per_door: List[ValidationResult] = []
        part_door_hits: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for idx, door in enumerate(doors):
            if not isinstance(door, dict) or "Cell" not in door or "Orientation" not in door:
                continue
            cell = tuple(map(int, door["Cell"]))
            orientation = int(door["Orientation"])
            obs = resolve_observation_from_cells(cell_to_parts, cell, orientation)
            result = resolve_vanilla_allowed_location_fallback(cell_to_parts, cell, orientation) if obs is None else None
            if result is None:
                result = self.validate_observation(obs)
            per_door.append(result)
            if result.observation is None:
                continue
            for part_index, sig in ((result.observation.a_index, result.observation.a), (result.observation.b_index, result.observation.b)):
                crew_rule = crew_rule_for_part(sig.part_id)
                if crew_rule is not None:
                    part_door_hits[(part_index, sig.part_id)].append(len(per_door) - 1)

        for key, indexes in part_door_hits.items():
            _, part_id = key
            crew_rule = crew_rule_for_part(part_id)
            if crew_rule is None or len(indexes) <= crew_rule["max_doors_total"]:
                continue
            for door_index in indexes[crew_rule["max_doors_total"]:]:
                result = per_door[door_index]
                result.allowed = False
                result.decision = "reject"
                result.confidence = "high"
                result.source = "override"
                result.reason = f"{crew_rule['label']} allows at most {crew_rule['max_doors_total']} door(s) total on the part"

        counts = Counter(result.decision for result in per_door)
        counts["doors_total"] = len(per_door)
        return DoorValidationSummary(per_door=per_door, counts=dict(counts))


def build_cell_map(parts: Sequence[dict]) -> Tuple[dict, Counter]:
    cell_to_parts: Dict[Coord, List[CandidatePart]] = defaultdict(list)
    unknown_part_ids: Counter = Counter()
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        part_id = normalize_part_id(part)
        if not part_id or "Location" not in part:
            continue
        if not is_vanilla_part_id(part_id):
            unknown_part_ids[part_id] += 1
            continue
        rotation = int(part.get("Rotation", 0)) % 4
        meta, inferred = infer_meta(part_id, rotation)
        if inferred:
            unknown_part_ids[part_id] += 1
        x0, y0 = map(int, part["Location"])
        candidate = CandidatePart(
            index=index,
            part_id=part_id,
            rotation=rotation,
            x0=x0,
            y0=y0,
            width=meta.width,
            height=meta.height,
            traversable=meta.traversable,
            footprint_tiles=meta.footprint_tiles,
            unblocked_tiles=meta.unblocked_tiles,
            blocked_travel_cells=meta.blocked_travel_cells,
            allowed_door_locations=meta.allowed_door_locations,
        )
        for dx, dy in meta.footprint_tiles:
            cell_to_parts[(x0 + dx, y0 + dy)].append(candidate)
    return cell_to_parts, unknown_part_ids


def choose_boundary_part(candidates: Sequence[CandidatePart], cell: Coord, side: str, banned: Optional[int] = None) -> Optional[Tuple[CandidatePart, SideSignature]]:
    matches: List[Tuple[int, int, int, str, CandidatePart, SideSignature]] = []
    for candidate in candidates:
        if banned is not None and candidate.index == banned:
            continue
        signature = candidate.signature_for_boundary(cell, side)
        if signature is None:
            continue
        walkable_rank = 0 if candidate.cell_is_walkable(cell) else 1
        traversable_rank = 0 if candidate.traversable else 1
        footprint_rank = len(candidate.footprint_tiles)
        matches.append((walkable_rank, traversable_rank, footprint_rank, candidate.part_id, candidate, signature))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4].index))
    _, _, _, _, candidate, signature = matches[0]
    return candidate, signature


def resolve_observation_from_cells(cell_to_parts: dict, cell: Coord, orientation: int) -> Optional[PlacementObservation]:
    adjacent_cells = shared_door_adjacent_cells(cell, orientation)
    if adjacent_cells is None:
        return None
    other, cell = adjacent_cells
    candidates_a = cell_to_parts.get(other)
    candidates_b = cell_to_parts.get(cell)
    if not candidates_a or not candidates_b:
        return None
    side_a, side_b = SIDE_BY_ORIENTATION[orientation]
    chosen_a = choose_boundary_part(candidates_a, other, side_a)
    if chosen_a is None:
        return None
    part_a, signature_a = chosen_a
    chosen_b = choose_boundary_part(candidates_b, cell, side_b, banned=part_a.index)
    if chosen_b is None:
        return None
    part_b, signature_b = chosen_b
    return PlacementObservation(
        orientation=orientation,
        door_cell=cell,
        a=signature_a,
        b=signature_b,
        a_index=part_a.index,
        b_index=part_b.index,
        a_part=part_a,
        b_part=part_b,
    )


def door_adjacent_cells(cell: Coord, orientation: int) -> Optional[Tuple[Coord, Coord]]:
    """Backward-compatible wrapper around the shared door-cell helper."""

    return shared_door_adjacent_cells(cell, orientation)


def match_allowed_door_cell_with_anchor_delta(part: CandidatePart, door_cell: Coord) -> Optional[Tuple[int, int]]:
    if not part.allowed_door_locations:
        return None
    ox, oy = ROTATION_NORMALIZATION_OFFSETS[part.rotation % 4]
    for dx, dy in FALLBACK_ALLOWED_LOCATION_ANCHOR_DELTAS:
        local = (door_cell[0] - (part.x0 + dx) + ox, door_cell[1] - (part.y0 + dy) + oy)
        if local in part.allowed_door_locations:
            return dx, dy
    return None


def resolve_vanilla_allowed_location_fallback(cell_to_parts: dict, cell: Coord, orientation: int) -> Optional[ValidationResult]:
    adjacent = door_adjacent_cells(cell, orientation)
    if adjacent is None:
        return None
    cell_a, cell_b = adjacent
    candidates_a = cell_to_parts.get(cell_a, [])
    candidates_b = cell_to_parts.get(cell_b, [])
    if not candidates_a or not candidates_b:
        return None
    shared_indexes = {part.index for part in candidates_a} & {part.index for part in candidates_b}
    if not shared_indexes:
        return None

    touched = []
    for side_label, world_cell, candidates in (("a", cell_a, candidates_a), ("b", cell_b, candidates_b)):
        for part in candidates:
            touched.append((side_label, world_cell, part))

    matches = []
    for side_label, world_cell, part in touched:
        anchor_delta = match_allowed_door_cell_with_anchor_delta(part, cell)
        if anchor_delta is None:
            continue
        opposite_candidates = candidates_b if side_label == "a" else candidates_a
        distinct_neighbors = [neighbor for neighbor in opposite_candidates if neighbor.index != part.index]
        if not distinct_neighbors:
            continue
        matches.append((part, distinct_neighbors[0], side_label, world_cell, anchor_delta))

    if not matches:
        return None

    part, neighbor, matched_side, matched_world_cell, anchor_delta = sorted(
        matches,
        key=lambda item: (
            0 if item[4] == (0, 0) else 1,
            0 if item[0].cell_is_walkable(item[3]) else 1,
            len(item[0].footprint_tiles),
            item[0].part_id,
            item[0].index,
        ),
    )[0]
    return ValidationResult(
        True,
        "medium",
        "fallback matched vanilla allowed_door_locations in shared-cell overlap / anchor-drift case",
        None,
        None,
        [],
        decision="allow",
        source="game_data_fallback",
        details={
            "matched_part_id": part.part_id,
            "matched_part_index": part.index,
            "neighbor_part_id": neighbor.part_id,
            "matched_side_cell": list(matched_world_cell),
            "door_cell": list(cell),
            "orientation": orientation,
            "anchor_delta": list(anchor_delta),
        },
    )


def resolve_door_observation(parts: Sequence[dict], door: dict) -> Optional[PlacementObservation]:
    if not isinstance(door, dict) or "Cell" not in door or "Orientation" not in door:
        return None
    cell_to_parts, _ = build_cell_map(parts)
    cell = tuple(map(int, door["Cell"]))
    return resolve_observation_from_cells(cell_to_parts, cell, int(door["Orientation"]))


def iter_potential_boundaries(cell_to_parts: dict) -> Iterator[Tuple[Coord, int]]:
    for x, y in sorted(cell_to_parts):
        if (x + 1, y) in cell_to_parts:
            yield (x + 1, y), 1
        if (x, y + 1) in cell_to_parts:
            yield (x, y + 1), 0


def default_overrides() -> dict:
    return {
        "schema_version": 3,
        "vanilla_only": True,
        "vanilla_scope": "Curated semantic overrides are intentionally restricted to vanilla Cosmoteer part IDs only. Any non-vanilla part is excluded/unresolved for this phase.",
        "reject_classes": {
            "semantic_rule": "Armor, structure, and wedges can never host doors.",
            "match_hints": list(REJECT_CLASS_HINTS),
        },
        "crew_family_map": {
            "small": ["cosmoteer.crew_quarters_small"],
            "med": ["cosmoteer.crew_quarters_med"],
            "large": ["cosmoteer.crew_quarters_large"],
        },
        "crew_rules": {
            "cosmoteer.crew_quarters_small": {
                "label": "bunk",
                "max_doors_total": 1,
                "notes": "Corpus-backed orientation mapping of the user's 1x2/2x1 bunk sketch into stored coordinates.",
                "allowed": [
                    {"rotation": 0, "side": "W", "offsets": [0], "width": 2, "height": 1},
                    {"rotation": 0, "side": "N", "offsets": [0, 1], "width": 2, "height": 1},
                    {"rotation": 1, "side": "W", "offsets": [0, 1], "width": 1, "height": 2},
                    {"rotation": 1, "side": "N", "offsets": [0], "width": 1, "height": 2},
                    {"rotation": 2, "side": "W", "offsets": [0], "width": 2, "height": 1},
                    {"rotation": 2, "side": "N", "offsets": [0, 1], "width": 2, "height": 1},
                    {"rotation": 3, "side": "W", "offsets": [0, 1], "width": 1, "height": 2},
                    {"rotation": 3, "side": "N", "offsets": [0], "width": 1, "height": 2},
                ],
            },
            "cosmoteer.crew_quarters_med": {
                "label": "quarters",
                "max_doors_total": 2,
                "notes": "Only one side family is valid; both door sites on that side may be used together.",
                "allowed": [
                    {"rotation": 0, "side": "W", "offsets": [0, 1], "width": 3, "height": 2},
                    {"rotation": 1, "side": "N", "offsets": [0, 1], "width": 2, "height": 3},
                    {"rotation": 2, "side": "W", "offsets": [0, 1], "width": 3, "height": 2},
                    {"rotation": 3, "side": "N", "offsets": [0, 1], "width": 2, "height": 3},
                ],
            },
            "cosmoteer.crew_quarters_large": {
                "label": "barracks",
                "max_doors_total": 2,
                "notes": "Middle-top and middle-bottom on the short sides, rotation-mapped from corpus evidence.",
                "allowed": [
                    {"rotation": 0, "side": "W", "offsets": [0, 2], "width": 4, "height": 3},
                    {"rotation": 1, "side": "N", "offsets": [0, 2], "width": 3, "height": 4},
                    {"rotation": 2, "side": "W", "offsets": [0, 2], "width": 4, "height": 3},
                    {"rotation": 3, "side": "N", "offsets": [0, 2], "width": 3, "height": 4},
                ],
            },
        },
    }


def classify_override_reject(sig: SideSignature) -> Optional[str]:
    low = sig.part_id.lower()
    if any(token in low for token in REJECT_CLASS_HINTS):
        return f"override rejects {sig.part_id}: armor/structure/wedge classes can never host doors"
    return None


def crew_rule_for_part(part_id: str) -> Optional[dict]:
    return default_overrides()["crew_rules"].get(part_id)


def signature_matches_allowed(sig: SideSignature, allowed: dict) -> bool:
    return (
        sig.rotation == int(allowed["rotation"])
        and sig.side == allowed["side"]
        and sig.offset in allowed["offsets"]
        and sig.width == int(allowed["width"])
        and sig.height == int(allowed["height"])
    )


def match_crew_override(observation: PlacementObservation) -> Optional[ValidationResult]:
    for sig in (observation.a, observation.b):
        rule = crew_rule_for_part(sig.part_id)
        if rule is None:
            continue
        if any(signature_matches_allowed(sig, allowed) for allowed in rule["allowed"]):
            return ValidationResult(
                True,
                "high",
                f"override allows {rule['label']} door site for {sig.part_id}",
                observation,
                None,
                [],
                decision="allow",
                source="override",
                details={"signature": sig.to_dict(), "rule": rule["label"]},
            )
        return ValidationResult(
            False,
            "high",
            f"override rejects {sig.part_id}: door site is outside curated {rule['label']} positions",
            observation,
            None,
            [],
            decision="reject",
            source="override",
            details={"signature": sig.to_dict(), "rule": rule["label"]},
        )
    return None


def infer_rules_from_corpus(input_dir: Path, output_path: Path, thresholds: Thresholds | None = None) -> dict:
    thresholds = thresholds or Thresholds()
    side_observed: Counter = Counter()
    side_possible: Counter = Counter()
    pair_observed: Counter = Counter()
    pair_possible: Counter = Counter()
    unknown_part_ids: Counter = Counter()

    stats: Counter = Counter()
    sample_failures: List[dict] = []

    for ship_path in iter_ship_files(input_dir):
        stats["ships_processed"] += 1
        data = orjson.loads(ship_path.read_bytes())

        parts = data.get("Parts", [])
        all_part_ids = [normalize_part_id(part) for part in parts if isinstance(part, dict)]
        modded_part_ids = sorted({part_id for part_id in all_part_ids if part_id and not is_vanilla_part_id(part_id)})
        if modded_part_ids:
            stats["ships_with_modded_parts"] += 1
            stats["modded_parts_total"] += len(modded_part_ids)
        cell_to_parts, ship_unknowns = build_cell_map(parts)
        unknown_part_ids.update(ship_unknowns)

        observed_doors = set()
        for door in data.get("Doors", []):
            if not isinstance(door, dict) or "Cell" not in door or "Orientation" not in door:
                continue
            stats["doors_total"] += 1
            obs = resolve_observation_from_cells(cell_to_parts, tuple(map(int, door["Cell"])), int(door["Orientation"]))
            if obs is None:
                stats["doors_unresolved"] += 1
                if modded_part_ids:
                    stats["doors_excluded_modded_context"] += 1
                if len(sample_failures) < 20:
                    sample_failures.append({"ship": ship_path.name, "door": door, "reason": "unresolved against occupied-cell boundary model"})
                continue
            stats["doors_resolved"] += 1
            observed_doors.add((tuple(map(int, door["Cell"])), int(door["Orientation"])))
            side_observed[obs.a.key()] += 1
            side_observed[obs.b.key()] += 1
            pair_observed[obs.pair_key()] += 1

        for cell, orientation in iter_potential_boundaries(cell_to_parts):
            obs = resolve_observation_from_cells(cell_to_parts, cell, orientation)
            if obs is None:
                continue
            pair_possible[obs.pair_key()] += 1
            side_possible[obs.a.key()] += 1
            side_possible[obs.b.key()] += 1
            stats["candidate_boundaries"] += 1
            if (cell, orientation) in observed_doors:
                stats["observed_boundaries"] += 1
            else:
                stats["unoccupied_candidate_boundaries"] += 1

    side_rules = {}
    for key, possible in side_possible.items():
        observed = side_observed.get(key, 0)
        ratio = observed / possible if possible else 0.0
        allow = observed >= thresholds.min_side_observations and ratio >= thresholds.min_side_ratio
        side_rules[key] = {"observed": observed, "possible": possible, "observed_ratio": ratio, "allow": allow}

    pair_rules = {}
    for key, possible in pair_possible.items():
        observed = pair_observed.get(key, 0)
        ratio = observed / possible if possible else 0.0
        allow = observed >= thresholds.min_pair_observations and ratio >= thresholds.min_pair_ratio
        pair_rules[key] = {"observed": observed, "possible": possible, "observed_ratio": ratio, "allow": allow}

    rules = DoorPlacementRules({
        "thresholds": thresholds.__dict__,
        "rules": {"side_rules": side_rules, "pair_rules": pair_rules},
        "overrides": default_overrides(),
    })

    validation = validate_corpus_against_rules(input_dir, rules)

    payload = {
        "schema_version": 3,
        "vanilla_filter": {
            "enabled": True,
            "namespace_prefix": VANILLA_NAMESPACE,
            "policy": "Only vanilla cosmoteer.* parts participate in generator-safe training/inference and curated overrides. Any non-vanilla part is excluded and validates as unresolved for this phase.",
        },
        "corpus": {
            "input_dir": str(input_dir),
            "ships_processed": stats["ships_processed"],
            "doors_total": stats["doors_total"],
            "doors_resolved": stats["doors_resolved"],
            "doors_unresolved": stats["doors_unresolved"],
            "candidate_boundaries": stats["candidate_boundaries"],
            "observed_boundaries": stats["observed_boundaries"],
            "unoccupied_candidate_boundaries": stats["unoccupied_candidate_boundaries"],
            "unknown_part_ids": dict(unknown_part_ids.most_common()),
            "ships_with_modded_parts": stats["ships_with_modded_parts"],
            "modded_parts_total": stats["modded_parts_total"],
            "doors_excluded_modded_context": stats["doors_excluded_modded_context"],
        },
        "coordinate_frame": {
            "door_cell_semantics": "Door Cell names the right/bottom occupied cell of the doorway span: orientation 0 joins (x,y-1)<->(x,y), orientation 1 joins (x-1,y)<->(x,y).",
            "part_anchor_semantics": "Part Location remains the normalized top-left anchor of the extracted rotated footprint tiles.",
            "allowed_door_location_semantics": "Game-file allowed_door_locations live in the same doorway frame after subtracting the rotation normalization offset: rot0=(0,0), rot1=(1,0), rot2=(1,1), rot3=(0,1).",
            "normalization": "Rules are stored in rotated local footprint coordinates as (side, offset) signatures plus part id and rotation.",
            "caveat": "Crew bunks / quarters may still drift historically from current game-data door locations, so remaining unresolveds there should be bucketed separately.",
        },
        "thresholds": thresholds.__dict__,
        "rules": {"side_rules": side_rules, "pair_rules": pair_rules},
        "overrides": default_overrides(),
        "validation": validation,
        "samples": {"unresolved_doors": sample_failures},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        fh.write(b"\n")
    return payload


def validate_corpus_against_rules(input_dir: Path, rules: DoorPlacementRules) -> dict:
    stats: Counter = Counter()
    for ship_path in iter_ship_files(input_dir):
        data = orjson.loads(ship_path.read_bytes())
        parts = data.get("Parts", [])
        summary = rules.validate_doors(parts, data.get("Doors", []))
        stats["ships_total"] += 1
        stats["doors_total"] += summary.counts.get("doors_total", 0)
        stats["doors_allow"] += summary.counts.get("allow", 0)
        stats["doors_reject"] += summary.counts.get("reject", 0)
        stats["doors_unresolved"] += summary.counts.get("unresolved", 0)
        if summary.counts.get("reject", 0):
            stats["ships_with_rejects"] += 1
        if summary.counts.get("unresolved", 0):
            stats["ships_with_unresolved"] += 1
    return dict(stats)
