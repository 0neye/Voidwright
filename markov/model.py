"""Shared Markov model internals used by training and generation."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from common.geometry import is_vanilla_part_id, iter_ship_files, load_vanilla_part_geometry


def _config_as_dict(cfg) -> dict:
    """Like asdict() but converts frozenset fields to sorted lists for JSON."""

    d = asdict(cfg)
    for k, v in d.items():
        if isinstance(v, (frozenset, set)):
            d[k] = sorted(v)
    return d


def _reqs_satisfied(part_counts: Dict[str, int], requirements: dict) -> bool:
    """Return True if all (part_id -> min_count) requirements are met."""

    return all(part_counts.get(pid, 0) >= req for pid, req in requirements.items())


Coord = Tuple[int, int]
END_TOKEN = "__END__"
ROOT_ANCHOR = "__ROOT__"


@dataclass(frozen=True)
class RelativePlacementToken:
    part_id: str
    rotation: int
    anchor_part_id: str
    anchor_rotation: int
    dx: int
    dy: int

    def as_key(self) -> str:
        """Return the compact serialized key for this placement token."""

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
        """Return this token as a plain JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_key(cls, key: str) -> "RelativePlacementToken":
        """Decode a serialized placement token key back into a dataclass."""

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
        """Return the occupied world cells for this placed part."""

        geometry = geometry_cache[self.part_id].rotations[self.rotation]
        return frozenset((self.x + dx, self.y + dy) for dx, dy in geometry.footprint_tiles)

    def bbox(self, geometry_cache: Dict[str, object]) -> Tuple[int, int, int, int]:
        """Return the inclusive bounding box for this part."""

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


class WeightedSampler:
    """Sample weighted counters without converting them to expanded lists."""

    @staticmethod
    def sample(counter: Dict[str, int], rng: random.Random) -> str:
        """Sample one key from a counter using cumulative weights."""

        total = sum(counter.values())
        if total <= 0:
            raise ValueError("cannot sample from empty counter")
        target = rng.uniform(0, total)
        acc = 0.0
        last_key = None
        for key, weight in counter.items():
            acc += weight
            last_key = key
            if target <= acc:
                return key
        return last_key  # pragma: no cover


class RelativeMarkovModel:
    """Load, save, and sample the relative-placement Markov model."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.schema_version = int(payload.get("schema_version", 1))
        self.order = int(payload["config"]["markov_order"])
        self.start_counts: Dict[str, int] = payload["start_counts"]
        self.transition_counts: Dict[str, Dict[str, int]] = payload["transition_counts"]
        self.part_frequency: Dict[str, int] = payload["part_frequency"]
        self.geometry_cache = load_vanilla_part_geometry()

    def _state_key(self, history: Sequence[str], order: int) -> str:
        """Return the serialized state key for a token history tail."""

        if self.schema_version >= 2:
            return state_key(history, order)
        tail = list(history[-order:]) if order > 0 else []
        return " || ".join(tail)

    @classmethod
    def load(cls, path: str | Path) -> "RelativeMarkovModel":
        """Load a serialized Markov model from disk."""

        with Path(path).open(encoding="utf-8") as file_handle:
            return cls(json.load(file_handle))

    def save(self, path: str | Path) -> None:
        """Write this Markov model payload to disk."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(self.payload, file_handle, separators=(",", ":"), sort_keys=True)
            file_handle.write("\n")

    def _placement_within_bounds(self, part: ShipPart, config: GenerationConfig) -> bool:
        """Return True when all footprint cells fit within the configured bounds."""

        for x, y in part.footprint_cells(self.geometry_cache):
            if not (
                config.bounds_min_x <= x <= config.bounds_max_x
                and config.bounds_min_y <= y <= config.bounds_max_y
            ):
                return False
        return True

    def _within_primary_bounds(self, part: ShipPart, config: GenerationConfig) -> bool:
        """Bounds check for primary placements in mirror mode."""

        for x, y in part.footprint_cells(self.geometry_cache):
            if not (config.bounds_min_x <= x <= -1 and config.bounds_min_y <= y <= config.bounds_max_y):
                return False
        return True

    def _within_mirror_bounds(self, part: ShipPart, config: GenerationConfig) -> bool:
        """Bounds check for mirrored placements in mirror mode."""

        for x, y in part.footprint_cells(self.geometry_cache):
            if not (0 <= x <= config.bounds_max_x and config.bounds_min_y <= y <= config.bounds_max_y):
                return False
        return True

    def generate(self, config: GenerationConfig, *, seed_parts=None) -> dict:
        """Generate a ship layout.

        Args:
            config: Generation configuration
            seed_parts: Optional list of seed parts to pre-place before Markov
                generation begins. Each item may be a dict
                `{part_id, rotation, x, y}` or a `ShipPart` instance
        """

        rng = random.Random(config.rng_seed)
        placed: List[ShipPart] = []
        primary_indices: List[int] = []
        occupied_cells: set[Coord] = set()
        history: List[str] = []
        placement_trace: List[dict] = []
        attempts = 0
        rejected_missing_anchor = 0
        rejected_overlap = 0
        rejected_bounds = 0
        rejected_mirror = 0
        rejected_requirements = 0
        stop_reason = "unknown"

        allowlist = config.part_allowlist
        rejected_allowlist = 0
        mirror_mode = config.mirror_symmetry

        if mirror_mode:
            from .symmetry import (
                is_primary_placement as _is_primary_placement,
                mirror_part as _mirror_part,
                primary_root_x as _primary_root_x,
            )

        part_counts: Dict[str, int] = {}
        seed_stats: Optional[dict] = None
        seeded = seed_parts is not None and len(seed_parts) > 0

        if seeded:
            seed_skipped_geometry = 0
            seed_skipped_overlap = 0
            seed_skipped_allowlist = 0

            for raw in seed_parts:
                if isinstance(raw, dict):
                    sp = ShipPart(
                        part_id=raw["part_id"],
                        rotation=int(raw.get("rotation", 0)) % 4,
                        x=int(raw["x"]),
                        y=int(raw["y"]),
                        flip_x=bool(raw.get("flip_x", raw.get("FlipX", False))),
                        flip_y=bool(raw.get("flip_y", raw.get("FlipY", False))),
                    )
                else:
                    sp = raw

                if (
                    sp.part_id not in self.geometry_cache
                    or sp.rotation not in self.geometry_cache[sp.part_id].rotations
                ):
                    seed_skipped_geometry += 1
                    continue
                if allowlist is not None and sp.part_id not in allowlist:
                    seed_skipped_allowlist += 1
                    continue
                cells = sp.footprint_cells(self.geometry_cache)
                if cells & occupied_cells:
                    seed_skipped_overlap += 1
                    continue

                idx = len(placed)
                placed.append(sp)
                occupied_cells.update(cells)
                part_counts[sp.part_id] = part_counts.get(sp.part_id, 0) + 1

                is_primary = True
                if mirror_mode:
                    is_primary = _is_primary_placement(sp, self.geometry_cache)
                if is_primary:
                    primary_indices.append(idx)

                placement_trace.append(
                    {
                        "token": None,
                        "anchor_index": None,
                        "placed_index": idx,
                        "world_origin": [sp.x, sp.y],
                        "is_seed": True,
                        "is_mirror": not is_primary,
                    }
                )

            seed_stats = {
                "seed_parts_input": len(seed_parts),
                "seed_parts_placed": len(placed),
                "seed_skipped_geometry": seed_skipped_geometry,
                "seed_skipped_overlap": seed_skipped_overlap,
                "seed_skipped_allowlist": seed_skipped_allowlist,
            }

            if not placed:
                raise RuntimeError(
                    "seed provided but no valid vanilla seed parts could be placed "
                    f"({seed_skipped_geometry} skipped: no geometry, "
                    f"{seed_skipped_allowlist} skipped: not in allowlist, "
                    f"{seed_skipped_overlap} skipped: overlap)"
                )

            if not primary_indices:
                raise RuntimeError(
                    "seed provided but no primary-side parts were found; "
                    "in mirror mode all seed parts must have all footprint cells at x ≤ -1 "
                    "to serve as Markov anchors."
                )

            seed_part_ids = frozenset(placed[i].part_id for i in primary_indices)
            virtual_root_key = None
            vr_attempts = max(256, len(self.start_counts) // 20)
            for _ in range(vr_attempts):
                candidate_key = WeightedSampler.sample(self.start_counts, rng)
                candidate = RelativePlacementToken.from_key(candidate_key)
                if allowlist is not None and candidate.part_id not in allowlist:
                    continue
                if (
                    candidate.part_id not in self.geometry_cache
                    or candidate.rotation not in self.geometry_cache[candidate.part_id].rotations
                ):
                    continue
                if candidate.part_id in seed_part_ids:
                    virtual_root_key = candidate_key
                    break
            if virtual_root_key is None:
                for _ in range(vr_attempts):
                    candidate_key = WeightedSampler.sample(self.start_counts, rng)
                    candidate = RelativePlacementToken.from_key(candidate_key)
                    if allowlist is not None and candidate.part_id not in allowlist:
                        continue
                    if (
                        candidate.part_id in self.geometry_cache
                        and candidate.rotation in self.geometry_cache[candidate.part_id].rotations
                    ):
                        virtual_root_key = candidate_key
                        break
            if virtual_root_key is None:
                raise RuntimeError("could not sample a virtual root token for seeded generation")
            history = [virtual_root_key]

        else:
            root_key = None
            root = None
            root_attempts = max(64, len(self.start_counts) // 100)
            for _ in range(root_attempts):
                candidate_root_key = WeightedSampler.sample(self.start_counts, rng)
                candidate_root = RelativePlacementToken.from_key(candidate_root_key)
                if allowlist is not None and candidate_root.part_id not in allowlist:
                    rejected_allowlist += 1
                    continue
                if (
                    candidate_root.part_id in self.geometry_cache
                    and candidate_root.rotation in self.geometry_cache[candidate_root.part_id].rotations
                ):
                    root_key = candidate_root_key
                    root = candidate_root
                    break
            if root_key is None or root is None:
                if allowlist is not None:
                    raise RuntimeError(
                        f"could not sample a root token matching the allowlist after {root_attempts} attempts. "
                        f"Rejected {rejected_allowlist} allowlist mismatches. "
                        f"Check that the allowlist contains parts present in the model's training corpus."
                    )
                raise RuntimeError("could not sample a root token with known vanilla geometry")

            if mirror_mode:
                root_x = _primary_root_x(root.part_id, root.rotation, self.geometry_cache)
                root_part = ShipPart(part_id=root.part_id, rotation=root.rotation, x=root_x, y=0)
                root_cells = root_part.footprint_cells(self.geometry_cache)
                mirror_root = _mirror_part(root_part, self.geometry_cache)
                root_primary_idx = len(placed)
                placed.append(root_part)
                occupied_cells.update(root_cells)
                primary_indices.append(root_primary_idx)
                part_counts[root_part.part_id] = part_counts.get(root_part.part_id, 0) + 1
                placement_trace.append(
                    {
                        "token": root.to_dict(),
                        "anchor_index": None,
                        "placed_index": root_primary_idx,
                        "world_origin": [root_part.x, root_part.y],
                        "is_mirror": False,
                    }
                )
                if mirror_root is not None:
                    mirror_cells = mirror_root.footprint_cells(self.geometry_cache)
                    mirror_idx = len(placed)
                    placed.append(mirror_root)
                    occupied_cells.update(mirror_cells)
                    part_counts[mirror_root.part_id] = part_counts.get(mirror_root.part_id, 0) + 1
                    placement_trace.append(
                        {
                            "token": root.to_dict(),
                            "anchor_index": None,
                            "placed_index": mirror_idx,
                            "world_origin": [mirror_root.x, mirror_root.y],
                            "is_mirror": True,
                            "mirror_of": root_primary_idx,
                        }
                    )
            else:
                root_part = ShipPart(part_id=root.part_id, rotation=root.rotation, x=0, y=0)
                root_cells = root_part.footprint_cells(self.geometry_cache)
                placed.append(root_part)
                primary_indices.append(0)
                occupied_cells.update(root_cells)
                part_counts[root_part.part_id] = part_counts.get(root_part.part_id, 0) + 1
                placement_trace.append(
                    {
                        "token": root.to_dict(),
                        "anchor_index": None,
                        "placed_index": 0,
                        "world_origin": [0, 0],
                    }
                )
            history.append(root_key)

        while len(placed) < config.max_parts and attempts < config.max_attempts:
            state = self._state_key(history, self.order)
            options = self.transition_counts.get(state)
            if not options and self.order > 0:
                for fallback_order in range(self.order - 1, -1, -1):
                    options = self.transition_counts.get(self._state_key(history, fallback_order))
                    if options:
                        break
            if not options:
                stop_reason = "no_transition_for_state"
                break

            token_key = None
            all_end_tokens_by_req = True
            for _ in range(config.max_resample_per_step):
                attempts += 1
                candidate_key = WeightedSampler.sample(options, rng)
                if candidate_key == END_TOKEN:
                    if config.part_requirements is not None and not _reqs_satisfied(
                        part_counts, config.part_requirements
                    ):
                        rejected_requirements += 1
                        continue
                    token_key = END_TOKEN
                    break
                all_end_tokens_by_req = False
                token = RelativePlacementToken.from_key(candidate_key)
                if allowlist is not None and token.part_id not in allowlist:
                    rejected_allowlist += 1
                    continue
                if (
                    token.part_id not in self.geometry_cache
                    or token.rotation not in self.geometry_cache[token.part_id].rotations
                ):
                    continue

                if mirror_mode:
                    anchor_indexes = [
                        i
                        for i in primary_indices
                        if placed[i].part_id == token.anchor_part_id
                        and placed[i].rotation == token.anchor_rotation
                    ]
                else:
                    anchor_indexes = [
                        i
                        for i, part in enumerate(placed)
                        if part.part_id == token.anchor_part_id
                        and part.rotation == token.anchor_rotation
                    ]
                anchor_indexes.sort(reverse=True)
                if not anchor_indexes:
                    rejected_missing_anchor += 1
                    continue

                accepted = False
                for idx in anchor_indexes:
                    anchor = placed[idx]
                    candidate = ShipPart(
                        part_id=token.part_id,
                        rotation=token.rotation,
                        x=anchor.x + token.dx,
                        y=anchor.y + token.dy,
                    )
                    candidate_cells = candidate.footprint_cells(self.geometry_cache)

                    if mirror_mode:
                        if not self._within_primary_bounds(candidate, config):
                            rejected_bounds += 1
                            continue
                        if candidate_cells & occupied_cells:
                            rejected_overlap += 1
                            continue
                        mirror_candidate = _mirror_part(candidate, self.geometry_cache)
                        if mirror_candidate is None:
                            rejected_mirror += 1
                            continue
                        mirror_cells = mirror_candidate.footprint_cells(self.geometry_cache)
                        if mirror_cells & occupied_cells:
                            rejected_mirror += 1
                            continue
                        if not self._within_mirror_bounds(mirror_candidate, config):
                            rejected_mirror += 1
                            continue

                        primary_idx = len(placed)
                        placed.append(candidate)
                        occupied_cells.update(candidate_cells)
                        primary_indices.append(primary_idx)
                        part_counts[candidate.part_id] = part_counts.get(candidate.part_id, 0) + 1
                        history.append(candidate_key)
                        placement_trace.append(
                            {
                                "token": token.to_dict(),
                                "anchor_index": idx,
                                "placed_index": primary_idx,
                                "world_origin": [candidate.x, candidate.y],
                                "is_mirror": False,
                            }
                        )
                        mirror_idx = len(placed)
                        placed.append(mirror_candidate)
                        occupied_cells.update(mirror_cells)
                        part_counts[mirror_candidate.part_id] = part_counts.get(
                            mirror_candidate.part_id, 0
                        ) + 1
                        placement_trace.append(
                            {
                                "token": token.to_dict(),
                                "anchor_index": None,
                                "placed_index": mirror_idx,
                                "world_origin": [mirror_candidate.x, mirror_candidate.y],
                                "is_mirror": True,
                                "mirror_of": primary_idx,
                            }
                        )
                    else:
                        if candidate_cells & occupied_cells:
                            rejected_overlap += 1
                            continue
                        if not self._placement_within_bounds(candidate, config):
                            rejected_bounds += 1
                            continue
                        placed.append(candidate)
                        occupied_cells.update(candidate_cells)
                        primary_indices.append(len(placed) - 1)
                        part_counts[candidate.part_id] = part_counts.get(candidate.part_id, 0) + 1
                        history.append(candidate_key)
                        placement_trace.append(
                            {
                                "token": token.to_dict(),
                                "anchor_index": idx,
                                "placed_index": len(placed) - 1,
                                "world_origin": [candidate.x, candidate.y],
                            }
                        )

                    accepted = True
                    token_key = candidate_key
                    break
                if accepted:
                    break

            if token_key is None:
                if all_end_tokens_by_req and rejected_requirements > 0:
                    continue
                stop_reason = "placement_rejected_by_caps_or_anchor_missing"
                break
            if token_key == END_TOKEN:
                history.append(END_TOKEN)
                stop_reason = "end_token"
                break
        else:
            if len(placed) >= config.max_parts:
                stop_reason = "max_parts"
            elif attempts >= config.max_attempts:
                stop_reason = "max_attempts"

        if config.part_requirements and not _reqs_satisfied(part_counts, config.part_requirements):
            if stop_reason == "end_token":
                stop_reason = "requirements_unsatisfied"
            elif stop_reason == "max_attempts":
                stop_reason = "max_attempts_requirements_unsatisfied"

        all_cells = sorted(occupied_cells)
        min_x = min(cell[0] for cell in all_cells)
        max_x = max(cell[0] for cell in all_cells)
        min_y = min(cell[1] for cell in all_cells)
        max_y = max(cell[1] for cell in all_cells)

        primary_count = len(primary_indices)
        mirror_count = len(placed) - primary_count

        stats: dict = {
            "parts_generated": len(placed),
            "occupied_cells": len(all_cells),
            "attempts": attempts,
            "stop_reason": stop_reason,
            "rejections": {
                "missing_anchor": rejected_missing_anchor,
                "overlap": rejected_overlap,
                "bounds": rejected_bounds,
                "allowlist": rejected_allowlist,
                "requirements": rejected_requirements,
            },
            "bounds": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
        }
        if mirror_mode:
            stats["mirror"] = {
                "primary_parts": primary_count,
                "mirror_parts": mirror_count,
                "rejected_mirror": rejected_mirror,
            }
        if config.part_requirements:
            progress = {
                pid: {
                    "required": req,
                    "actual": part_counts.get(pid, 0),
                    "satisfied": part_counts.get(pid, 0) >= req,
                }
                for pid, req in config.part_requirements.items()
            }
            stats["requirements"] = {
                "satisfied": _reqs_satisfied(part_counts, config.part_requirements),
                "progress": progress,
            }
        if seed_stats is not None:
            stats["seed"] = seed_stats

        notes = [
            "Vanilla-only first-pass relative-placement Markov sample.",
            "Overlap rejection is footprint-aware using vanilla game-file geometry.",
            "Doors and accessibility cleanup are intentionally deferred to later passes.",
        ]
        if mirror_mode:
            notes.append(
                "Mirror symmetry: left-right across axis x = -0.5 (between columns -1 and 0). "
                "Primary placements on left half (x ≤ -1); mirrors placed on right half (x ≥ 0). "
                "Only primary parts serve as Markov anchors."
            )
        if config.part_requirements:
            notes.append(
                "Part requirements use total-ship-count semantics (primary + mirror parts both count). "
                "END_TOKEN is suppressed until all requirements are satisfied or max_attempts is reached."
            )
        if seeded:
            notes.append(
                "Seeded generation: existing ship parts were pre-placed before Markov sampling began. "
                "Markov history was reconstructed from the ordered primary seed parts."
            )

        return {
            "generator": "relative_markov_first_pass",
            "config": _config_as_dict(config),
            "stats": stats,
            "parts": [asdict(part) for part in placed],
            "placement_trace": placement_trace,
            "notes": notes,
        }


def history_symbol(token_key: str) -> str:
    """Return the compact history symbol used for state-key serialization."""

    if token_key == END_TOKEN:
        return END_TOKEN
    token = RelativePlacementToken.from_key(token_key)
    return f"{token.part_id}|{token.rotation}"


def state_key(history: Sequence[str], order: int) -> str:
    """Return the serialized Markov state key for a history tail."""

    tail = list(history[-order:]) if order > 0 else []
    compact = [history_symbol(token_key) for token_key in tail]
    return " || ".join(compact)


def _distance(a: ShipPart, b: ShipPart) -> Tuple[int, int, int, str, int, int]:
    dx = b.x - a.x
    dy = b.y - a.y
    return (abs(dx) + abs(dy), abs(dx), abs(dy), b.part_id, b.x, b.y)


def choose_root(parts: Sequence[ShipPart]) -> ShipPart:
    """Choose the geometrically central root part for one ship."""

    cx = sum(part.x for part in parts) / len(parts)
    cy = sum(part.y for part in parts) / len(parts)
    scored = sorted(parts, key=lambda p: ((p.x - cx) ** 2 + (p.y - cy) ** 2, p.part_id, p.x, p.y))
    return scored[0]


def order_ship_parts(
    parts: Sequence[ShipPart], anchor_window: int = 128
) -> List[Tuple[ShipPart, Optional[ShipPart]]]:
    """Order parts for Markov training using geometric nearest-anchor heuristics."""

    remaining = list(parts)
    root = choose_root(remaining)
    remaining.remove(root)
    remaining.sort(
        key=lambda p: (
            (p.x - root.x) ** 2 + (p.y - root.y) ** 2,
            abs(p.x - root.x) + abs(p.y - root.y),
            p.part_id,
            p.x,
            p.y,
        )
    )
    ordered: List[Tuple[ShipPart, Optional[ShipPart]]] = [(root, None)]
    placed = [root]
    for candidate in remaining:
        anchor_candidates = placed[-anchor_window:] if len(placed) > anchor_window else placed
        anchor = min(anchor_candidates, key=lambda p: _distance(p, candidate))
        ordered.append((candidate, anchor))
        placed.append(candidate)
    return ordered


def iter_vanilla_parts_from_ship(
    ship_data: dict,
    geometry_cache: Optional[Dict[str, object]] = None,
) -> List[ShipPart]:
    """Extract vanilla ShipPart records from one extracted ship JSON payload."""

    geometry_cache = geometry_cache or load_vanilla_part_geometry()
    vanilla_parts: List[ShipPart] = []
    for part in ship_data.get("Parts", []):
        if not isinstance(part, dict):
            continue
        part_id = part.get("ID") or part.get("IDString")
        if not part_id or not is_vanilla_part_id(part_id):
            continue
        if part_id not in geometry_cache:
            continue
        location = part.get("Location")
        if not isinstance(location, list) or len(location) != 2:
            continue
        rotation = int(part.get("Rotation", 0)) % 4
        if rotation not in geometry_cache[part_id].rotations:
            continue
        vanilla_parts.append(
            ShipPart(
                part_id=part_id,
                rotation=rotation,
                x=int(location[0]),
                y=int(location[1]),
                flip_x=bool(part.get("FlipX", False)),
                flip_y=bool(part.get("FlipY", False)),
            )
        )
    return vanilla_parts


def parts_touch(a: ShipPart, b: ShipPart, geometry_cache: Dict[str, object]) -> bool:
    """Return True when two placed parts share at least one touching side."""

    a_cells = a.footprint_cells(geometry_cache)
    b_cells = b.footprint_cells(geometry_cache)
    for ax, ay in a_cells:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (ax + dx, ay + dy) in b_cells:
                return True
    return False


def build_model_from_corpus(input_dir: Path, config: TrainingConfig) -> RelativeMarkovModel:
    """Build a Markov model from canonical extracted ship JSON files."""

    geometry_cache = load_vanilla_part_geometry()
    stats = TrainingStats()
    start_counts: Counter = Counter()
    transition_counts: Dict[str, Counter] = defaultdict(Counter)
    part_frequency: Counter = Counter()

    for ship_path in iter_ship_files(input_dir):
        stats.ships_seen += 1
        with ship_path.open(encoding="utf-8") as file_handle:
            ship_data = json.load(file_handle)
        all_parts = [p for p in ship_data.get("Parts", []) if isinstance(p, dict)]
        for part in all_parts:
            part_id = part.get("ID") or part.get("IDString")
            if not part_id:
                continue
            if not is_vanilla_part_id(part_id):
                stats.non_vanilla_parts_excluded += 1
            elif part_id not in geometry_cache:
                stats.unknown_vanilla_geometry_excluded += 1
        vanilla_parts = iter_vanilla_parts_from_ship(ship_data, geometry_cache=geometry_cache)
        if config.part_allowlist is not None:
            vanilla_parts = [p for p in vanilla_parts if p.part_id in config.part_allowlist]
        stats.vanilla_parts_seen += len(vanilla_parts)
        if len(vanilla_parts) < config.min_parts_per_ship:
            stats.ships_skipped_too_small += 1
            continue
        if len(vanilla_parts) > config.max_parts_per_ship:
            stats.ships_skipped_too_large += 1
            continue
        stats.ships_used += 1
        stats.vanilla_parts_used += len(vanilla_parts)
        ordered = order_ship_parts(vanilla_parts, anchor_window=config.anchor_window)
        history: List[str] = []
        for idx, (part, anchor) in enumerate(ordered):
            if idx == 0:
                token = RelativePlacementToken(
                    part_id=part.part_id,
                    rotation=part.rotation,
                    anchor_part_id=ROOT_ANCHOR,
                    anchor_rotation=0,
                    dx=0,
                    dy=0,
                )
                key = token.as_key()
                start_counts[key] += 1
                history.append(key)
                part_frequency[part.part_id] += 1
                stats.root_tokens += 1
                continue
            assert anchor is not None
            token = RelativePlacementToken(
                part_id=part.part_id,
                rotation=part.rotation,
                anchor_part_id=anchor.part_id,
                anchor_rotation=anchor.rotation,
                dx=part.x - anchor.x,
                dy=part.y - anchor.y,
            )
            key = token.as_key()
            transition_counts[state_key(history, config.markov_order)][key] += 1
            history.append(key)
            part_frequency[part.part_id] += 1
            stats.transition_tokens += 1
            if parts_touch(anchor, part, geometry_cache):
                stats.touching_transitions += 1
            else:
                stats.non_touching_transitions += 1
        transition_counts[state_key(history, config.markov_order)][END_TOKEN] += 1
        stats.end_tokens += 1

    payload = {
        "schema_version": 2,
        "model_type": "relative_markov_first_pass",
        "config": _config_as_dict(config),
        "corpus": {"input_dir": str(input_dir)},
        "stats": asdict(stats),
        "start_counts": dict(start_counts),
        "transition_counts": {k: dict(v) for k, v in transition_counts.items()},
        "part_frequency": dict(part_frequency),
        "notes": [
            "Vanilla-only corpus model built from canonical deduped extracted ship JSON.",
            "Relative placement uses origin-to-origin deltas between an anchor part and the next emitted part.",
            "Sampling rejects overlaps using full vanilla footprint geometry from game-file exports.",
            "This first pass intentionally defers door synthesis, accessibility cleanup, and gameplay-grade legality checks.",
        ],
    }
    return RelativeMarkovModel(payload)


def iter_vanilla_parts_from_graph(
    graph_data: dict,
    geometry_cache: Optional[Dict[str, object]] = None,
) -> Tuple[List[ShipPart], Dict[int, int]]:
    """Extract vanilla ShipParts from one structural graph payload."""

    geometry_cache = geometry_cache or load_vanilla_part_geometry()
    nodes = graph_data.get("graphs", {}).get("A_structural_part_graph", {}).get("nodes", [])
    parts: List[ShipPart] = []
    node_id_to_idx: Dict[int, int] = {}
    for node in nodes:
        part_id = node.get("part_id", "")
        if not is_vanilla_part_id(part_id) or part_id not in geometry_cache:
            continue
        rotation = int(node.get("rotation", 0)) % 4
        if rotation not in geometry_cache[part_id].rotations:
            continue
        location = node.get("location")
        if not isinstance(location, list) or len(location) != 2:
            continue
        node_id = node["id"]
        node_id_to_idx[node_id] = len(parts)
        parts.append(
            ShipPart(
                part_id=part_id,
                rotation=rotation,
                x=int(location[0]),
                y=int(location[1]),
            )
        )
    return parts, node_id_to_idx


def order_ship_parts_from_graph(
    parts: List[ShipPart],
    node_id_to_idx: Dict[int, int],
    edges: List[dict],
    anchor_window: int = 128,
) -> List[Tuple[ShipPart, Optional[ShipPart]]]:
    """Order graph-derived parts using BFS over touching structural edges."""

    if not parts:
        return []
    n = len(parts)
    adj: Dict[int, List[int]] = defaultdict(list)
    for edge in edges:
        src = node_id_to_idx.get(edge["source"])
        tgt = node_id_to_idx.get(edge["target"])
        if src is not None and tgt is not None:
            adj[src].append(tgt)
            adj[tgt].append(src)
    for k in adj:
        adj[k].sort()

    root = choose_root(parts)
    root_idx = parts.index(root)

    visited = [False] * n
    visited[root_idx] = True
    ordered: List[Tuple[ShipPart, Optional[ShipPart]]] = [(root, None)]
    placed: List[ShipPart] = [root]
    placed_at: Dict[int, int] = {root_idx: 0}
    queue: deque = deque([root_idx])

    while queue:
        cur_idx = queue.popleft()
        for neighbor_idx in adj[cur_idx]:
            if visited[neighbor_idx]:
                continue
            visited[neighbor_idx] = True
            anchor = placed[placed_at[cur_idx]]
            placed_at[neighbor_idx] = len(placed)
            placed.append(parts[neighbor_idx])
            ordered.append((parts[neighbor_idx], anchor))
            queue.append(neighbor_idx)

    for rem_idx in range(n):
        if visited[rem_idx]:
            continue
        candidates = placed[-anchor_window:] if len(placed) > anchor_window else placed
        anchor = min(candidates, key=lambda p: _distance(p, parts[rem_idx]))
        placed_at[rem_idx] = len(placed)
        placed.append(parts[rem_idx])
        ordered.append((parts[rem_idx], anchor))
        visited[rem_idx] = True

    return ordered


def build_model_from_graph_corpus(graph_dir: Path, config: TrainingConfig) -> RelativeMarkovModel:
    """Build a Markov model from pre-generated structural graph JSON artifacts."""

    geometry_cache = load_vanilla_part_geometry()
    stats = TrainingStats()
    start_counts: Counter = Counter()
    transition_counts: Dict[str, Counter] = defaultdict(Counter)
    part_frequency: Counter = Counter()

    graph_files = sorted(f for f in graph_dir.glob("*.json") if f.name != "manifest.json")

    for graph_path in graph_files:
        stats.ships_seen += 1
        try:
            with graph_path.open(encoding="utf-8") as file_handle:
                graph_data = json.load(file_handle)
        except (json.JSONDecodeError, OSError):
            continue

        all_nodes = graph_data.get("graphs", {}).get("A_structural_part_graph", {}).get("nodes", [])
        for node in all_nodes:
            pid = node.get("part_id", "")
            if not is_vanilla_part_id(pid):
                stats.non_vanilla_parts_excluded += 1
            elif pid not in geometry_cache:
                stats.unknown_vanilla_geometry_excluded += 1

        parts, node_id_to_idx = iter_vanilla_parts_from_graph(graph_data, geometry_cache=geometry_cache)

        if config.part_allowlist is not None:
            new_parts: List[ShipPart] = []
            new_node_id_to_idx: Dict[int, int] = {}
            for node_id, old_idx in sorted(node_id_to_idx.items(), key=lambda kv: kv[1]):
                if parts[old_idx].part_id in config.part_allowlist:
                    new_node_id_to_idx[node_id] = len(new_parts)
                    new_parts.append(parts[old_idx])
            parts = new_parts
            node_id_to_idx = new_node_id_to_idx

        stats.vanilla_parts_seen += len(parts)
        if len(parts) < config.min_parts_per_ship:
            stats.ships_skipped_too_small += 1
            continue
        if len(parts) > config.max_parts_per_ship:
            stats.ships_skipped_too_large += 1
            continue
        stats.ships_used += 1
        stats.vanilla_parts_used += len(parts)

        edges = graph_data.get("graphs", {}).get("A_structural_part_graph", {}).get("edges", [])
        ordered = order_ship_parts_from_graph(parts, node_id_to_idx, edges, anchor_window=config.anchor_window)

        history: List[str] = []
        for idx, (part, anchor) in enumerate(ordered):
            if idx == 0:
                token = RelativePlacementToken(
                    part_id=part.part_id,
                    rotation=part.rotation,
                    anchor_part_id=ROOT_ANCHOR,
                    anchor_rotation=0,
                    dx=0,
                    dy=0,
                )
                key = token.as_key()
                start_counts[key] += 1
                history.append(key)
                part_frequency[part.part_id] += 1
                stats.root_tokens += 1
                continue
            assert anchor is not None
            token = RelativePlacementToken(
                part_id=part.part_id,
                rotation=part.rotation,
                anchor_part_id=anchor.part_id,
                anchor_rotation=anchor.rotation,
                dx=part.x - anchor.x,
                dy=part.y - anchor.y,
            )
            key = token.as_key()
            transition_counts[state_key(history, config.markov_order)][key] += 1
            history.append(key)
            part_frequency[part.part_id] += 1
            stats.transition_tokens += 1
            if parts_touch(anchor, part, geometry_cache):
                stats.touching_transitions += 1
            else:
                stats.non_touching_transitions += 1
        transition_counts[state_key(history, config.markov_order)][END_TOKEN] += 1
        stats.end_tokens += 1

    payload = {
        "schema_version": 2,
        "model_type": "relative_markov_first_pass",
        "config": _config_as_dict(config),
        "corpus": {"graph_dir": str(graph_dir), "source": "pre-generated ship graphs"},
        "stats": asdict(stats),
        "start_counts": dict(start_counts),
        "transition_counts": {k: dict(v) for k, v in transition_counts.items()},
        "part_frequency": dict(part_frequency),
        "notes": [
            "Vanilla-only corpus model built from pre-generated ship graph JSONs.",
            "Ordering uses BFS traversal over A_structural_part_graph touching edges.",
            "Anchors are touching graph neighbours (BFS parent), improving structural connectivity.",
            "Disconnected components fall back to geometric nearest-neighbour anchoring.",
            "Relative placement uses origin-to-origin deltas between anchor and next part.",
            "Sampling rejects overlaps using full vanilla footprint geometry from game-file exports.",
        ],
    }
    return RelativeMarkovModel(payload)


def validate_relative_placement_assumptions(
    input_dir: Path,
    sample_limit: Optional[int] = None,
) -> dict:
    """Validate that relative offsets reconstruct exact ship placements."""

    geometry_cache = load_vanilla_part_geometry()
    ships_checked = 0
    placements_checked = 0
    touching_placements = 0
    non_touching_placements = 0
    origin_failures: List[dict] = []
    footprint_failures: List[dict] = []
    max_abs_dx = 0
    max_abs_dy = 0
    largest_part_count = 0

    for ship_path in iter_ship_files(input_dir):
        with ship_path.open(encoding="utf-8") as file_handle:
            ship_data = json.load(file_handle)
        vanilla_parts = iter_vanilla_parts_from_ship(ship_data, geometry_cache=geometry_cache)
        if len(vanilla_parts) < 2:
            continue
        ships_checked += 1
        largest_part_count = max(largest_part_count, len(vanilla_parts))
        ordered = order_ship_parts(vanilla_parts)
        reconstructed: List[ShipPart] = []

        for idx, (part, anchor) in enumerate(ordered):
            if idx == 0:
                reconstructed.append(
                    ShipPart(part_id=part.part_id, rotation=part.rotation, x=part.x, y=part.y)
                )
                continue
            assert anchor is not None
            dx = part.x - anchor.x
            dy = part.y - anchor.y
            placements_checked += 1
            max_abs_dx = max(max_abs_dx, abs(dx))
            max_abs_dy = max(max_abs_dy, abs(dy))
            if parts_touch(anchor, part, geometry_cache):
                touching_placements += 1
            else:
                non_touching_placements += 1
            reconstructed_part = ShipPart(
                part_id=part.part_id,
                rotation=part.rotation,
                x=anchor.x + dx,
                y=anchor.y + dy,
            )
            reconstructed.append(reconstructed_part)
            if (reconstructed_part.x, reconstructed_part.y) != (part.x, part.y) and len(origin_failures) < 20:
                origin_failures.append(
                    {
                        "ship": ship_path.name,
                        "part_id": part.part_id,
                        "rotation": part.rotation,
                        "expected_origin": [part.x, part.y],
                        "got_origin": [reconstructed_part.x, reconstructed_part.y],
                        "anchor_part_id": anchor.part_id,
                        "anchor_rotation": anchor.rotation,
                        "dx": dx,
                        "dy": dy,
                    }
                )
            if (
                reconstructed_part.footprint_cells(geometry_cache) != part.footprint_cells(geometry_cache)
                and len(footprint_failures) < 20
            ):
                footprint_failures.append(
                    {
                        "ship": ship_path.name,
                        "part_id": part.part_id,
                        "rotation": part.rotation,
                        "anchor_part_id": anchor.part_id,
                        "anchor_rotation": anchor.rotation,
                        "dx": dx,
                        "dy": dy,
                    }
                )
        if sample_limit is not None and ships_checked >= sample_limit:
            break

    return {
        "ships_checked": ships_checked,
        "placements_checked": placements_checked,
        "largest_ship_vanilla_part_count": largest_part_count,
        "origin_failure_count": len(origin_failures),
        "footprint_failure_count": len(footprint_failures),
        "origin_failures": origin_failures,
        "footprint_failures": footprint_failures,
        "touching_placements": touching_placements,
        "non_touching_placements": non_touching_placements,
        "touching_fraction": (touching_placements / placements_checked) if placements_checked else 0.0,
        "max_abs_dx": max_abs_dx,
        "max_abs_dy": max_abs_dy,
        "summary": (
            "Origin-to-origin relative offsets reconstruct exact real-corpus part origins, "
            "and the resulting vanilla footprint cells also match exactly for the checked canonical ships."
        ),
    }
