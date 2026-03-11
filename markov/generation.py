"""Sampling helpers for runtime Markov ship generation."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Optional

from .types import Coord, END_TOKEN, GenerationConfig, RelativePlacementToken, ShipPart, _config_as_dict

__all__ = ["WeightedSampler", "generate_ship_layout"]


def _requirements_satisfied(part_counts: Dict[str, int], requirements: dict) -> bool:
    """Return True if all (part_id -> min_count) requirements are met"""

    return all(part_counts.get(part_id, 0) >= required for part_id, required in requirements.items())


class WeightedSampler:
    """Sample weighted counters without converting them to expanded lists."""

    @staticmethod
    def sample(counter: Dict[str, int], rng: random.Random) -> str:
        """Sample one key from a counter using cumulative weights"""

        total_weight = sum(counter.values())
        if total_weight <= 0:
            raise ValueError("cannot sample from empty counter")
        target_weight = rng.uniform(0, total_weight)
        accumulated_weight = 0.0
        last_key = None
        for key, weight in counter.items():
            accumulated_weight += weight
            last_key = key
            if target_weight <= accumulated_weight:
                return key
        return last_key  # pragma: no cover


def generate_ship_layout(model, config: GenerationConfig, *, seed_parts=None) -> dict:
    """Generate a ship layout using a loaded relative-placement Markov model

    Args:
        model: `RelativeMarkovModel` instance that owns payload and geometry state
        config: Generation settings
        seed_parts: Optional list of pre-placed seed parts as dicts or `ShipPart`

    Returns:
        Generation payload containing `parts`, `stats`, and `placement_trace`
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

        # First place every valid seed part exactly once before normal sampling
        for raw_part in seed_parts:
            if isinstance(raw_part, dict):
                seed_part = ShipPart(
                    part_id=raw_part["part_id"],
                    rotation=int(raw_part.get("rotation", 0)) % 4,
                    x=int(raw_part["x"]),
                    y=int(raw_part["y"]),
                    flip_x=bool(raw_part.get("flip_x", raw_part.get("FlipX", False))),
                    flip_y=bool(raw_part.get("flip_y", raw_part.get("FlipY", False))),
                )
            else:
                seed_part = raw_part

            if (
                seed_part.part_id not in model.geometry_cache
                or seed_part.rotation not in model.geometry_cache[seed_part.part_id].rotations
            ):
                seed_skipped_geometry += 1
                continue
            if allowlist is not None and seed_part.part_id not in allowlist:
                seed_skipped_allowlist += 1
                continue

            seed_cells = seed_part.footprint_cells(model.geometry_cache)
            if seed_cells & occupied_cells:
                seed_skipped_overlap += 1
                continue

            placed_index = len(placed)
            placed.append(seed_part)
            occupied_cells.update(seed_cells)
            part_counts[seed_part.part_id] = part_counts.get(seed_part.part_id, 0) + 1

            is_primary = True
            if mirror_mode:
                is_primary = _is_primary_placement(seed_part, model.geometry_cache)
            if is_primary:
                primary_indices.append(placed_index)

            placement_trace.append(
                {
                    "token": None,
                    "anchor_index": None,
                    "placed_index": placed_index,
                    "world_origin": [seed_part.x, seed_part.y],
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

        seed_part_ids = frozenset(placed[idx].part_id for idx in primary_indices)
        virtual_root_key = None
        virtual_root_attempts = max(256, len(model.start_counts) // 20)

        # Prefer virtual roots already represented in the seed, then fall back
        for _ in range(virtual_root_attempts):
            candidate_key = WeightedSampler.sample(model.start_counts, rng)
            candidate = RelativePlacementToken.from_key(candidate_key)
            if allowlist is not None and candidate.part_id not in allowlist:
                continue
            if (
                candidate.part_id not in model.geometry_cache
                or candidate.rotation not in model.geometry_cache[candidate.part_id].rotations
            ):
                continue
            if candidate.part_id in seed_part_ids:
                virtual_root_key = candidate_key
                break
        if virtual_root_key is None:
            for _ in range(virtual_root_attempts):
                candidate_key = WeightedSampler.sample(model.start_counts, rng)
                candidate = RelativePlacementToken.from_key(candidate_key)
                if allowlist is not None and candidate.part_id not in allowlist:
                    continue
                if (
                    candidate.part_id in model.geometry_cache
                    and candidate.rotation in model.geometry_cache[candidate.part_id].rotations
                ):
                    virtual_root_key = candidate_key
                    break
        if virtual_root_key is None:
            raise RuntimeError("could not sample a virtual root token for seeded generation")
        history = [virtual_root_key]
    else:
        root_key = None
        root = None
        root_attempts = max(64, len(model.start_counts) // 100)
        for _ in range(root_attempts):
            candidate_root_key = WeightedSampler.sample(model.start_counts, rng)
            candidate_root = RelativePlacementToken.from_key(candidate_root_key)
            if allowlist is not None and candidate_root.part_id not in allowlist:
                rejected_allowlist += 1
                continue
            if (
                candidate_root.part_id in model.geometry_cache
                and candidate_root.rotation in model.geometry_cache[candidate_root.part_id].rotations
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
            root_x = _primary_root_x(root.part_id, root.rotation, model.geometry_cache)
            root_part = ShipPart(part_id=root.part_id, rotation=root.rotation, x=root_x, y=0)
            root_cells = root_part.footprint_cells(model.geometry_cache)
            mirror_root = _mirror_part(root_part, model.geometry_cache)
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
                mirror_cells = mirror_root.footprint_cells(model.geometry_cache)
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
            root_cells = root_part.footprint_cells(model.geometry_cache)
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
        state = model._state_key(history, model.order)
        options = model.transition_counts.get(state)
        if not options and model.order > 0:
            for fallback_order in range(model.order - 1, -1, -1):
                options = model.transition_counts.get(model._state_key(history, fallback_order))
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
                if config.part_requirements is not None and not _requirements_satisfied(
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
                token.part_id not in model.geometry_cache
                or token.rotation not in model.geometry_cache[token.part_id].rotations
            ):
                continue

            if mirror_mode:
                anchor_indexes = [
                    idx
                    for idx in primary_indices
                    if placed[idx].part_id == token.anchor_part_id
                    and placed[idx].rotation == token.anchor_rotation
                ]
            else:
                anchor_indexes = [
                    idx
                    for idx, part in enumerate(placed)
                    if part.part_id == token.anchor_part_id and part.rotation == token.anchor_rotation
                ]
            anchor_indexes.sort(reverse=True)
            if not anchor_indexes:
                rejected_missing_anchor += 1
                continue

            accepted = False
            for anchor_idx in anchor_indexes:
                anchor_part = placed[anchor_idx]
                candidate = ShipPart(
                    part_id=token.part_id,
                    rotation=token.rotation,
                    x=anchor_part.x + token.dx,
                    y=anchor_part.y + token.dy,
                )
                candidate_cells = candidate.footprint_cells(model.geometry_cache)

                if mirror_mode:
                    if not model._within_primary_bounds(candidate, config):
                        rejected_bounds += 1
                        continue
                    if candidate_cells & occupied_cells:
                        rejected_overlap += 1
                        continue
                    mirror_candidate = _mirror_part(candidate, model.geometry_cache)
                    if mirror_candidate is None:
                        rejected_mirror += 1
                        continue
                    mirror_cells = mirror_candidate.footprint_cells(model.geometry_cache)
                    if mirror_cells & occupied_cells:
                        rejected_mirror += 1
                        continue
                    if not model._within_mirror_bounds(mirror_candidate, config):
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
                            "anchor_index": anchor_idx,
                            "placed_index": primary_idx,
                            "world_origin": [candidate.x, candidate.y],
                            "is_mirror": False,
                        }
                    )

                    mirror_idx = len(placed)
                    placed.append(mirror_candidate)
                    occupied_cells.update(mirror_cells)
                    part_counts[mirror_candidate.part_id] = (
                        part_counts.get(mirror_candidate.part_id, 0) + 1
                    )
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
                    if not model._placement_within_bounds(candidate, config):
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
                            "anchor_index": anchor_idx,
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

    if config.part_requirements and not _requirements_satisfied(part_counts, config.part_requirements):
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
            part_id: {
                "required": required_count,
                "actual": part_counts.get(part_id, 0),
                "satisfied": part_counts.get(part_id, 0) >= required_count,
            }
            for part_id, required_count in config.part_requirements.items()
        }
        stats["requirements"] = {
            "satisfied": _requirements_satisfied(part_counts, config.part_requirements),
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
