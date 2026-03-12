"""Sampling helpers for runtime Markov ship generation."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Optional

from ship_layout.connectivity import parts_structurally_touch
from ship_layout.validation import occupied_cells_are_mirror_balanced, part_overlaps_occupied_cells
from visualizer.events import VisualizationPart

from .types import Coord, END_TOKEN, GenerationConfig, RelativePlacementToken, ShipPart, _config_as_dict

__all__ = ["WeightedSampler", "generate_ship_layout"]


def _requirements_satisfied(part_counts: Dict[str, int], requirements: dict) -> bool:
    """Return True if all (part_id -> min_count) requirements are met"""

    return all(part_counts.get(part_id, 0) >= required for part_id, required in requirements.items())


def _is_structurally_connected_to_anchor(
    candidate_part: ShipPart,
    anchor_part: ShipPart,
    geometry_cache: Dict[str, object],
) -> bool:
    """Return True when a candidate part shares a structural hull side with its anchor.

    Args:
        candidate_part: Newly sampled part placement candidate
        anchor_part: Existing placed part chosen as this token's anchor
        geometry_cache: Loaded vanilla geometry metadata used for hull checks

    Returns:
        True when candidate and anchor share at least one attachable side segment
    """

    return parts_structurally_touch(candidate_part, anchor_part, geometry_cache)


def _visualization_part_from_ship_part(part: ShipPart) -> VisualizationPart:
    """Convert one ship part into the shared visualization shape."""

    return VisualizationPart(
        part_id=part.part_id,
        rotation=part.rotation,
        x=part.x,
        y=part.y,
        flip_x=part.flip_x,
        flip_y=part.flip_y,
    )


def _emit_part_placed(event_sink, part: ShipPart, *, message: str, metadata: dict) -> None:
    """Record one accepted placement when visualization is enabled."""

    if event_sink is None:
        return
    event_sink.part_placed(
        part=_visualization_part_from_ship_part(part),
        message=message,
        metadata=metadata,
    )


def _emit_attempt_rejected(
    event_sink,
    *,
    reason: str,
    part: ShipPart | None = None,
    message: str,
    metadata: dict,
) -> None:
    """Record one rejected attempt when visualization is enabled."""

    if event_sink is None:
        return
    event_sink.attempt_rejected(
        reason=reason,
        part=_visualization_part_from_ship_part(part) if part is not None else None,
        message=message,
        metadata=metadata,
    )


def _resolve_transition_options_for_history(model, history: List[str]) -> Optional[Dict[str, int]]:
    """Resolve transition options for the current history with order fallback.

    Args:
        model: Loaded `RelativeMarkovModel` that stores transition counters
        history: Token-key history in emission order

    Returns:
        Transition counter for the best available state, or None when no state
        has any transition options
    """

    state_options = model.transition_counts.get(model._state_key(history, model.order))
    if state_options:
        return state_options
    if model.order > 0:
        for fallback_order in range(model.order - 1, -1, -1):
            fallback_options = model.transition_counts.get(model._state_key(history, fallback_order))
            if fallback_options:
                return fallback_options
    return None


def _filtered_counter(counter: Dict[str, int], allowed_keys: set[str]) -> Dict[str, int]:
    """Build a weighted counter subset that keeps only positive allowed keys."""

    return {key: weight for key, weight in counter.items() if weight > 0 and key in allowed_keys}


def _has_seed_viable_transition(
    model,
    root_key: str,
    *,
    available_anchor_signatures: set[tuple[str, int]],
    allowlist,
) -> bool:
    """Return True when seeded bootstrap root can emit at least one viable token.

    Args:
        model: Loaded `RelativeMarkovModel` with transition counts
        root_key: Candidate synthetic seeded-history root token key
        available_anchor_signatures: Anchor signatures currently present in the seeded ship
        allowlist: Optional allowed part-id set from generation config

    Returns:
        True when the root state's transition options include at least one
        non-END token that can reference an available seed anchor signature and
        passes lightweight part-id/geometry validation
    """

    transition_options = _resolve_transition_options_for_history(model, [root_key])
    if not transition_options:
        return False

    for token_key in transition_options:
        if token_key == END_TOKEN:
            continue
        token = RelativePlacementToken.from_key(token_key)
        anchor_signature = (token.anchor_part_id, token.anchor_rotation)
        if anchor_signature not in available_anchor_signatures:
            continue
        if allowlist is not None and token.part_id not in allowlist:
            continue
        if (
            token.part_id not in model.geometry_cache
            or token.rotation not in model.geometry_cache[token.part_id].rotations
        ):
            continue
        return True

    return False


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


def generate_ship_layout(model, config: GenerationConfig, *, seed_parts=None, event_sink=None) -> dict:
    """Generate a ship layout using a loaded relative-placement Markov model

    Args:
        model: `RelativeMarkovModel` instance that owns payload and geometry state
        config: Generation settings
        seed_parts: Optional list of pre-placed seed parts as dicts or `ShipPart`
        event_sink: Optional visualization recorder for generation attempts

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
    rejected_structural = 0
    rejected_requirements = 0
    stop_reason = "unknown"

    allowlist = config.part_allowlist
    rejected_allowlist = 0
    mirror_mode = config.mirror_symmetry

    if mirror_mode:
        from .symmetry import (
            is_anchor_eligible_mirror_primary as _is_anchor_eligible_mirror_primary,
            mirror_part as _mirror_part,
            primary_root_x as _primary_root_x,
        )

    part_counts: Dict[str, int] = {}
    seed_stats: Optional[dict] = None
    seeded = seed_parts is not None and len(seed_parts) > 0

    if event_sink is not None:
        event_sink.sample_started(config=_config_as_dict(config), seeded=seeded)

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
                _emit_attempt_rejected(
                    event_sink,
                    reason="seed_geometry",
                    part=seed_part,
                    message="Seed part skipped: missing vanilla geometry",
                    metadata={"is_seed": True},
                )
                continue
            if allowlist is not None and seed_part.part_id not in allowlist:
                seed_skipped_allowlist += 1
                _emit_attempt_rejected(
                    event_sink,
                    reason="seed_allowlist",
                    part=seed_part,
                    message="Seed part skipped: not in allowlist",
                    metadata={"is_seed": True},
                )
                continue

            seed_cells = seed_part.footprint_cells(model.geometry_cache)
            if part_overlaps_occupied_cells(seed_part, model.geometry_cache, occupied_cells):
                seed_skipped_overlap += 1
                _emit_attempt_rejected(
                    event_sink,
                    reason="seed_overlap",
                    part=seed_part,
                    message="Seed part skipped: overlaps existing seed placement",
                    metadata={"is_seed": True},
                )
                continue

            placed_index = len(placed)
            placed.append(seed_part)
            occupied_cells.update(seed_cells)
            part_counts[seed_part.part_id] = part_counts.get(seed_part.part_id, 0) + 1

            is_primary_anchor = True
            if mirror_mode:
                is_primary_anchor = _is_anchor_eligible_mirror_primary(seed_part, model.geometry_cache)
            if is_primary_anchor:
                primary_indices.append(placed_index)

            placement_trace.append(
                {
                    "token": None,
                    "anchor_index": None,
                    "placed_index": placed_index,
                    "world_origin": [seed_part.x, seed_part.y],
                    "is_seed": True,
                    "is_mirror": not is_primary_anchor,
                }
            )
            _emit_part_placed(
                event_sink,
                seed_part,
                message="Accepted seed placement",
                metadata={
                    "placed_index": placed_index,
                    "is_seed": True,
                    "is_mirror": not is_primary_anchor,
                },
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

        if mirror_mode and not occupied_cells_are_mirror_balanced(occupied_cells):
            raise RuntimeError(
                "seed provided but occupied footprint is not mirror-symmetric around x = -0.5; "
                "mirror mode accepts asymmetric part placement only when occupied cells match "
                "on both sides of the mirror axis."
            )

        if not primary_indices:
            raise RuntimeError(
                "seed provided but no mirror-primary anchor candidates were found; "
                "mirror mode requires at least one left-side or self-mirroring centerline "
                "part to serve as a Markov anchor."
            )

        if mirror_mode:
            anchor_indexes_for_seed = list(primary_indices)
        else:
            anchor_indexes_for_seed = list(range(len(placed)))

        available_anchor_signatures = {
            (placed[idx].part_id, placed[idx].rotation) for idx in anchor_indexes_for_seed
        }
        virtual_root_key = None
        virtual_root_selection = "unselected"

        valid_start_keys: set[str] = set()
        compatible_start_keys: set[str] = set()
        viable_start_keys: set[str] = set()

        # Build root-key buckets once so seeded startup can sample from the
        # most compatible distribution instead of repeatedly retrying the full
        # root counter and risking incompatible anchor signatures.
        for start_key in model.start_counts:
            start_token = RelativePlacementToken.from_key(start_key)
            if allowlist is not None and start_token.part_id not in allowlist:
                continue
            if (
                start_token.part_id not in model.geometry_cache
                or start_token.rotation not in model.geometry_cache[start_token.part_id].rotations
            ):
                continue
            valid_start_keys.add(start_key)
            if (start_token.part_id, start_token.rotation) in available_anchor_signatures:
                compatible_start_keys.add(start_key)
            if _has_seed_viable_transition(
                model,
                start_key,
                available_anchor_signatures=available_anchor_signatures,
                allowlist=allowlist,
            ):
                viable_start_keys.add(start_key)

        preferred_buckets = [
            ("compatible_with_viable_transition", compatible_start_keys & viable_start_keys),
            ("viable_transition_only", viable_start_keys),
            ("compatible_signature_only", compatible_start_keys),
            ("valid_start_only", valid_start_keys),
        ]
        for selection_name, candidate_bucket in preferred_buckets:
            if not candidate_bucket:
                continue
            weighted_bucket = _filtered_counter(model.start_counts, candidate_bucket)
            if not weighted_bucket:
                continue
            virtual_root_key = WeightedSampler.sample(weighted_bucket, rng)
            virtual_root_selection = selection_name
            break

        if virtual_root_key is None:
            raise RuntimeError("could not sample a virtual root token for seeded generation")
        history = [virtual_root_key]
        if seed_stats is not None:
            chosen_virtual_root = RelativePlacementToken.from_key(virtual_root_key)
            seed_stats["virtual_root"] = {
                "selection": virtual_root_selection,
                "part_id": chosen_virtual_root.part_id,
                "rotation": chosen_virtual_root.rotation,
                "available_anchor_signatures": len(available_anchor_signatures),
            }
    else:
        root_key = None
        root = None
        root_attempts = max(64, len(model.start_counts) // 100)
        for _ in range(root_attempts):
            candidate_root_key = WeightedSampler.sample(model.start_counts, rng)
            candidate_root = RelativePlacementToken.from_key(candidate_root_key)
            if allowlist is not None and candidate_root.part_id not in allowlist:
                rejected_allowlist += 1
                _emit_attempt_rejected(
                    event_sink,
                    reason="root_allowlist",
                    message="Root token rejected by allowlist",
                    metadata={"token": candidate_root.to_dict()},
                )
                continue
            if (
                candidate_root.part_id in model.geometry_cache
                and candidate_root.rotation in model.geometry_cache[candidate_root.part_id].rotations
            ):
                root_key = candidate_root_key
                root = candidate_root
                break
            _emit_attempt_rejected(
                event_sink,
                reason="root_geometry",
                message="Root token rejected: missing vanilla geometry",
                metadata={"token": candidate_root.to_dict()},
            )
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
            _emit_part_placed(
                event_sink,
                root_part,
                message="Accepted root placement",
                metadata={"placed_index": root_primary_idx, "anchor_index": None, "is_mirror": False},
            )
            if mirror_root is not None:
                mirror_cells = mirror_root.footprint_cells(model.geometry_cache)
                if mirror_cells != root_cells:
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
                    _emit_part_placed(
                        event_sink,
                        mirror_root,
                        message="Accepted mirrored root placement",
                        metadata={
                            "placed_index": mirror_idx,
                            "anchor_index": None,
                            "is_mirror": True,
                            "mirror_of": root_primary_idx,
                        },
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
            _emit_part_placed(
                event_sink,
                root_part,
                message="Accepted root placement",
                metadata={"placed_index": 0, "anchor_index": None, "is_mirror": False},
            )
        history.append(root_key)

    while len(placed) < config.max_parts and attempts < config.max_attempts:
        options = _resolve_transition_options_for_history(model, history)
        if not options:
            stop_reason = "no_transition_for_state"
            break

        # Step-local mutable options avoid wasting retries on the exact same
        # token after it has already been proven invalid for the current state.
        step_options = {token_key: weight for token_key, weight in options.items() if weight > 0}
        token_key = None
        all_end_tokens_by_req = True
        for _ in range(config.max_resample_per_step):
            if not step_options:
                break
            attempts += 1
            candidate_key = WeightedSampler.sample(step_options, rng)
            if candidate_key == END_TOKEN:
                if config.part_requirements is not None and not _requirements_satisfied(
                    part_counts, config.part_requirements
                ):
                    rejected_requirements += 1
                    _emit_attempt_rejected(
                        event_sink,
                        reason="requirements",
                        message="END token rejected until requirements are satisfied",
                        metadata={"attempts": attempts},
                    )
                    step_options.pop(candidate_key, None)
                    continue
                token_key = END_TOKEN
                break

            all_end_tokens_by_req = False
            token = RelativePlacementToken.from_key(candidate_key)
            if allowlist is not None and token.part_id not in allowlist:
                rejected_allowlist += 1
                _emit_attempt_rejected(
                    event_sink,
                    reason="allowlist",
                    message="Candidate rejected by allowlist",
                    metadata={"attempts": attempts, "token": token.to_dict()},
                )
                step_options.pop(candidate_key, None)
                continue
            if (
                token.part_id not in model.geometry_cache
                or token.rotation not in model.geometry_cache[token.part_id].rotations
            ):
                _emit_attempt_rejected(
                    event_sink,
                    reason="geometry",
                    message="Candidate rejected: missing vanilla geometry",
                    metadata={"attempts": attempts, "token": token.to_dict()},
                )
                step_options.pop(candidate_key, None)
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
                _emit_attempt_rejected(
                    event_sink,
                    reason="missing_anchor",
                    message="Candidate rejected: no matching anchor was available",
                    metadata={"attempts": attempts, "token": token.to_dict()},
                )
                step_options.pop(candidate_key, None)
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

                # Tokens are trained from anchor-relative deltas but still need
                # runtime geometry validation because wedge/overhang placements
                # can be adjacent without a true structural hull connection.
                if not _is_structurally_connected_to_anchor(
                    candidate,
                    anchor_part,
                    model.geometry_cache,
                ):
                    rejected_structural += 1
                    _emit_attempt_rejected(
                        event_sink,
                        reason="structural",
                        part=candidate,
                        message="Candidate rejected: no structural contact with anchor",
                        metadata={
                            "attempts": attempts,
                            "token": token.to_dict(),
                            "anchor_index": anchor_idx,
                        },
                    )
                    continue

                if mirror_mode:
                    if not model._within_primary_bounds(candidate, config):
                        rejected_bounds += 1
                        _emit_attempt_rejected(
                            event_sink,
                            reason="bounds",
                            part=candidate,
                            message="Candidate rejected: primary placement left configured bounds",
                            metadata={
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "anchor_index": anchor_idx,
                            },
                        )
                        continue
                    if part_overlaps_occupied_cells(candidate, model.geometry_cache, occupied_cells):
                        rejected_overlap += 1
                        _emit_attempt_rejected(
                            event_sink,
                            reason="overlap",
                            part=candidate,
                            message="Candidate rejected: overlaps existing ship cells",
                            metadata={
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "anchor_index": anchor_idx,
                            },
                        )
                        continue
                    mirror_candidate = _mirror_part(candidate, model.geometry_cache)
                    if mirror_candidate is None:
                        rejected_mirror += 1
                        _emit_attempt_rejected(
                            event_sink,
                            reason="mirror",
                            part=candidate,
                            message="Candidate rejected: could not construct mirrored placement",
                            metadata={
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "anchor_index": anchor_idx,
                            },
                        )
                        continue
                    mirror_cells = mirror_candidate.footprint_cells(model.geometry_cache)
                    if part_overlaps_occupied_cells(mirror_candidate, model.geometry_cache, occupied_cells):
                        rejected_mirror += 1
                        _emit_attempt_rejected(
                            event_sink,
                            reason="mirror_overlap",
                            part=mirror_candidate,
                            message="Candidate rejected: mirrored placement overlaps existing ship cells",
                            metadata={
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "anchor_index": anchor_idx,
                            },
                        )
                        continue
                    if not model._within_mirror_bounds(mirror_candidate, config):
                        rejected_mirror += 1
                        _emit_attempt_rejected(
                            event_sink,
                            reason="mirror_bounds",
                            part=mirror_candidate,
                            message="Candidate rejected: mirrored placement left configured bounds",
                            metadata={
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "anchor_index": anchor_idx,
                            },
                        )
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
                    _emit_part_placed(
                        event_sink,
                        candidate,
                        message="Accepted primary placement",
                        metadata={
                            "placed_index": primary_idx,
                            "anchor_index": anchor_idx,
                            "attempts": attempts,
                            "token": token.to_dict(),
                            "is_mirror": False,
                        },
                    )

                    # If mirroring maps this placement back onto the same occupied
                    # cells, keep only one centered part instead of creating an
                    # overlapping duplicate "mirror" copy.
                    if mirror_cells != candidate_cells:
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
                        _emit_part_placed(
                            event_sink,
                            mirror_candidate,
                            message="Accepted mirrored placement",
                            metadata={
                                "placed_index": mirror_idx,
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "is_mirror": True,
                                "mirror_of": primary_idx,
                            },
                        )
                else:
                    if part_overlaps_occupied_cells(candidate, model.geometry_cache, occupied_cells):
                        rejected_overlap += 1
                        _emit_attempt_rejected(
                            event_sink,
                            reason="overlap",
                            part=candidate,
                            message="Candidate rejected: overlaps existing ship cells",
                            metadata={
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "anchor_index": anchor_idx,
                            },
                        )
                        continue
                    if not model._placement_within_bounds(candidate, config):
                        rejected_bounds += 1
                        _emit_attempt_rejected(
                            event_sink,
                            reason="bounds",
                            part=candidate,
                            message="Candidate rejected: left configured bounds",
                            metadata={
                                "attempts": attempts,
                                "token": token.to_dict(),
                                "anchor_index": anchor_idx,
                            },
                        )
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
                    _emit_part_placed(
                        event_sink,
                        candidate,
                        message="Accepted placement",
                        metadata={
                            "placed_index": len(placed) - 1,
                            "anchor_index": anchor_idx,
                            "attempts": attempts,
                            "token": token.to_dict(),
                            "is_mirror": False,
                        },
                    )

                accepted = True
                token_key = candidate_key
                break
            if accepted:
                break
            step_options.pop(candidate_key, None)

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
            "structural": rejected_structural,
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
        "Anchor deltas are additionally validated with structural hull-side contact checks.",
        "Doors and accessibility cleanup are intentionally deferred to later passes.",
    ]
    if mirror_mode:
        notes.append(
            "Mirror symmetry: left-right across axis x = -0.5 (between columns -1 and 0). "
            "Primary anchors can be left-side placements or self-mirroring centerline straddlers. "
            "Mirrored companions are emitted only when the reflected footprint is distinct."
        )
    if config.part_requirements:
        notes.append(
            "Part requirements use total-ship-count semantics (primary + mirror parts both count). "
            "END_TOKEN is suppressed until all requirements are satisfied or max_attempts is reached."
        )
    if seeded:
        notes.append(
            "Seeded generation: existing ship parts were pre-placed before Markov sampling began. "
            "Seeded startup chooses a virtual root token that is compatible with available seed anchors."
        )

    payload = {
        "generator": "relative_markov_first_pass",
        "config": _config_as_dict(config),
        "stats": stats,
        "parts": [asdict(part) for part in placed],
        "placement_trace": placement_trace,
        "notes": notes,
    }
    if event_sink is not None:
        event_sink.sample_finished(
            stats=stats,
            stop_reason=stop_reason,
            message="Generation finished",
        )
    return payload
