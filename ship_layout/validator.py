"""Stateful placement validator for ship generation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .connectivity import parts_structurally_touch
from .geometry import footprint_cells
from .symmetry import mirror_part
from .types import PlacedPart
from .validation import (
    footprint_is_mirror_balanced,
    is_anchor_eligible_mirror_primary,
    is_mirror_placement,
    is_primary_placement,
    occupied_cells_are_mirror_balanced,
    placement_within_bounds,
)

__all__ = ["PlacementValidator", "ValidationResult"]


@dataclass
class ValidationResult:
    """Result of a single placement validation check.

    ``accepted`` is True when the candidate passed all checks.
    ``rejection`` names the first failing check (None when accepted).

    Rejection vocabulary:
        "geometry_unknown"   — part_id or rotation absent from geometry cache
        "allowlist"          — part_id not in configured allowlist
        "connectivity"       — candidate and anchor share no structural hull side
        "overlap"            — candidate footprint overlaps occupied cells
        "bounds"             — candidate falls outside configured bounds
        "mirror_construction" — mirrored placement could not be constructed
        "mirror_overlap"     — mirrored footprint overlaps occupied cells
        "mirror_bounds"      — mirrored placement falls outside configured bounds

    ``mirror_companion`` is the computed mirrored PlacedPart when mirror mode is
    active and the result is accepted.  It is None when the candidate is
    self-mirroring (companion would duplicate the primary) or when not in mirror
    mode.

    ``primary_cells`` and ``companion_cells`` are the pre-computed footprint sets
    returned to avoid redundant geometry lookups in the caller.
    """

    accepted: bool
    rejection: Optional[str]
    mirror_companion: Optional[PlacedPart]
    primary_cells: Optional[frozenset]
    companion_cells: Optional[frozenset]


class PlacementValidator:
    """Stateful validator for ship part placements.

    Initialized once per generation run with geometry, bounds, and optional
    allowlist/requirements.  Callers invoke ``validate_candidate`` per anchor
    attempt and ``validate_seed_part`` for pre-placed seed parts.
    """

    def __init__(
        self,
        geometry_cache: Dict[str, object],
        *,
        min_x: int,
        max_x: int,
        min_y: int,
        max_y: int,
        mirror_mode: bool = False,
        part_allowlist=None,
        part_requirements=None,
    ):
        self._geometry_cache = geometry_cache
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._mirror_mode = mirror_mode
        self._allowlist = part_allowlist
        self._requirements = part_requirements

    def _reject(self, reason: str) -> ValidationResult:
        return ValidationResult(
            accepted=False,
            rejection=reason,
            mirror_companion=None,
            primary_cells=None,
            companion_cells=None,
        )

    def validate_candidate(
        self,
        candidate,
        anchor,
        occupied_cells,
    ) -> ValidationResult:
        """Validate one placement candidate against an anchor and current ship state.

        Args:
            candidate: Candidate part placement (ShipPart or PlacedPart or dict)
            anchor: Anchor part placement this candidate attaches to
            occupied_cells: Set of currently occupied world cells

        Returns:
            ValidationResult with accepted flag, optional rejection reason, and
            pre-computed footprint cell sets.
        """

        placed = PlacedPart.from_object(candidate)

        # 1. Geometry unknown
        geom = self._geometry_cache.get(placed.part_id)
        if geom is None or placed.rotation not in geom.rotations:
            return self._reject("geometry_unknown")

        # 2. Allowlist
        if self._allowlist is not None and placed.part_id not in self._allowlist:
            return self._reject("allowlist")

        # 3. Connectivity
        if not parts_structurally_touch(placed, anchor, self._geometry_cache):
            return self._reject("connectivity")

        primary_cells = footprint_cells(placed, self._geometry_cache)

        if self._mirror_mode:
            # 4. Primary bounds: must be within global bounds AND be left-side or self-mirroring
            if not placement_within_bounds(
                placed,
                self._geometry_cache,
                min_x=self._min_x,
                max_x=self._max_x,
                min_y=self._min_y,
                max_y=self._max_y,
            ) or not (
                is_primary_placement(placed, self._geometry_cache)
                or footprint_is_mirror_balanced(placed, self._geometry_cache)
            ):
                return self._reject("bounds")

            # 5. Primary overlap
            if primary_cells & occupied_cells:
                return self._reject("overlap")

            # 6. Mirror construction
            mirror_companion = mirror_part(placed, self._geometry_cache)
            if mirror_companion is None:
                return self._reject("mirror_construction")

            companion_cells = footprint_cells(mirror_companion, self._geometry_cache)

            # 7. Mirror overlap
            if companion_cells & occupied_cells:
                return self._reject("mirror_overlap")

            # 8. Mirror bounds: must be within global bounds AND be right-side or self-mirroring
            if not placement_within_bounds(
                mirror_companion,
                self._geometry_cache,
                min_x=self._min_x,
                max_x=self._max_x,
                min_y=self._min_y,
                max_y=self._max_y,
            ) or not (
                is_mirror_placement(mirror_companion, self._geometry_cache)
                or footprint_is_mirror_balanced(mirror_companion, self._geometry_cache)
            ):
                return self._reject("mirror_bounds")

            # 9. Collapse check: if the mirror maps onto the same cells, emit only one part
            if companion_cells == primary_cells:
                mirror_companion = None
                companion_cells = None

            return ValidationResult(
                accepted=True,
                rejection=None,
                mirror_companion=mirror_companion,
                primary_cells=primary_cells,
                companion_cells=companion_cells,
            )

        else:
            # Non-mirror mode

            # 4. Overlap
            if primary_cells & occupied_cells:
                return self._reject("overlap")

            # 5. Bounds
            if not placement_within_bounds(
                placed,
                self._geometry_cache,
                min_x=self._min_x,
                max_x=self._max_x,
                min_y=self._min_y,
                max_y=self._max_y,
            ):
                return self._reject("bounds")

            return ValidationResult(
                accepted=True,
                rejection=None,
                mirror_companion=None,
                primary_cells=primary_cells,
                companion_cells=None,
            )

    def validate_seed_part(
        self,
        candidate,
        occupied_cells,
    ) -> ValidationResult:
        """Light validation for seed parts: geometry + allowlist + overlap only.

        Connectivity and bounds are skipped — seeds are trusted to provide
        valid starting coordinates.
        """

        placed = PlacedPart.from_object(candidate)

        geom = self._geometry_cache.get(placed.part_id)
        if geom is None or placed.rotation not in geom.rotations:
            return self._reject("geometry_unknown")

        if self._allowlist is not None and placed.part_id not in self._allowlist:
            return self._reject("allowlist")

        primary_cells = footprint_cells(placed, self._geometry_cache)
        if primary_cells & occupied_cells:
            return self._reject("overlap")

        return ValidationResult(
            accepted=True,
            rejection=None,
            mirror_companion=None,
            primary_cells=primary_cells,
            companion_cells=None,
        )

    def seed_state_mirror_valid(self, occupied_cells) -> bool:
        """Return True when the seeded ship state satisfies mirror symmetry."""

        return not self._mirror_mode or occupied_cells_are_mirror_balanced(occupied_cells)

    def has_mirror_eligible_anchor(self, placed_parts) -> bool:
        """Return True when any placed part can serve as a mirror-mode primary anchor."""

        return any(
            is_anchor_eligible_mirror_primary(p, self._geometry_cache)
            for p in placed_parts
        )

    def requirements_satisfied(self, part_counts: dict) -> bool:
        """Return True when all part requirements are met, or no requirements are set."""

        if self._requirements is None:
            return True
        return all(
            part_counts.get(pid, 0) >= req
            for pid, req in self._requirements.items()
        )
