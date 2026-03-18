"""Abstract interface for static ship visualization backends."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from visualizer.icons import PartIconLibrary

__all__ = ["StaticVisualizationBackend"]


class StaticVisualizationBackend(ABC):
    """Contract for static-image ship visualization backends.

    The CLI instantiates one backend per render subcommand. For each input
    ship the CLI calls :meth:`render_ship` with a fully-expanded graph payload
    and a pre-built icon library. All heavy initialization (e.g. icon loading)
    should be deferred to the CLI layer, not repeated per ship.
    """

    #: Short hyphenated identifier used as the CLI subcommand name.
    #: Example: ``"spatial-zones"``.
    name: str

    #: Default output directory path (relative to project root).
    #: Example: ``"out/visualizations/spatial-zones"``.
    default_output_dir: str

    @abstractmethod
    def register_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to *parser*.

        The shared arguments ``--input``, ``--output-dir``, ``--icons-root``,
        and ``--game-root`` are registered by the CLI layer and must not be
        added here.
        """

    @abstractmethod
    def render_ship(
        self,
        ship_name: str,
        expanded_data: dict[str, Any],
        flip_map: dict[tuple[int, int], tuple[bool, bool]],
        output_dir: Path,
        icon_library: PartIconLibrary,
        args: argparse.Namespace,
    ) -> Path:
        """Render one ship to a static PNG and return the output file path.

        Parameters
        ----------
        ship_name:
            Display name derived from the input filename.
        expanded_data:
            Fully enriched graph dict. Contains
            ``graphs["A_structural_part_graph"]`` and
            ``graphs["X_expansion_structural"]``.
        flip_map:
            ``(x2, y2) -> (flip_x, flip_y)`` map for icon rendering.
        output_dir:
            Directory where the PNG should be saved. Already created by CLI.
        icon_library:
            Pre-built icon library (blueprints preferred). One instance shared
            across all ships in a single CLI invocation.
        args:
            Parsed CLI arguments (includes any backend-specific args).
        """
