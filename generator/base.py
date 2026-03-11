"""Abstract interfaces for generator backends."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["GeneratorBackend", "add_visualization_arguments"]


def add_visualization_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared visualization flags for generator backends."""

    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Render an MP4 visualizing generation attempts and accepted placements",
    )
    parser.add_argument(
        "--visualization-fps",
        type=int,
        default=24,
        metavar="FPS",
        help="Frame rate for the visualization MP4 (default: 24)",
    )
    parser.add_argument(
        "--icons-root",
        type=Path,
        default=None,
        help=(
            "Optional path to a Terran part-icon root "
            "(for example Data/ships/terran)"
        ),
    )
    parser.add_argument(
        "--game-root",
        type=Path,
        default=None,
        help=(
            "Optional path to a local Cosmoteer install root. "
            "Used to resolve Data/ships/terran automatically."
        ),
    )


class GeneratorBackend(ABC):
    """Backend contract for runtime ship generation integrations."""

    name: str

    @abstractmethod
    def register_generate_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the backend-specific generator parser."""

    @abstractmethod
    def run_generate(self, args: argparse.Namespace) -> int:
        """Execute a backend-specific generation request."""
