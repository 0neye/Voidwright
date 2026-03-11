"""Abstract interfaces for generator backends."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

__all__ = ["GeneratorBackend"]


class GeneratorBackend(ABC):
    """Backend contract for runtime ship generation integrations."""

    name: str

    @abstractmethod
    def register_generate_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the backend-specific generator parser."""

    @abstractmethod
    def run_generate(self, args: argparse.Namespace) -> int:
        """Execute a backend-specific generation request."""
