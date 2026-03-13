"""Abstract interfaces for graph expansion backends."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

__all__ = ["ExpansionBackend"]


class ExpansionBackend(ABC):
    """Backend contract for graph expansion module integrations."""

    name: str

    @abstractmethod
    def register_expand_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the backend-specific expand parser."""

    @abstractmethod
    def run_expand(self, args: argparse.Namespace) -> int:
        """Execute a backend-specific expand request."""
