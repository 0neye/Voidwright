"""Abstract interfaces for training backends."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

__all__ = ["TrainingBackend"]


class TrainingBackend(ABC):
    """Backend contract for training module integrations."""

    name: str

    @abstractmethod
    def register_stats_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the backend-specific corpus-stats parser."""

    @abstractmethod
    def register_build_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the backend-specific build parser."""

    @abstractmethod
    def register_validate_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the backend-specific validation parser."""

    @abstractmethod
    def run_build(self, args: argparse.Namespace) -> int:
        """Execute a backend-specific build request."""

    @abstractmethod
    def run_validate(self, args: argparse.Namespace) -> int:
        """Execute a backend-specific validation request."""

    @abstractmethod
    def run_stats(self, args: argparse.Namespace) -> int:
        """Execute a backend-specific corpus-stats request."""
