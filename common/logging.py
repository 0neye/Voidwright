"""Shared logging helpers for command-line entrypoints."""

from __future__ import annotations

import logging

__all__ = ["configure_logging"]


def configure_logging(verbose: bool) -> None:
    """Configure repository-wide CLI logging.

    Args:
        verbose: When True, emit debug logging instead of info logging
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
