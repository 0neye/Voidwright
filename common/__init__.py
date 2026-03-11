"""Shared utilities used across preprocessing, training, and generation modules."""

from .cosmoteer_install import (
    DEFAULT_LOCAL_ICON_CACHE_ROOT,
    find_cosmoteer_install_root,
    resolve_terran_part_icons_root,
)
from .files import (
    is_supported_ship_png,
    iter_json_files,
    iter_ship_png_files,
    output_name_for_ship_png,
)
from .logging import configure_logging

__all__ = [
    "DEFAULT_LOCAL_ICON_CACHE_ROOT",
    "configure_logging",
    "find_cosmoteer_install_root",
    "is_supported_ship_png",
    "iter_json_files",
    "iter_ship_png_files",
    "output_name_for_ship_png",
    "resolve_terran_part_icons_root",
]
