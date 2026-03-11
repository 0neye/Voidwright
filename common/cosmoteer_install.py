"""Helpers for locating a local Cosmoteer install and Terran part icons."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

__all__ = [
    "DEFAULT_LOCAL_ICON_CACHE_ROOT",
    "find_cosmoteer_install_root",
    "iter_steam_install_paths",
    "iter_steam_library_paths",
    "parse_steam_libraryfolders_vdf",
    "resolve_terran_part_icons_root",
]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_ICON_CACHE_ROOT = REPO_ROOT / "assets" / "local" / "cosmoteer-icons" / "terran"
_STEAM_REGISTRY_KEYS = (
    r"SOFTWARE\WOW6432Node\Valve\Steam",
    r"SOFTWARE\Valve\Steam",
)
_STEAM_LIBRARY_PATH_RE = re.compile(r'"path"\s*"([^"]+)"')
_STEAM_LIBRARYFOLDERS_RELATIVE_PATHS = (
    Path("steamapps") / "libraryfolders.vdf",
    Path("config") / "libraryfolders.vdf",
)


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Return normalized paths in first-seen order without duplicates."""

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return tuple(deduped)


def parse_steam_libraryfolders_vdf(raw_text: str) -> tuple[Path, ...]:
    """Extract Steam library paths from ``libraryfolders.vdf`` text."""

    libraries = []
    for matched_path in _STEAM_LIBRARY_PATH_RE.findall(raw_text):
        libraries.append(Path(matched_path.replace("\\\\", "\\")))
    return _dedupe_paths(libraries)


def _iter_libraryfolders_vdf_paths(steam_install_path: Path) -> tuple[Path, ...]:
    """Return likely ``libraryfolders.vdf`` locations for a Steam install."""

    return tuple(
        steam_install_path / relative_path
        for relative_path in _STEAM_LIBRARYFOLDERS_RELATIVE_PATHS
    )


def iter_steam_install_paths() -> tuple[Path, ...]:
    """Return Steam install paths discovered from the local Windows registry."""

    if os.name != "nt":
        return ()

    try:
        import winreg
    except ImportError:
        return ()

    install_paths: list[Path] = []
    hives = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    for hive in hives:
        for registry_key in _STEAM_REGISTRY_KEYS:
            try:
                with winreg.OpenKey(hive, registry_key) as opened_key:
                    install_path, _value_type = winreg.QueryValueEx(opened_key, "InstallPath")
            except FileNotFoundError:
                continue
            if install_path:
                install_paths.append(Path(install_path))
    return _dedupe_paths(install_paths)


def iter_steam_library_paths(
    steam_install_paths: Iterable[Path] | None = None,
) -> tuple[Path, ...]:
    """Return Steam library roots discovered from one or more Steam installs."""

    resolved_install_paths = tuple(steam_install_paths or iter_steam_install_paths())
    library_paths: list[Path] = []
    for install_path in resolved_install_paths:
        library_paths.append(install_path)
        for libraryfolders_path in _iter_libraryfolders_vdf_paths(install_path):
            if not libraryfolders_path.exists():
                continue
            library_text = libraryfolders_path.read_text(encoding="utf-8")
            library_paths.extend(parse_steam_libraryfolders_vdf(library_text))
    return _dedupe_paths(library_paths)


def find_cosmoteer_install_root(
    steam_install_paths: Iterable[Path] | None = None,
) -> Path | None:
    """Locate the installed Cosmoteer game root, when present."""

    for library_path in iter_steam_library_paths(steam_install_paths):
        candidate_root = library_path / "steamapps" / "common" / "Cosmoteer"
        if (candidate_root / "Data" / "ships" / "terran").is_dir():
            return candidate_root
    return None


def _validate_icons_root(candidate_root: Path) -> Path | None:
    """Return the normalized icon root when it looks valid."""

    resolved_root = candidate_root.expanduser()
    if not resolved_root.is_dir():
        return None
    if next(resolved_root.rglob("icon.png"), None) is None:
        return None
    return resolved_root


def resolve_terran_part_icons_root(
    *,
    icons_root: str | Path | None = None,
    game_root: str | Path | None = None,
    fallback_root: str | Path | None = None,
) -> Path:
    """Resolve the Terran part-icon root using overrides, Steam discovery, or fallback."""

    if icons_root is not None:
        validated_root = _validate_icons_root(Path(icons_root))
        if validated_root is None:
            raise FileNotFoundError(
                f"Part icon root does not exist or has no icon.png files: {Path(icons_root)}"
            )
        return validated_root

    if game_root is not None:
        candidate_icons_root = Path(game_root) / "Data" / "ships" / "terran"
        validated_root = _validate_icons_root(candidate_icons_root)
        if validated_root is None:
            raise FileNotFoundError(
                "Cosmoteer game root does not contain Terran icons at "
                f"{candidate_icons_root}"
            )
        return validated_root

    discovered_root = find_cosmoteer_install_root()
    if discovered_root is not None:
        discovered_icons_root = _validate_icons_root(discovered_root / "Data" / "ships" / "terran")
        if discovered_icons_root is not None:
            return discovered_icons_root

    resolved_fallback_root = (
        Path(fallback_root) if fallback_root is not None else DEFAULT_LOCAL_ICON_CACHE_ROOT
    )
    validated_fallback = _validate_icons_root(resolved_fallback_root)
    if validated_fallback is not None:
        return validated_fallback

    raise FileNotFoundError(
        "Unable to locate Cosmoteer Terran part icons automatically. "
        f"Use --icons-root or --game-root, or place icons under {resolved_fallback_root}."
    )
