"""Shared file-discovery helpers for ship corpus artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "is_supported_ship_png",
    "output_name_for_ship_png",
    "iter_ship_png_files",
    "iter_json_files",
    "prune_stale_json_outputs",
]


def is_supported_ship_png(path: Path) -> bool:
    """Return True when *path* looks like a supported ship image."""

    name = path.name.lower()
    return name.endswith(".ship.png") or (name.endswith(".png") and ".ship__msg" in name)


def output_name_for_ship_png(source_path: Path) -> str:
    """Return the extracted JSON filename for a source ship PNG."""

    name = source_path.name
    if name.lower().endswith(".png"):
        return f"{name[:-len('.png')]}.json"
    return f"{name}.json"


def iter_ship_png_files(input_paths: Sequence[Path]) -> Iterator[Path]:
    """Yield supported ship PNG files from file and directory inputs.

    Args:
        input_paths: One or more input files or directories to scan

    Yields:
        Matching `.ship.png`-style files in deterministic order
    """

    discovered_paths: set[Path] = set()
    for input_path in input_paths:
        if input_path.is_file():
            if is_supported_ship_png(input_path):
                discovered_paths.add(input_path)
            continue
        if input_path.is_dir():
            for candidate_path in input_path.rglob("*"):
                if candidate_path.is_file() and is_supported_ship_png(candidate_path):
                    discovered_paths.add(candidate_path)
    for discovered_path in sorted(discovered_paths):
        yield discovered_path


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    """Return all JSON files under *input_dir* in deterministic order."""

    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def prune_stale_json_outputs(
    output_dir: Path,
    expected_names: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> int:
    """Delete JSON files in *output_dir* that are not in *expected_names*.

    Args:
        output_dir: Directory to prune.
        expected_names: Filenames that should be kept.
        exclude: Additional filenames to keep regardless of *expected_names*
            (e.g. ``["manifest.json"]``).

    Returns:
        Number of files deleted.
    """

    keep = set(expected_names) | set(exclude)
    pruned = 0
    for path in sorted(output_dir.glob("*.json")):
        if path.name not in keep:
            path.unlink()
            pruned += 1
    return pruned
