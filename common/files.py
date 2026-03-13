"""Shared file-discovery helpers for ship corpus artifacts."""

from __future__ import annotations

import orjson
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "is_supported_ship_png",
    "output_name_for_ship_png",
    "iter_ship_png_files",
    "iter_json_files",
    "prune_stale_json_outputs",
    "inputs_needing_regeneration",
    "write_output_version",
]

# Hidden sentinel file written to each managed output directory.  It stores
# one or more version keys so callers can detect when a schema or backend
# version bump requires full regeneration of that directory's outputs.
_VERSION_SENTINEL = ".pipeline-version.json"


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

    Hidden files (names starting with ``"."``) are never deleted so that
    pipeline sentinel files such as :data:`_VERSION_SENTINEL` are preserved.

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
        if path.name.startswith("."):
            continue
        if path.name not in keep:
            path.unlink()
            pruned += 1
    return pruned


def inputs_needing_regeneration(
    input_files: Sequence[Path],
    output_dir: Path,
    *,
    current_version: int,
    version_key: str,
) -> list[Path]:
    """Return input files whose outputs need to be generated or regenerated.

    Reads the version stored in *output_dir*'s :data:`_VERSION_SENTINEL` file
    and compares it against *current_version*:

    - **Version mismatch or no sentinel**: all *input_files* are returned so
      the caller regenerates every output (e.g. after a schema bump).
    - **Version matches**: only input files whose corresponding output file is
      absent or older than the input are returned, enabling fast incremental
      reruns while still catching edited inputs.

    Call :func:`write_output_version` after successfully writing all outputs
    to persist the current version for the next run.

    Args:
        input_files: Full set of input files for this run.
        output_dir: Directory that receives generated output files.
        current_version: Version value that represents up-to-date output.
        version_key: Key used to store and retrieve the version in the sentinel.

    Returns:
        Subset of *input_files* that require (re)generation.
    """

    stored_version: int | None = None
    sentinel = output_dir / _VERSION_SENTINEL
    if sentinel.exists():
        try:
            data = orjson.loads(sentinel.read_text(encoding="utf-8"))
            val = data.get(version_key)
            if val is not None:
                stored_version = int(val)
        except Exception:
            pass

    if stored_version != current_version:
        return list(input_files)

    # Version matches — only process files whose outputs are absent or stale.
    def _needs_regen(f: Path) -> bool:
        out = output_dir / f.name
        if not out.exists():
            return True
        return f.stat().st_mtime > out.stat().st_mtime

    return [f for f in input_files if _needs_regen(f)]


def write_output_version(output_dir: Path, version_key: str, version: int) -> None:
    """Persist a version value in *output_dir*'s :data:`_VERSION_SENTINEL` file.

    Multiple version keys can coexist in the same sentinel so a directory used
    by more than one versioned stage retains all its markers.

    Args:
        output_dir: Target directory (must already exist).
        version_key: Key under which to store the version value.
        version: Current version number to record.
    """

    sentinel = output_dir / _VERSION_SENTINEL
    data: dict = {}
    if sentinel.exists():
        try:
            data = orjson.loads(sentinel.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[version_key] = version
    sentinel.write_text(
        orjson.dumps(data, option=orjson.OPT_INDENT_2).decode() + "\n",
        encoding="utf-8",
    )
