"""Shared helpers for author-based ship opt-in filtering."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Collection, Sequence

import orjson

from common.cosmoteer import parse_ship_png
from common.files import iter_ship_png_files

__all__ = [
    "DEFAULT_OPT_IN_CSV_PATH",
    "OPT_IN_AUTHOR_NAMES_COLUMN",
    "delete_non_opted_in_ship_files",
    "load_opt_in_author_names",
]

OPT_IN_AUTHOR_NAMES_COLUMN = (
    "Your Cosmoteer ship author names to include (Exact match; comma-separated list)"
    "\n\nThis is the name shown in-game in the Author field."
)
DEFAULT_OPT_IN_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "Voidwright Ship Design Opt-In Form.csv"
)


def load_opt_in_author_names(csv_path: str | Path = DEFAULT_OPT_IN_CSV_PATH) -> set[str]:
    """Load exact-match author names from the Excelsior opt-in CSV.

    Args:
        csv_path: Path to the CSV export from the opt-in form

    Returns:
        A set of exact author-name matches that should be included.
        Returns an empty set (filtering disabled) when the CSV is missing.
    """

    resolved_csv_path = Path(csv_path)
    if not resolved_csv_path.exists():
        logging.warning(
            "Opt-in CSV %s was not found, so author filtering is disabled",
            resolved_csv_path,
        )
        return set()

    with resolved_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or OPT_IN_AUTHOR_NAMES_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Opt-in CSV {resolved_csv_path} is missing the expected author-name column"
            )

        author_names: set[str] = set()
        for row in reader:
            raw_author_names = row.get(OPT_IN_AUTHOR_NAMES_COLUMN, "")
            if not raw_author_names:
                continue

            # The form stores exact-match names as a comma-separated list in one column
            parsed_names = (
                candidate_name.strip()
                for candidate_name in raw_author_names.split(",")
            )
            author_names.update(
                candidate_name for candidate_name in parsed_names if candidate_name
            )

    return author_names


def _load_filter_cache(cache_path: Path | None) -> dict[str, dict]:
    if cache_path is None:
        return {}
    try:
        return orjson.loads(cache_path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not read ship filter cache %s; starting fresh: %s", cache_path, exc)
        return {}


def _save_filter_cache(cache_path: Path | None, cache: dict[str, dict]) -> None:
    if cache_path is None:
        return
    try:
        cache_path.write_bytes(orjson.dumps(cache))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not write ship filter cache %s: %s", cache_path, exc)


def delete_non_opted_in_ship_files(
    input_paths: Sequence[str | Path],
    opt_in_author_names: Collection[str],
    *,
    cache_path: Path | None = None,
) -> dict[str, object]:
    """Delete any ship PNG whose embedded ``Author`` is not in the opt-in list.

    When ``opt_in_author_names`` is empty the function is a no-op (all files
    are kept), mirroring the behaviour when the opt-in CSV is absent.

    When ``cache_path`` is provided the function persists a stat-keyed author
    cache so that unchanged files are not re-parsed on subsequent runs.  Each
    cache entry stores ``mtime``, ``size``, ``author``, and ``parse_ok``; an
    entry is considered stale when either ``mtime`` or ``size`` differs from
    the current ``stat()`` result.

    Args:
        input_paths: File and directory inputs to scan for ship PNGs
        opt_in_author_names: Exact author names that should be kept
        cache_path: Optional path to a JSON author-cache file for skipping
            re-parsing of unchanged ship files across pipeline runs

    Returns:
        A summary describing how many files were scanned, deleted, or left alone
    """

    resolved_input_paths = [Path(path) for path in input_paths]
    discovered_ship_paths = list(iter_ship_png_files(resolved_input_paths))
    deleted_ship_paths: list[str] = []
    deleted_authors: set[str] = set()
    parse_failure_paths: list[str] = []

    if not opt_in_author_names:
        return {
            "ship_files_scanned": len(discovered_ship_paths),
            "ship_files_deleted": 0,
            "ship_files_kept": len(discovered_ship_paths),
            "parse_failures": 0,
            "deleted_ship_paths": deleted_ship_paths,
            "matched_authors": [],
            "parse_failure_paths": parse_failure_paths,
        }

    cache = _load_filter_cache(cache_path)
    cache_dirty = False
    seen_keys: set[str] = set()

    for ship_path in discovered_ship_paths:
        cache_key = str(ship_path.resolve())
        seen_keys.add(cache_key)
        stat = ship_path.stat()
        entry = cache.get(cache_key)

        if (
            entry is not None
            and entry.get("mtime") == stat.st_mtime
            and entry.get("size") == stat.st_size
        ):
            # Cache hit: skip parsing entirely
            if not entry.get("parse_ok", False):
                parse_failure_paths.append(str(ship_path))
                continue
            author_name = entry.get("author")
        else:
            # Cache miss or stale: parse the file and update the cache
            try:
                ship_data = parse_ship_png(ship_path)
                raw_author = ship_data.get("Author")
                author_name = raw_author if isinstance(raw_author, str) else None
                cache[cache_key] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "author": author_name,
                    "parse_ok": True,
                }
            except Exception as exc:  # noqa: BLE001
                parse_failure_paths.append(str(ship_path))
                logging.warning("Could not inspect ship author for %s: %s", ship_path, exc)
                cache[cache_key] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "author": None,
                    "parse_ok": False,
                }
                cache_dirty = True
                continue
            cache_dirty = True

        if isinstance(author_name, str) and author_name in opt_in_author_names:
            continue

        # Delete the original ship image before downstream stages can consume it
        ship_path.unlink(missing_ok=True)
        if cache.pop(cache_key, None) is not None:
            cache_dirty = True
        seen_keys.discard(cache_key)
        deleted_ship_paths.append(str(ship_path))
        if isinstance(author_name, str):
            deleted_authors.add(author_name)
        logging.info("Deleted non-opted-in ship %s by author %r", ship_path, author_name)

    # Prune orphaned entries (files removed from the corpus since the last run)
    stale_keys = cache.keys() - seen_keys
    if stale_keys:
        for key in stale_keys:
            del cache[key]
        cache_dirty = True

    if cache_dirty:
        _save_filter_cache(cache_path, cache)

    return {
        "ship_files_scanned": len(discovered_ship_paths),
        "ship_files_deleted": len(deleted_ship_paths),
        "ship_files_kept": len(discovered_ship_paths) - len(deleted_ship_paths),
        "parse_failures": len(parse_failure_paths),
        "deleted_ship_paths": deleted_ship_paths,
        "matched_authors": sorted(deleted_authors),
        "parse_failure_paths": parse_failure_paths,
    }
