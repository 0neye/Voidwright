"""Shared helpers for author-based ship opt-out filtering."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Collection, Sequence

from common.cosmoteer import parse_ship_png
from common.files import iter_ship_png_files

__all__ = [
    "DEFAULT_OPT_OUT_CSV_PATH",
    "OPT_OUT_AUTHOR_NAMES_COLUMN",
    "delete_opted_out_ship_files",
    "load_opt_out_author_names",
]

OPT_OUT_AUTHOR_NAMES_COLUMN = (
    "Your Cosmoteer ship author names to exclude (Exact match; comma-separated list)"
)
DEFAULT_OPT_OUT_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "Voidwright Ship Design Opt-Out Form (Excelsior Community).csv"
)


def load_opt_out_author_names(csv_path: str | Path = DEFAULT_OPT_OUT_CSV_PATH) -> set[str]:
    """Load exact-match author names from the Excelsior opt-out CSV.

    Args:
        csv_path: Path to the CSV export from the opt-out form

    Returns:
        A set of exact author-name matches that should be excluded
    """

    resolved_csv_path = Path(csv_path)
    if not resolved_csv_path.exists():
        logging.warning(
            "Opt-out CSV %s was not found, so author filtering is disabled",
            resolved_csv_path,
        )
        return set()

    with resolved_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or OPT_OUT_AUTHOR_NAMES_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Opt-out CSV {resolved_csv_path} is missing the expected author-name column"
            )

        author_names: set[str] = set()
        for row in reader:
            raw_author_names = row.get(OPT_OUT_AUTHOR_NAMES_COLUMN, "")
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


def delete_opted_out_ship_files(
    input_paths: Sequence[str | Path],
    opt_out_author_names: Collection[str],
) -> dict[str, object]:
    """Delete any ship PNG whose embedded `Author` matches the opt-out list.

    Args:
        input_paths: File and directory inputs to scan for ship PNGs
        opt_out_author_names: Exact author names that should be excluded

    Returns:
        A summary describing how many files were scanned, deleted, or left alone
    """

    resolved_input_paths = [Path(path) for path in input_paths]
    discovered_ship_paths = list(iter_ship_png_files(resolved_input_paths))
    deleted_ship_paths: list[str] = []
    matched_authors: set[str] = set()
    parse_failure_paths: list[str] = []

    if not opt_out_author_names:
        return {
            "ship_files_scanned": len(discovered_ship_paths),
            "ship_files_deleted": 0,
            "ship_files_kept": len(discovered_ship_paths),
            "parse_failures": 0,
            "deleted_ship_paths": deleted_ship_paths,
            "matched_authors": [],
            "parse_failure_paths": parse_failure_paths,
        }

    for ship_path in discovered_ship_paths:
        try:
            ship_data = parse_ship_png(ship_path)
        except Exception as exc:  # noqa: BLE001
            parse_failure_paths.append(str(ship_path))
            logging.warning("Could not inspect ship author for %s: %s", ship_path, exc)
            continue

        author_name = ship_data.get("Author")
        if not isinstance(author_name, str) or author_name not in opt_out_author_names:
            continue

        # Delete the original ship image before downstream stages can consume it
        ship_path.unlink(missing_ok=True)
        deleted_ship_paths.append(str(ship_path))
        matched_authors.add(author_name)
        logging.info("Deleted opted-out ship %s by author %s", ship_path, author_name)

    return {
        "ship_files_scanned": len(discovered_ship_paths),
        "ship_files_deleted": len(deleted_ship_paths),
        "ship_files_kept": len(discovered_ship_paths) - len(deleted_ship_paths),
        "parse_failures": len(parse_failure_paths),
        "deleted_ship_paths": deleted_ship_paths,
        "matched_authors": sorted(matched_authors),
        "parse_failure_paths": parse_failure_paths,
    }
