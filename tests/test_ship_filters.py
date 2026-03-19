"""Tests for author-based ship opt-in filtering."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common.cosmoteer import create_ship_png_bytes
from common.files import output_name_for_ship_png
from common.ship_filters import delete_non_opted_in_ship_files, load_opt_in_author_names
from preprocessing.pipeline import run_pipeline


def _build_ship_payload(*, name: str, author: str) -> dict:
    """Build a minimal valid ship payload for filter tests."""

    return {
        "Version": 1,
        "Name": name,
        "Author": author,
        "FlightDirection": 0,
        "Parts": [
            {
                "ID": "cosmoteer.corridor",
                "Location": [0, 0],
                "Rotation": 0,
            }
        ],
        "Doors": [],
    }


def _write_ship_png(path: Path, *, name: str, author: str) -> None:
    """Write one `.ship.png` fixture to disk."""

    path.write_bytes(create_ship_png_bytes(_build_ship_payload(name=name, author=author)))


def _write_opt_in_csv(path: Path, author_names: str) -> None:
    """Write a minimal opt-in CSV fixture with the expected author column."""

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Timestamp",
                "Your Cosmoteer ship author names to include (Exact match; comma-separated list)\n\nThis is the name shown in-game in the Author field.",
            ]
        )
        writer.writerow(["2026/03/15 4:45:58 PM MDT", author_names])


def test_load_opt_in_author_names_splits_and_trims_entries(tmp_path: Path) -> None:
    """CSV parsing should preserve exact names while trimming separator whitespace."""

    csv_path = tmp_path / "opt-in.csv"
    _write_opt_in_csv(csv_path, "alpha, beta ,gamma")

    assert load_opt_in_author_names(csv_path) == {"alpha", "beta", "gamma"}


def test_delete_non_opted_in_ship_files_uses_exact_author_matches(tmp_path: Path) -> None:
    """Filtering should keep exact opt-in matches and delete all others."""

    csv_path = tmp_path / "opt-in.csv"
    _write_opt_in_csv(csv_path, "allowed")
    opted_in_ship_path = tmp_path / "opted-in.ship.png"
    case_variant_ship_path = tmp_path / "case-variant.ship.png"
    excluded_ship_path = tmp_path / "excluded.ship.png"
    _write_ship_png(opted_in_ship_path, name="OptedIn", author="allowed")
    _write_ship_png(case_variant_ship_path, name="CaseVariant", author="Allowed")
    _write_ship_png(excluded_ship_path, name="Excluded", author="excluded")

    filter_summary = delete_non_opted_in_ship_files(
        [tmp_path],
        load_opt_in_author_names(csv_path),
    )

    assert filter_summary["ship_files_scanned"] == 3
    assert filter_summary["ship_files_deleted"] == 2
    assert opted_in_ship_path.exists()
    assert not case_variant_ship_path.exists()
    assert not excluded_ship_path.exists()


def test_delete_non_opted_in_ship_files_cache_skips_reparsing(tmp_path: Path) -> None:
    """A warm cache should skip re-parsing unchanged files on the second call."""

    import json

    csv_path = tmp_path / "opt-in.csv"
    _write_opt_in_csv(csv_path, "allowed")
    ship_path = tmp_path / "kept.ship.png"
    _write_ship_png(ship_path, name="Kept", author="allowed")
    cache_path = tmp_path / ".ship-filter-cache.json"
    opt_in = load_opt_in_author_names(csv_path)

    # First call: cold cache — file should be parsed and cache written
    delete_non_opted_in_ship_files([tmp_path], opt_in, cache_path=cache_path)
    assert cache_path.exists()
    cache_after_first = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(cache_after_first) == 1
    entry = next(iter(cache_after_first.values()))
    assert entry["author"] == "allowed"
    assert entry["parse_ok"] is True

    # Overwrite the file with garbage of the same size and restore mtime so the
    # cache entry remains valid by (mtime, size).  A re-parse of this data would
    # fail, so a kept result proves the cache was used instead.
    import os

    original_stat = ship_path.stat()
    ship_path.write_bytes(b"\x00" * original_stat.st_size)
    os.utime(ship_path, (original_stat.st_atime, original_stat.st_mtime))

    # Second call: warm cache — should use cached author, not attempt to parse
    result = delete_non_opted_in_ship_files([tmp_path], opt_in, cache_path=cache_path)
    assert result["ship_files_kept"] == 1
    assert result["parse_failures"] == 0


def test_pipeline_deletes_non_opted_in_source_ships_before_extraction(tmp_path: Path) -> None:
    """The preprocessing pipeline should delete non-opted-in source ships before extract runs."""

    input_dir = tmp_path / "downloaded_ships"
    graph_output_dir = tmp_path / "generated_graphs"
    csv_path = tmp_path / "opt-in.csv"
    excluded_ship_path = input_dir / "excluded.ship.png"
    opted_in_ship_path = input_dir / "opted-in.ship.png"
    input_dir.mkdir()
    _write_opt_in_csv(csv_path, "opted-in-author")
    _write_ship_png(excluded_ship_path, name="Excluded", author="excluded-author")
    _write_ship_png(opted_in_ship_path, name="OptedIn", author="opted-in-author")

    payload = run_pipeline(
        input_paths=[input_dir],
        output_dir=graph_output_dir,
        opt_in_csv=csv_path,
        extract_workers=1,
        extract_executor="thread",
        canonicalize_workers=1,
        canonicalize_executor="thread",
        graph_workers=1,
        graph_executor="thread",
    )

    manifest = json.loads((graph_output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert payload["opt_out_filter"]["ship_files_deleted"] == 1
    assert payload["graphs"]["ships_processed"] == 1
    assert not excluded_ship_path.exists()
    assert opted_in_ship_path.exists()
    assert manifest["sample_outputs"] == [output_name_for_ship_png(opted_in_ship_path)]


def test_pipeline_validates_missing_inputs_before_filtering_ships(tmp_path: Path) -> None:
    """A missing input path should abort before the pipeline mutates valid source dirs."""

    input_dir = tmp_path / "downloaded_ships"
    missing_input_dir = tmp_path / "missing_ships"
    csv_path = tmp_path / "opt-in.csv"
    opted_in_ship_path = input_dir / "opted-in.ship.png"
    input_dir.mkdir()
    _write_opt_in_csv(csv_path, "opted-in-author")
    _write_ship_png(opted_in_ship_path, name="OptedIn", author="opted-in-author")

    try:
        run_pipeline(
            input_paths=[input_dir, missing_input_dir],
            output_dir=tmp_path / "generated_graphs",
            opt_in_csv=csv_path,
            extract_workers=1,
            extract_executor="thread",
            canonicalize_workers=1,
            canonicalize_executor="thread",
            graph_workers=1,
            graph_executor="thread",
        )
    except RuntimeError as exc:
        assert "Input path does not exist" in str(exc)
    else:
        raise AssertionError("run_pipeline should fail for missing input paths")

    assert opted_in_ship_path.exists()
