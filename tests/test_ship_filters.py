"""Tests for author-based ship opt-out filtering."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common.cosmoteer import create_ship_png_bytes
from common.files import output_name_for_ship_png
from common.ship_filters import delete_opted_out_ship_files, load_opt_out_author_names
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


def _write_opt_out_csv(path: Path, author_names: str) -> None:
    """Write a minimal opt-out CSV fixture with the expected author column."""

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Timestamp",
                "Your Cosmoteer ship author names to exclude (Exact match; comma-separated list)",
            ]
        )
        writer.writerow(["2026/03/11 4:36:05 PM MDT", author_names])


def test_load_opt_out_author_names_splits_and_trims_entries(tmp_path: Path) -> None:
    """CSV parsing should preserve exact names while trimming separator whitespace."""

    csv_path = tmp_path / "opt-out.csv"
    _write_opt_out_csv(csv_path, "alpha, beta ,gamma")

    assert load_opt_out_author_names(csv_path) == {"alpha", "beta", "gamma"}


def test_delete_opted_out_ship_files_uses_exact_author_matches(tmp_path: Path) -> None:
    """Filtering should delete exact author matches without folding case."""

    csv_path = tmp_path / "opt-out.csv"
    _write_opt_out_csv(csv_path, "blocked")
    blocked_ship_path = tmp_path / "blocked.ship.png"
    case_variant_ship_path = tmp_path / "case-variant.ship.png"
    allowed_ship_path = tmp_path / "allowed.ship.png"
    _write_ship_png(blocked_ship_path, name="Blocked", author="blocked")
    _write_ship_png(case_variant_ship_path, name="CaseVariant", author="Blocked")
    _write_ship_png(allowed_ship_path, name="Allowed", author="allowed")

    filter_summary = delete_opted_out_ship_files(
        [tmp_path],
        load_opt_out_author_names(csv_path),
    )

    assert filter_summary["ship_files_scanned"] == 3
    assert filter_summary["ship_files_deleted"] == 1
    assert filter_summary["matched_authors"] == ["blocked"]
    assert not blocked_ship_path.exists()
    assert case_variant_ship_path.exists()
    assert allowed_ship_path.exists()


def test_pipeline_deletes_opted_out_source_ships_before_extraction(tmp_path: Path) -> None:
    """The preprocessing pipeline should delete blocked source ships before extract runs."""

    input_dir = tmp_path / "downloaded_ships"
    graph_output_dir = tmp_path / "generated_graphs"
    csv_path = tmp_path / "opt-out.csv"
    blocked_ship_path = input_dir / "blocked.ship.png"
    allowed_ship_path = input_dir / "allowed.ship.png"
    input_dir.mkdir()
    _write_opt_out_csv(csv_path, "blocked")
    _write_ship_png(blocked_ship_path, name="Blocked", author="blocked")
    _write_ship_png(allowed_ship_path, name="Allowed", author="allowed")

    payload = run_pipeline(
        input_paths=[input_dir],
        output_dir=graph_output_dir,
        opt_out_csv=csv_path,
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
    assert not blocked_ship_path.exists()
    assert allowed_ship_path.exists()
    assert manifest["sample_outputs"] == [output_name_for_ship_png(allowed_ship_path)]


def test_pipeline_validates_missing_inputs_before_deleting_opted_out_ships(tmp_path: Path) -> None:
    """A missing input path should abort before the pipeline mutates valid source dirs."""

    input_dir = tmp_path / "downloaded_ships"
    missing_input_dir = tmp_path / "missing_ships"
    csv_path = tmp_path / "opt-out.csv"
    blocked_ship_path = input_dir / "blocked.ship.png"
    input_dir.mkdir()
    _write_opt_out_csv(csv_path, "blocked")
    _write_ship_png(blocked_ship_path, name="Blocked", author="blocked")

    try:
        run_pipeline(
            input_paths=[input_dir, missing_input_dir],
            output_dir=tmp_path / "generated_graphs",
            opt_out_csv=csv_path,
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

    assert blocked_ship_path.exists()
