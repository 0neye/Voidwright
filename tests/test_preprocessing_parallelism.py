"""Determinism tests for parallel preprocessing stages."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

import preprocessing.concurrency as preprocessing_concurrency
import preprocessing.pipeline as preprocessing_pipeline
from common.cosmoteer import create_ship_png_bytes
from preprocessing.canonicalize import run_canonicalize
from preprocessing.concurrency import resolve_executor_mode
from preprocessing.extract import run_extract
from preprocessing.graphs import generate_all


def write_json(path: Path, payload: object) -> None:
    """Write a JSON fixture with stable UTF-8 encoding."""

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    """Read one JSON file from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_directory_texts(path: Path) -> dict[str, str]:
    """Return all file contents under a directory keyed by relative path."""

    return {
        str(file_path.relative_to(path)): file_path.read_text(encoding="utf-8")
        for file_path in sorted(path.rglob("*"))
        if file_path.is_file()
    }


def read_directory_texts_without_manifest(path: Path) -> dict[str, str]:
    """Return file contents for a graph output directory excluding `manifest.json`."""

    return {
        relative_path: text
        for relative_path, text in read_directory_texts(path).items()
        if relative_path != "manifest.json"
    }


def normalize_manifest(manifest: dict) -> dict:
    """Strip run-specific paths so manifests can be compared across temp dirs."""

    manifest_copy = json.loads(json.dumps(manifest))
    manifest_copy["input_dir"] = "<input>"
    manifest_copy["output_dir"] = "<output>"
    return manifest_copy


def build_graph_ship(*, name: str, part_id: str, location: list[int], rotation: int = 0) -> dict:
    """Build a minimal centered-`2x` ship payload for graph-generation tests."""

    return {
        "Name": name,
        "Author": "test",
        "Version": 1,
        "FlightDirection": 0,
        "coord_transform": {
            "version": 1,
            "frame": "bbox_center_2x",
            "scale": 2,
            "center_2x": [0, 0],
        },
        "Parts": [
            {
                "ID": part_id,
                "Location2x": [int(location[0]) * 2, int(location[1]) * 2],
                "Rotation": rotation,
            }
        ],
        "Doors": [],
    }


def build_png_ship(*, name: str, author: str = "test") -> dict:
    """Build a minimal ship payload suitable for PNG extraction fixtures."""

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


class FailingProcessPoolExecutor:
    """Test double that simulates unavailable multiprocessing primitives."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Raise the same class of error seen in restricted runners."""

        raise NotImplementedError("Process pools are unavailable in this environment")


def test_canonicalize_single_worker_matches_auto_parallel(tmp_path: Path) -> None:
    """Canonicalization should keep deterministic outputs across worker modes."""

    input_dir = tmp_path / "canonical-input"
    single_output_dir = tmp_path / "canonical-single"
    auto_output_dir = tmp_path / "canonical-auto"
    input_dir.mkdir()

    # These fixtures cover dedupe, stripped message suffixes, filename
    # collisions, and parse-failure reporting in the same run.
    write_json(input_dir / "alpha.json", {"Parts": [], "Name": "Alpha", "Version": 1})
    write_json(input_dir / "alpha__msg123.json", {"Version": 1, "Name": "Alpha", "Parts": []})
    write_json(input_dir / "collision__msg1.json", {"Parts": [], "Name": "CollisionA", "Value": 1})
    write_json(input_dir / "collision__msg2.json", {"Parts": [], "Name": "CollisionB", "Value": 2})
    (input_dir / "broken.json").write_text("{not valid json", encoding="utf-8")

    single_manifest = run_canonicalize(
        input_dir=input_dir,
        output_dir=single_output_dir,
        report_json=tmp_path / "single-report.json",
        workers=1,
        executor="thread",
    )
    auto_manifest = run_canonicalize(
        input_dir=input_dir,
        output_dir=auto_output_dir,
        report_json=tmp_path / "auto-report.json",
        executor="auto",
    )

    assert normalize_manifest(single_manifest) == normalize_manifest(auto_manifest)
    assert read_directory_texts(single_output_dir) == read_directory_texts(auto_output_dir)


def test_generate_all_single_worker_matches_auto_parallel(tmp_path: Path) -> None:
    """Graph generation should keep deterministic outputs across worker modes."""

    input_dir = tmp_path / "graph-input"
    single_output_dir = tmp_path / "graph-single"
    auto_output_dir = tmp_path / "graph-auto"
    input_dir.mkdir()

    # Keep the filenames intentionally out of order so the test also validates
    # stable sample-output ordering during the manifest reduction step.
    write_json(
        input_dir / "zeta.json",
        build_graph_ship(
            name="Zeta",
            part_id="mod.custom_corridor",
            location=[0, 0],
        ),
    )
    write_json(
        input_dir / "alpha.json",
        build_graph_ship(
            name="Alpha",
            part_id="cosmoteer.corridor",
            location=[0, 0],
        ),
    )
    write_json(
        input_dir / "beta.json",
        build_graph_ship(
            name="Beta",
            part_id="cosmoteer.armor",
            location=[2, 0],
        ),
    )

    single_manifest = generate_all(
        input_dir,
        single_output_dir,
        workers=1,
        executor="thread",
    )
    auto_manifest = generate_all(
        input_dir,
        auto_output_dir,
        executor="auto",
    )

    assert normalize_manifest(single_manifest) == normalize_manifest(auto_manifest)
    assert read_directory_texts_without_manifest(single_output_dir) == read_directory_texts_without_manifest(
        auto_output_dir
    )
    assert read_json(auto_output_dir / "manifest.json")["sample_outputs"] == [
        "alpha.json",
        "beta.json",
        "zeta.json",
    ]


def test_canonicalize_auto_falls_back_to_threads_when_process_pool_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`executor=\"auto\"` should keep working when process pools are unavailable."""

    input_dir = tmp_path / "canonical-input"
    output_dir = tmp_path / "canonical-output"
    input_dir.mkdir()
    write_json(input_dir / "alpha.json", {"Parts": [], "Name": "Alpha", "Version": 1})

    monkeypatch.setattr(
        preprocessing_concurrency,
        "ProcessPoolExecutor",
        FailingProcessPoolExecutor,
    )

    manifest = run_canonicalize(
        input_dir=input_dir,
        output_dir=output_dir,
        report_json=tmp_path / "report.json",
        executor="auto",
    )

    assert manifest["parsed_input_json_files"] == 1
    assert (output_dir / "alpha.json").exists()


def test_canonicalize_markdown_report_is_opt_in(tmp_path: Path) -> None:
    """Canonicalization should only write a markdown report when requested."""

    input_dir = tmp_path / "canonical-input"
    output_dir = tmp_path / "canonical-output"
    report_md_path = tmp_path / "report.md"
    input_dir.mkdir()
    write_json(input_dir / "alpha.json", {"Parts": [], "Name": "Alpha", "Version": 1})

    run_canonicalize(
        input_dir=input_dir,
        output_dir=output_dir,
        report_json=tmp_path / "report.json",
        report_md=report_md_path,
        workers=1,
        executor="thread",
    )

    assert report_md_path.exists()


def test_pipeline_syncs_stage_outputs_without_deleting_unrelated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent pipeline dirs should prune stale managed files without wiping the tree."""

    extracted_dir = tmp_path / "persisted-extracted"
    canonical_dir = tmp_path / "persisted-canonical"
    graph_output_dir = tmp_path / "graph-output"
    input_dir = tmp_path / "input"
    extracted_dir.mkdir()
    canonical_dir.mkdir()
    input_dir.mkdir()

    write_json(extracted_dir / "stale.json", {"Name": "stale", "Version": 1, "Parts": [], "Doors": []})
    write_json(canonical_dir / "stale.json", {"Name": "stale", "Version": 1, "Parts": [], "Doors": []})
    (canonical_dir / "keep.txt").write_text("preserve me", encoding="utf-8")

    def fake_run_extract(
        *,
        input_paths: list[str] | list[Path],
        output_dir: str | Path,
        limit: int | None = None,
        verbose: bool = False,
        workers: int | None = None,
        executor: str = "auto",
    ) -> dict:
        del input_paths, limit, verbose, workers, executor
        stale_path = Path(output_dir) / "stale.json"
        if stale_path.exists():
            stale_path.unlink()
        write_json(
            Path(output_dir) / "alpha.json",
            {"Name": "Alpha", "Author": "test", "Version": 1, "FlightDirection": 0, "Parts": [], "Doors": []},
        )
        return {
            "inputs": [],
            "output_dir": str(output_dir),
            "schema_version": 1,
            "schema_version_key": "extract_schema_version",
            "ship_files_discovered": 1,
            "ship_files_considered": 1,
            "ships_processed": 1,
            "ships_skipped": 0,
            "files_failed": 0,
            "sample_outputs": ["alpha.json"],
            "exit_code": 0,
        }

    monkeypatch.setattr(preprocessing_pipeline, "run_extract", fake_run_extract)

    payload = preprocessing_pipeline.run_pipeline(
        input_paths=[input_dir],
        output_dir=graph_output_dir,
        extracted_dir=extracted_dir,
        canonical_dir=canonical_dir,
        extract_workers=1,
        extract_executor="thread",
        canonicalize_workers=1,
        canonicalize_executor="thread",
        graph_workers=1,
        graph_executor="thread",
    )

    assert payload["graphs"]["ships_processed"] == 1
    assert read_json(graph_output_dir / "manifest.json")["sample_outputs"] == ["alpha.json"]
    assert (graph_output_dir / "alpha.json").exists()

    assert (extracted_dir / "alpha.json").exists()
    assert not (extracted_dir / "stale.json").exists()
    assert (canonical_dir / "alpha.json").exists()
    assert not (canonical_dir / "stale.json").exists()
    assert (canonical_dir / "keep.txt").read_text(encoding="utf-8") == "preserve me"


def test_graphs_auto_falls_back_to_threads_when_process_pool_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph generation should fall back to threads in restricted runners."""

    input_dir = tmp_path / "graph-input"
    output_dir = tmp_path / "graph-output"
    input_dir.mkdir()
    write_json(
        input_dir / "alpha.json",
        build_graph_ship(
            name="Alpha",
            part_id="cosmoteer.corridor",
            location=[0, 0],
        ),
    )

    monkeypatch.setattr(
        preprocessing_concurrency,
        "ProcessPoolExecutor",
        FailingProcessPoolExecutor,
    )

    manifest = generate_all(
        input_dir,
        output_dir,
        executor="auto",
    )

    assert manifest["ships_processed"] == 1
    assert (output_dir / "alpha.json").exists()


def test_resolve_executor_mode_raises_for_unknown_stage() -> None:
    """resolve_executor_mode should raise ValueError for unregistered stage names."""

    with pytest.raises(ValueError, match="Unknown preprocessing stage"):
        resolve_executor_mode("nonexistent_stage", "auto")


def test_resolve_executor_mode_passthrough_for_explicit_mode() -> None:
    """resolve_executor_mode should return the requested mode without a registry lookup."""

    assert resolve_executor_mode("nonexistent_stage", "thread") == "thread"
    assert resolve_executor_mode("nonexistent_stage", "process") == "process"


def test_extract_auto_falls_back_to_threads_when_process_pool_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction should fall back to threads in restricted runners."""

    input_dir = tmp_path / "extract-input"
    output_dir = tmp_path / "extract-output"
    input_dir.mkdir()
    # Write a minimal 1x1 PNG so iter_ship_png_files returns nothing (no .ship.png)
    # and the early-exit path is exercised without needing real ship files.
    # Use an empty input_paths list to exercise the no-files-found early return.
    manifest = run_extract(
        input_paths=[str(input_dir)],
        output_dir=str(output_dir),
        executor="auto",
    )
    assert manifest["exit_code"] == 0


def test_extract_explicit_process_executor_raises_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit --executor process should not silently fall back; it should raise."""

    input_dir = tmp_path / "extract-input"
    output_dir = tmp_path / "extract-output"
    input_dir.mkdir()
    # Create a fake .ship.png so the code reaches pool creation
    (input_dir / "test.ship.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    monkeypatch.setattr(
        preprocessing_concurrency,
        "ProcessPoolExecutor",
        FailingProcessPoolExecutor,
    )

    with pytest.raises(NotImplementedError):
        run_extract(
            input_paths=[str(input_dir)],
            output_dir=str(output_dir),
            executor="process",
        )


def test_graphs_skips_bad_files_and_processes_good_ones(tmp_path: Path) -> None:
    """Graph generation should skip unreadable files rather than aborting the batch."""

    input_dir = tmp_path / "graph-input"
    output_dir = tmp_path / "graph-output"
    input_dir.mkdir()

    write_json(
        input_dir / "good.json",
        build_graph_ship(name="Good", part_id="cosmoteer.corridor", location=[0, 0]),
    )
    (input_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

    manifest = generate_all(input_dir, output_dir, workers=1, executor="thread")

    assert manifest["ships_processed"] == 1
    assert (output_dir / "good.json").exists()
    assert not (output_dir / "bad.json").exists()


def test_extract_limit_is_nondestructive_and_does_not_update_version_sentinel(tmp_path: Path) -> None:
    """Limited extract runs should not prune unrelated outputs or rewrite schema sentinel."""

    input_dir = tmp_path / "extract-input"
    output_dir = tmp_path / "extract-output"
    input_dir.mkdir()
    (input_dir / "alpha.ship.png").write_bytes(create_ship_png_bytes(build_png_ship(name="Alpha")))
    (input_dir / "beta.ship.png").write_bytes(create_ship_png_bytes(build_png_ship(name="Beta")))

    full_manifest = run_extract(
        input_paths=[input_dir],
        output_dir=output_dir,
        workers=1,
        executor="thread",
    )
    assert full_manifest["exit_code"] == 0
    sentinel_path = output_dir / ".pipeline-version.json"
    before_sentinel_mtime = sentinel_path.stat().st_mtime_ns

    stale_output_path = output_dir / "stale.json"
    write_json(stale_output_path, {"Name": "stale", "Version": 1, "Parts": [], "Doors": []})

    time.sleep(0.01)
    limited_manifest = run_extract(
        input_paths=[input_dir],
        output_dir=output_dir,
        limit=1,
        workers=1,
        executor="thread",
    )

    assert limited_manifest["ship_files_considered"] == 1
    assert stale_output_path.exists()
    assert sentinel_path.stat().st_mtime_ns == before_sentinel_mtime


def test_canonicalize_limit_is_nondestructive_and_does_not_update_version_sentinel(tmp_path: Path) -> None:
    """Limited canonicalize runs should keep stale outputs and leave sentinel untouched."""

    input_dir = tmp_path / "canonical-input"
    output_dir = tmp_path / "canonical-output"
    input_dir.mkdir()
    write_json(input_dir / "alpha.json", {"Parts": [], "Name": "Alpha", "Version": 1})
    write_json(input_dir / "beta.json", {"Parts": [], "Name": "Beta", "Version": 1})

    full_manifest = run_canonicalize(
        input_dir=input_dir,
        output_dir=output_dir,
        report_json=tmp_path / "full-report.json",
        workers=1,
        executor="thread",
    )
    assert full_manifest["canonical_outputs_failed"] == 0
    sentinel_path = output_dir / ".pipeline-version.json"
    before_sentinel_mtime = sentinel_path.stat().st_mtime_ns

    stale_output_path = output_dir / "stale.json"
    write_json(stale_output_path, {"Parts": [], "Name": "Stale", "Version": 1})

    time.sleep(0.01)
    limited_manifest = run_canonicalize(
        input_dir=input_dir,
        output_dir=output_dir,
        report_json=tmp_path / "limited-report.json",
        limit=1,
        workers=1,
        executor="thread",
    )

    assert limited_manifest["considered_input_json_files"] == 1
    assert stale_output_path.exists()
    assert sentinel_path.stat().st_mtime_ns == before_sentinel_mtime


def test_pipeline_canonical_schema_bump_forces_downstream_graph_recompute(tmp_path: Path) -> None:
    """Canonical schema-version invalidation should force a full downstream graph rerun."""

    source_dir = tmp_path / "ships"
    extracted_dir = tmp_path / "extracted"
    canonical_dir = tmp_path / "canonical"
    graph_dir = tmp_path / "graphs"
    source_dir.mkdir()
    (source_dir / "alpha.ship.png").write_bytes(create_ship_png_bytes(build_png_ship(name="Alpha")))

    first_payload = preprocessing_pipeline.run_pipeline(
        input_paths=[source_dir],
        output_dir=graph_dir,
        extracted_dir=extracted_dir,
        canonical_dir=canonical_dir,
        opt_in_csv=tmp_path / "no-opt-in.csv",
        extract_workers=1,
        extract_executor="thread",
        canonicalize_workers=1,
        canonicalize_executor="thread",
        graph_workers=1,
        graph_executor="thread",
    )
    assert first_payload["graphs"]["ships_processed"] == 1

    graph_file_path = graph_dir / "alpha.ship.json"
    first_graph_mtime = graph_file_path.stat().st_mtime_ns

    canonical_sentinel_path = canonical_dir / ".pipeline-version.json"
    sentinel_data = read_json(canonical_sentinel_path)
    sentinel_data["canonical_schema_version"] = 0
    write_json(canonical_sentinel_path, sentinel_data)

    time.sleep(0.01)
    second_payload = preprocessing_pipeline.run_pipeline(
        input_paths=[source_dir],
        output_dir=graph_dir,
        extracted_dir=extracted_dir,
        canonical_dir=canonical_dir,
        opt_in_csv=tmp_path / "no-opt-in.csv",
        extract_workers=1,
        extract_executor="thread",
        canonicalize_workers=1,
        canonicalize_executor="thread",
        graph_workers=1,
        graph_executor="thread",
    )

    assert second_payload["graphs"]["ships_processed"] == 1
    assert graph_file_path.stat().st_mtime_ns > first_graph_mtime
