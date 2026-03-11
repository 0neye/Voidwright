"""End-to-end preprocessing pipeline from local `.ship.png` to graph JSON."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import tempfile
from typing import Sequence

from common.ship_filters import (
    DEFAULT_OPT_OUT_CSV_PATH,
    delete_opted_out_ship_files,
    load_opt_out_author_names,
)

from .canonicalize import run_canonicalize
from .concurrency import add_concurrency_arguments
from .extract import run_extract
from .graphs import generate_all

__all__ = ["build_parser", "run_pipeline", "main"]

_EXTRACTED_SYNC_MANIFEST = ".pipeline-managed-extracted.txt"
_CANONICAL_SYNC_MANIFEST = ".pipeline-managed-canonical.txt"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the end-to-end preprocessing pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the full preprocessing pipeline: local .ship.png inputs -> extracted JSON "
            "-> canonical JSON -> graph JSON outputs."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more local ship PNG files or directories",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_ship_graphs_canonical",
        help="Directory to write the final graph JSON corpus",
    )
    parser.add_argument(
        "--write-extracted-dir",
        default=None,
        help="Optional directory to persist extracted JSON artifacts",
    )
    parser.add_argument(
        "--write-canonical-dir",
        default=None,
        help="Optional directory to persist canonical deduplicated JSON artifacts",
    )
    parser.add_argument(
        "--report-json",
        default="out/ship_canonicalization_report.json",
        help="Canonicalization report path when canonical outputs are persisted",
    )
    parser.add_argument(
        "--report-md",
        default=None,
        help="Optional canonicalization markdown report path when canonical outputs are persisted",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for graph generation during partial runs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during extraction",
    )
    parser.add_argument(
        "--opt-out-csv",
        default=str(DEFAULT_OPT_OUT_CSV_PATH),
        help="CSV containing exact author names to exclude before extraction",
    )
    add_concurrency_arguments(
        parser,
        worker_flag="--extract-workers",
        executor_flag="--extract-executor",
        help_prefix="pipeline extraction",
    )
    add_concurrency_arguments(
        parser,
        worker_flag="--canonicalize-workers",
        executor_flag="--canonicalize-executor",
        help_prefix="pipeline canonicalization",
    )
    add_concurrency_arguments(
        parser,
        worker_flag="--graph-workers",
        executor_flag="--graph-executor",
        help_prefix="pipeline graph generation",
    )
    return parser


def _read_managed_relative_paths(manifest_path: Path) -> set[str]:
    """Load the set of stage-managed relative file paths from a manifest."""

    if not manifest_path.exists():
        return set()
    return {
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _remove_file_and_empty_parents(path: Path, *, stop_dir: Path) -> None:
    """Remove one file and any newly empty parent directories below *stop_dir*."""

    if path.exists():
        path.unlink()

    current = path.parent
    while current != stop_dir and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _sync_stage_outputs(
    source_dir: Path,
    destination_dir: Path,
    *,
    manifest_name: str,
) -> None:
    """Persist one stage's current outputs without deleting unrelated files.

    The pipeline itself always works from isolated temp directories so later
    stages only see the current run's artifacts. When a persistent directory is
    requested, we sync just the stage-managed files into it and use a small
    manifest to prune stale managed outputs from previous runs.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_dir / manifest_name
    previous_relative_paths = _read_managed_relative_paths(manifest_path)
    current_relative_paths = {
        str(path.relative_to(source_dir))
        for path in sorted(source_dir.rglob("*"))
        if path.is_file()
    }

    for relative_path in sorted(previous_relative_paths - current_relative_paths):
        _remove_file_and_empty_parents(destination_dir / relative_path, stop_dir=destination_dir)

    for relative_path in sorted(current_relative_paths):
        source_path = source_dir / relative_path
        destination_path = destination_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    manifest_contents = "\n".join(sorted(current_relative_paths))
    if manifest_contents:
        manifest_contents += "\n"
    manifest_path.write_text(manifest_contents, encoding="utf-8")


def run_pipeline(
    input_paths: Sequence[str | Path],
    output_dir: str | Path = "generated_ship_graphs_canonical",
    write_extracted_dir: str | Path | None = None,
    write_canonical_dir: str | Path | None = None,
    report_json: str | Path = "out/ship_canonicalization_report.json",
    report_md: str | Path | None = None,
    limit: int | None = None,
    verbose: bool = False,
    opt_out_csv: str | Path = DEFAULT_OPT_OUT_CSV_PATH,
    extract_workers: int | None = None,
    extract_executor: str = "auto",
    canonicalize_workers: int | None = None,
    canonicalize_executor: str = "auto",
    graph_workers: int | None = None,
    graph_executor: str = "auto",
) -> dict:
    """Run the local ship preprocessing pipeline.

    Args:
        input_paths: Local `.ship.png` files or directories to preprocess
        output_dir: Final graph JSON output directory
        write_extracted_dir: Optional persistent extracted JSON directory
        write_canonical_dir: Optional persistent canonical JSON directory
        report_json: Canonicalization JSON report path when persisting outputs
        report_md: Optional canonicalization markdown report path when persisting outputs
        limit: Optional limit for the graph-generation stage
        verbose: When True, enable verbose extraction logging
        opt_out_csv: CSV containing exact author names to filter before extraction
        extract_workers: Optional extraction worker-count override
        extract_executor: Extraction executor mode override
        canonicalize_workers: Optional canonicalization worker-count override
        canonicalize_executor: Canonicalization executor mode override
        graph_workers: Optional graph-generation worker-count override
        graph_executor: Graph-generation executor mode override

    Returns:
        Summary payload describing the produced artifacts
    """

    final_graph_output_dir = Path(output_dir)
    persistent_extracted_dir = Path(write_extracted_dir) if write_extracted_dir else None
    persistent_canonical_dir = Path(write_canonical_dir) if write_canonical_dir else None
    report_json_path = Path(report_json)
    report_md_path = Path(report_md) if report_md else None
    resolved_input_paths = [Path(input_path) for input_path in input_paths]
    missing_input_paths = [path for path in resolved_input_paths if not path.exists()]
    if missing_input_paths:
        missing_inputs_text = ", ".join(str(path) for path in missing_input_paths)
        raise RuntimeError(f"Input path does not exist: {missing_inputs_text}")

    # Validate all requested inputs up front so failed invocations never mutate
    # the user's ship corpus as a side effect of opt-out filtering.
    opt_out_filter = delete_opted_out_ship_files(
        resolved_input_paths,
        load_opt_out_author_names(opt_out_csv),
    )
    filtered_input_paths = [path for path in resolved_input_paths if path.exists()]

    with tempfile.TemporaryDirectory(prefix="ship_preprocess_") as temp_dir:
        temp_root = Path(temp_dir)
        extracted_dir = temp_root / "extracted"
        canonical_dir = temp_root / "canonical"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        canonical_dir.mkdir(parents=True, exist_ok=True)

        extract_exit_code = run_extract(
            input_paths=filtered_input_paths,
            output_dir=extracted_dir,
            verbose=verbose,
            workers=extract_workers,
            executor=extract_executor,
        )
        if extract_exit_code not in (0, 2):
            raise RuntimeError(f"Extraction failed with exit code {extract_exit_code}")
        if persistent_extracted_dir is not None:
            _sync_stage_outputs(
                extracted_dir,
                persistent_extracted_dir,
                manifest_name=_EXTRACTED_SYNC_MANIFEST,
            )

        # Always canonicalize before graph generation so the final graph output has
        # already gone through deduplication and preprocessing normalization.
        canonicalize_manifest = run_canonicalize(
            input_dir=extracted_dir,
            output_dir=canonical_dir,
            report_json=report_json_path,
            report_md=report_md_path,
            workers=canonicalize_workers,
            executor=canonicalize_executor,
        )
        graph_manifest = generate_all(
            canonical_dir,
            final_graph_output_dir,
            limit=limit,
            workers=graph_workers,
            executor=graph_executor,
        )
        if persistent_canonical_dir is not None:
            _sync_stage_outputs(
                canonical_dir,
                persistent_canonical_dir,
                manifest_name=_CANONICAL_SYNC_MANIFEST,
            )

        return {
            "inputs": [str(Path(input_path)) for input_path in input_paths],
            "final_graph_output_dir": str(final_graph_output_dir),
            "extracted_output_dir": str(persistent_extracted_dir) if persistent_extracted_dir else None,
            "canonical_output_dir": str(persistent_canonical_dir) if persistent_canonical_dir else None,
            "canonicalization_report_json": str(report_json_path),
            "canonicalization_report_md": str(report_md_path) if report_md_path else None,
            "extract_exit_code": extract_exit_code,
            "opt_out_filter": opt_out_filter,
            "canonicalization": canonicalize_manifest,
            "graphs": graph_manifest,
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preprocessing pipeline CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_pipeline(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        write_extracted_dir=args.write_extracted_dir,
        write_canonical_dir=args.write_canonical_dir,
        report_json=args.report_json,
        report_md=args.report_md,
        limit=args.limit,
        verbose=args.verbose,
        opt_out_csv=args.opt_out_csv,
        extract_workers=args.extract_workers,
        extract_executor=args.extract_executor,
        canonicalize_workers=args.canonicalize_workers,
        canonicalize_executor=args.canonicalize_executor,
        graph_workers=args.graph_workers,
        graph_executor=args.graph_executor,
    )
    print(payload["graphs"]["ships_processed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
