"""End-to-end preprocessing pipeline from local `.ship.png` to graph JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from common.ship_filters import (
    DEFAULT_OPT_IN_CSV_PATH,
    delete_non_opted_in_ship_files,
    load_opt_in_author_names,
)

from .canonicalize import run_canonicalize
from .concurrency import add_concurrency_arguments
from .extract import run_extract
from .graphs import generate_all

__all__ = ["build_parser", "run_pipeline", "main"]


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
        "--extracted-dir",
        default="extracted_ship_data",
        help="Directory to write extracted JSON artifacts",
    )
    parser.add_argument(
        "--canonical-dir",
        default="extracted_ship_data_canonical",
        help="Directory to write canonical deduplicated JSON artifacts",
    )
    parser.add_argument(
        "--report-json",
        default="out/ship_canonicalization_report.json",
        help="Canonicalization report JSON path",
    )
    parser.add_argument(
        "--report-md",
        default=None,
        help="Optional canonicalization markdown report path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for partial validation runs at each stage",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during extraction",
    )
    parser.add_argument(
        "--opt-in-csv",
        default=str(DEFAULT_OPT_IN_CSV_PATH),
        help="CSV containing exact author names to include before extraction",
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
    parser.add_argument(
        "--expansion-output-dir",
        default=None,
        help=(
            "Optional directory to write enriched graph JSON. "
            "When omitted, graph expansion is skipped."
        ),
    )
    parser.add_argument(
        "--expansion-backend",
        default="structural",
        help="Graph expansion pipeline name to use (default: structural)",
    )
    add_concurrency_arguments(
        parser,
        worker_flag="--expansion-workers",
        executor_flag="--expansion-executor",
        help_prefix="pipeline graph expansion",
    )
    return parser


def run_pipeline(
    input_paths: Sequence[str | Path],
    output_dir: str | Path = "generated_ship_graphs_canonical",
    extracted_dir: str | Path = "extracted_ship_data",
    canonical_dir: str | Path = "extracted_ship_data_canonical",
    report_json: str | Path = "out/ship_canonicalization_report.json",
    report_md: str | Path | None = None,
    limit: int | None = None,
    verbose: bool = False,
    opt_in_csv: str | Path = DEFAULT_OPT_IN_CSV_PATH,
    filter_cache_path: Path | None = None,
    extract_workers: int | None = None,
    extract_executor: str = "auto",
    canonicalize_workers: int | None = None,
    canonicalize_executor: str = "auto",
    graph_workers: int | None = None,
    graph_executor: str = "auto",
    expansion_output_dir: str | Path | None = None,
    expansion_backend: str = "structural",
    expansion_workers: int | None = None,
    expansion_executor: str = "auto",
) -> dict:
    """Run the local ship preprocessing pipeline.

    Args:
        input_paths: Local `.ship.png` files or directories to preprocess
        output_dir: Final graph JSON output directory
        extracted_dir: Persistent extracted JSON output directory
        canonical_dir: Persistent canonical JSON output directory
        report_json: Canonicalization JSON report path
        report_md: Optional canonicalization markdown report path
        limit: Optional validation subset size for each stage
        verbose: When True, enable verbose extraction logging
        opt_in_csv: CSV containing exact author names to keep before extraction
        filter_cache_path: Optional path for the author-cache JSON file used to
            skip re-parsing unchanged ship files during opt-in filtering
        extract_workers: Optional extraction worker-count override
        extract_executor: Extraction executor mode override
        canonicalize_workers: Optional canonicalization worker-count override
        canonicalize_executor: Canonicalization executor mode override
        graph_workers: Optional graph-generation worker-count override
        graph_executor: Graph-generation executor mode override
        expansion_output_dir: Optional directory for enriched graph JSON. When None, expansion is skipped.
        expansion_backend: Graph expansion pipeline name (default: ``"structural"``)
        expansion_workers: Optional graph-expansion worker-count override
        expansion_executor: Graph-expansion executor mode override

    Returns:
        Summary payload describing the produced artifacts
    """

    final_graph_output_dir = Path(output_dir)
    extracted_output_dir = Path(extracted_dir)
    canonical_output_dir = Path(canonical_dir)
    report_json_path = Path(report_json)
    report_md_path = Path(report_md) if report_md else None
    resolved_input_paths = [Path(input_path) for input_path in input_paths]
    missing_input_paths = [path for path in resolved_input_paths if not path.exists()]
    if missing_input_paths:
        missing_inputs_text = ", ".join(str(path) for path in missing_input_paths)
        raise RuntimeError(f"Input path does not exist: {missing_inputs_text}")

    # Validate all requested inputs up front so failed invocations never mutate
    # the user's ship corpus as a side effect of opt-in filtering.
    opt_out_filter = delete_non_opted_in_ship_files(
        resolved_input_paths,
        load_opt_in_author_names(opt_in_csv),
        cache_path=filter_cache_path,
    )
    filtered_input_paths = [path for path in resolved_input_paths if path.exists()]

    extract_manifest = run_extract(
        input_paths=filtered_input_paths,
        output_dir=extracted_output_dir,
        limit=limit,
        verbose=verbose,
        workers=extract_workers,
        executor=extract_executor,
    )
    extract_exit_code = int(extract_manifest.get("exit_code", 0 if extract_manifest.get("files_failed", 0) == 0 else 2))
    if extract_exit_code not in (0, 2):
        raise RuntimeError(f"Extraction failed with exit code {extract_exit_code}")

    # Always canonicalize before graph generation so the final graph output has
    # already gone through deduplication and preprocessing normalization.
    canonicalize_manifest = run_canonicalize(
        input_dir=extracted_output_dir,
        output_dir=canonical_output_dir,
        report_json=report_json_path,
        report_md=report_md_path,
        limit=limit,
        workers=canonicalize_workers,
        executor=canonicalize_executor,
    )
    graph_manifest = generate_all(
        canonical_output_dir,
        final_graph_output_dir,
        limit=limit,
        workers=graph_workers,
        executor=graph_executor,
    )

    expansion_result = None
    if expansion_output_dir is not None:
        from graph_expansion.structural import EXPANSION_NAME, expand_dir as expand_graphs

        expansion_output_path = Path(expansion_output_dir)
        if expansion_backend != EXPANSION_NAME:
            raise ValueError(
                f"Unknown graph expansion pipeline {expansion_backend!r}. Only {EXPANSION_NAME!r} is supported."
            )
        expansion_result = expand_graphs(
            input_dir=final_graph_output_dir,
            output_dir=expansion_output_path,
            workers=expansion_workers,
            executor=expansion_executor,
        )

    return {
        "inputs": [str(Path(input_path)) for input_path in input_paths],
        "final_graph_output_dir": str(final_graph_output_dir),
        "extracted_output_dir": str(extracted_output_dir),
        "canonical_output_dir": str(canonical_output_dir),
        "canonicalization_report_json": str(report_json_path),
        "canonicalization_report_md": str(report_md_path) if report_md_path else None,
        "extract_exit_code": extract_exit_code,
        "extraction": extract_manifest,
        "opt_out_filter": opt_out_filter,
        "canonicalization": canonicalize_manifest,
        "graphs": graph_manifest,
        "graph_expansion": expansion_result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preprocessing pipeline CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    # Derive a default filter cache path from the first input directory so that
    # unchanged ship files are not re-parsed on every pipeline run.
    first_input = Path(args.inputs[0])
    filter_cache_dir = first_input if first_input.is_dir() else first_input.parent
    filter_cache_path = filter_cache_dir / ".ship-filter-cache.json"
    payload = run_pipeline(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        extracted_dir=args.extracted_dir,
        canonical_dir=args.canonical_dir,
        report_json=args.report_json,
        report_md=args.report_md,
        limit=args.limit,
        verbose=args.verbose,
        opt_in_csv=args.opt_in_csv,
        filter_cache_path=filter_cache_path,
        extract_workers=args.extract_workers,
        extract_executor=args.extract_executor,
        canonicalize_workers=args.canonicalize_workers,
        canonicalize_executor=args.canonicalize_executor,
        graph_workers=args.graph_workers,
        graph_executor=args.graph_executor,
        expansion_output_dir=args.expansion_output_dir,
        expansion_backend=args.expansion_backend,
        expansion_workers=args.expansion_workers,
        expansion_executor=args.expansion_executor,
    )
    print(payload["graphs"]["ships_processed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
