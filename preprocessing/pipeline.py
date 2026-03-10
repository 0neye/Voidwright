"""End-to-end preprocessing pipeline from local `.ship.png` to graph JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Sequence

from .canonicalize import run_canonicalize
from .extract import run_extract
from .graphs import generate_all


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
        default="SHIP_CANONICALIZATION_REPORT.md",
        help="Canonicalization markdown report path when canonical outputs are persisted",
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
    return parser


def run_pipeline(
    input_paths: Sequence[str | Path],
    output_dir: str | Path = "generated_ship_graphs_canonical",
    write_extracted_dir: str | Path | None = None,
    write_canonical_dir: str | Path | None = None,
    report_json: str | Path = "out/ship_canonicalization_report.json",
    report_md: str | Path = "SHIP_CANONICALIZATION_REPORT.md",
    limit: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run the local ship preprocessing pipeline.

    Args:
        input_paths: Local `.ship.png` files or directories to preprocess
        output_dir: Final graph JSON output directory
        write_extracted_dir: Optional persistent extracted JSON directory
        write_canonical_dir: Optional persistent canonical JSON directory
        report_json: Canonicalization JSON report path when persisting outputs
        report_md: Canonicalization markdown report path when persisting outputs
        limit: Optional limit for the graph-generation stage
        verbose: When True, enable verbose extraction logging

    Returns:
        Summary payload describing the produced artifacts
    """

    final_graph_output_dir = Path(output_dir)
    persistent_extracted_dir = Path(write_extracted_dir) if write_extracted_dir else None
    persistent_canonical_dir = Path(write_canonical_dir) if write_canonical_dir else None
    report_json_path = Path(report_json)
    report_md_path = Path(report_md)

    with tempfile.TemporaryDirectory(prefix="ship_preprocess_") as temp_dir:
        temp_root = Path(temp_dir)
        extracted_dir = persistent_extracted_dir or (temp_root / "extracted")
        canonical_dir = persistent_canonical_dir or (temp_root / "canonical")
        if persistent_extracted_dir is None:
            extracted_dir.mkdir(parents=True, exist_ok=True)
        if persistent_canonical_dir is None:
            canonical_dir.mkdir(parents=True, exist_ok=True)

        extract_exit_code = run_extract(
            input_paths=input_paths,
            output_dir=extracted_dir,
            verbose=verbose,
        )
        if extract_exit_code not in (0, 2):
            raise RuntimeError(f"Extraction failed with exit code {extract_exit_code}")

        # Always canonicalize before graph generation so the final graph output has
        # already gone through deduplication and preprocessing normalization.
        canonicalize_manifest = run_canonicalize(
            input_dir=extracted_dir,
            output_dir=canonical_dir,
            report_json=report_json_path,
            report_md=report_md_path,
        )
        graph_manifest = generate_all(canonical_dir, final_graph_output_dir, limit=limit)

        return {
            "inputs": [str(Path(input_path)) for input_path in input_paths],
            "final_graph_output_dir": str(final_graph_output_dir),
            "extracted_output_dir": str(extracted_dir) if persistent_extracted_dir else None,
            "canonical_output_dir": str(canonical_dir) if persistent_canonical_dir else None,
            "canonicalization_report_json": str(report_json_path),
            "canonicalization_report_md": str(report_md_path),
            "extract_exit_code": extract_exit_code,
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
    )
    print(payload["graphs"]["ships_processed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
