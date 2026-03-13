"""Extract embedded ship JSON payloads from local `.ship.png` files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import Sequence

import orjson

from common.files import (
    inputs_needing_regeneration,
    iter_ship_png_files,
    output_name_for_ship_png,
    prune_stale_json_outputs,
    write_output_version,
)
from common.logging import configure_logging
from common.cosmoteer import parse_ship_png
from .concurrency import (
    add_concurrency_arguments,
    create_executor_factory,
    resolve_executor_mode,
    resolve_worker_count,
)
from .relative_coords import apply_relative_coords_transform

__all__ = ["build_parser", "run_extract", "main"]

_EXTRACT_SCHEMA_VERSION = 2
_EXTRACT_SCHEMA_VERSION_KEY = "extract_schema_version"


class _MappedInputFile:
    """Proxy input record for regeneration checks with mapped output names."""

    def __init__(self, source_path: Path, output_name: str) -> None:
        self.source_path = source_path
        self.name = output_name

    def stat(self):  # noqa: ANN201
        """Use the source PNG mtime for staleness checks."""

        return self.source_path.stat()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the extraction stage."""

    parser = argparse.ArgumentParser(
        description="Extract ship JSON payloads from local .ship.png files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more input ship PNG files or directories",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_ship_data",
        help="Directory to write extracted ship JSON files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional non-destructive limit for partial extraction runs",
    )
    add_concurrency_arguments(
        parser,
        help_prefix="ship extraction",
    )
    return parser


def _extract_single(source_image_path: str, output_json_path: str) -> tuple[bool, str, str | None]:
    """Extract one embedded ship payload into a JSON file."""

    source_path = Path(source_image_path)
    destination_path = Path(output_json_path)

    try:
        ship_data = parse_ship_png(source_path)
        ship_data = apply_relative_coords_transform(ship_data)
        destination_path.write_text(
            orjson.dumps(
                ship_data,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            ).decode(),
            encoding="utf-8",
        )
        return True, str(source_path), None
    except Exception as exc:  # noqa: BLE001
        return False, str(source_path), repr(exc)


def run_extract(
    input_paths: Sequence[str | Path],
    output_dir: str | Path = "extracted_ship_data",
    verbose: bool = False,
    workers: int | None = None,
    executor: str = "auto",
    limit: int | None = None,
) -> dict:
    """Extract ship JSON files from local ship images.

    Args:
        input_paths: File and directory inputs containing `.ship.png` files
        output_dir: Destination directory for extracted JSON payloads
        verbose: When True, emit per-file progress logging
        workers: Optional worker-count override for extraction tasks
        executor: Executor mode override: `auto`, `thread`, or `process`
        limit: Optional non-destructive limit for partial extraction runs

    Returns:
        Manifest describing the extraction run, including schema version,
        discovered and processed ship counts, failure counts, and sample outputs
    """

    configure_logging(verbose)

    resolved_input_paths = [Path(path) for path in input_paths]
    missing_inputs = [path for path in resolved_input_paths if not path.exists()]
    if missing_inputs:
        for missing_input in missing_inputs:
            logging.error("Input path does not exist: %s", missing_input)
        raise FileNotFoundError("One or more extraction input paths do not exist")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_images = list(iter_ship_png_files(resolved_input_paths))
    logging.info(
        "Found %d supported ship PNG file(s) across %d input path(s)",
        len(source_images),
        len(resolved_input_paths),
    )
    if not source_images:
        logging.info("No ship PNG files found")
        return {
            "inputs": [str(path) for path in resolved_input_paths],
            "output_dir": str(output_path),
            "schema_version": _EXTRACT_SCHEMA_VERSION,
            "schema_version_key": _EXTRACT_SCHEMA_VERSION_KEY,
            "ship_files_discovered": 0,
            "ship_files_considered": 0,
            "ships_processed": 0,
            "ships_skipped": 0,
            "files_failed": 0,
            "limit": limit,
            "sample_outputs": [],
            "exit_code": 0,
        }

    # Map each discovered ship image to its expected JSON output filename so
    # incremental regeneration can use the version sentinel stored in the
    # output directory. The mapping remains deterministic because
    # iter_ship_png_files already yields paths in sorted order.
    image_output_pairs: list[tuple[Path, str]] = [
        (source_image, output_name_for_ship_png(source_image))
        for source_image in source_images
    ]
    if limit is not None:
        # Apply non-destructive limit before regeneration checks so partial
        # runs only touch a deterministic subset of inputs while leaving the
        # rest of the corpus untouched.
        image_output_pairs = image_output_pairs[:limit]

    output_name_to_source: dict[str, Path] = {
        output_name: source_image for source_image, output_name in image_output_pairs
    }
    source_to_output_name: dict[Path, str] = {
        source_image: output_name for source_image, output_name in image_output_pairs
    }
    candidate_inputs = [
        _MappedInputFile(source_image, output_name)
        for source_image, output_name in image_output_pairs
    ]

    # Use the shared incremental-regeneration helper against the mapped output
    # filenames. When the stored schema version differs from the current
    # extractor schema, every candidate output is regenerated on the next full
    # run; once a full run succeeds the version sentinel suppresses redundant
    # work until the schema changes again.
    files_to_process_outputs, skipped_output_files = inputs_needing_regeneration(
        candidate_inputs,
        output_path,
        current_version=_EXTRACT_SCHEMA_VERSION,
        version_key=_EXTRACT_SCHEMA_VERSION_KEY,
    )
    ships_skipped = len(skipped_output_files)
    if ships_skipped:
        logging.info(
            "Skipping %d up-to-date extracted ship JSON file(s) in %s",
            ships_skipped,
            output_path,
        )

    files_to_process = [
        output_name_to_source[output_file.name] for output_file in files_to_process_outputs
    ]

    success_count = 0
    failure_count = 0
    sample_outputs: list[str] = []

    if files_to_process:
        executor_mode = resolve_executor_mode("extract", executor)
        worker_count = resolve_worker_count(
            task_count=len(files_to_process),
            stage_name="extract",
            requested_workers=workers,
            requested_mode=executor,
        )
        executor_type = create_executor_factory(executor_mode)
        try:
            pool_ctx = executor_type(max_workers=worker_count)
        except (NotImplementedError, OSError, PermissionError):
            if executor == "auto" and executor_mode == "process":
                logging.warning(
                    "Process pool unavailable; falling back to thread executor for extraction"
                )
                executor_mode = "thread"
                pool_ctx = ThreadPoolExecutor(max_workers=worker_count)
            else:
                raise
        logging.info("Using %d %s worker(s)", worker_count, executor_mode)

        with pool_ctx as work_executor:
            future_to_source = {
                work_executor.submit(
                    _extract_single,
                    str(source_image_path),
                    str(output_path / source_to_output_name[source_image_path]),
                ): source_image_path
                for source_image_path in files_to_process
            }

            completed_count = 0
            total_to_process = len(files_to_process)
            for future in as_completed(future_to_source):
                completed_count += 1
                source_image_path = future_to_source[future]
                try:
                    ok, source_path_text, error_text = future.result()
                except Exception as exc:  # noqa: BLE001
                    failure_count += 1
                    logging.exception(
                        "[%d/%d] Worker crashed for %s: %s",
                        completed_count,
                        total_to_process,
                        source_image_path,
                        exc,
                    )
                    continue

                if ok:
                    success_count += 1
                    output_name = source_to_output_name[source_image_path]
                    if len(sample_outputs) < 10:
                        sample_outputs.append(output_name)
                    if verbose or completed_count == total_to_process or completed_count % 100 == 0:
                        logging.info(
                            "[%d/%d] Wrote %s",
                            completed_count,
                            total_to_process,
                            output_path / output_name,
                        )
                else:
                    failure_count += 1
                    logging.error(
                        "[%d/%d] Failed to parse %s: %s",
                        completed_count,
                        total_to_process,
                        source_path_text,
                        error_text,
                    )

    files_failed = failure_count

    # On full runs (no limit) prune stale extracted JSON outputs that are no
    # longer associated with any discovered ship image. Limited runs are
    # non-destructive and must not delete unrelated outputs.
    if limit is None:
        pruned_count = prune_stale_json_outputs(
            output_path,
            (output_name_for_ship_png(source_image) for source_image in source_images),
        )
        if pruned_count:
            logging.info(
                "Pruned %d stale extracted ship JSON file(s) from %s",
                pruned_count,
                output_path,
            )

        # Always persist the version sentinel on full runs. Failed extractions
        # produce no output file, so there is no stale artifact to hide — the
        # failed source will simply be retried on the next full run.
        write_output_version(
            output_path,
            _EXTRACT_SCHEMA_VERSION_KEY,
            _EXTRACT_SCHEMA_VERSION,
        )

    logging.info(
        "Done. Parsed successfully: %d | Failed: %d | Skipped (up-to-date): %d",
        success_count,
        failure_count,
        ships_skipped,
    )

    manifest = {
        "inputs": [str(path) for path in resolved_input_paths],
        "output_dir": str(output_path),
        "schema_version": _EXTRACT_SCHEMA_VERSION,
        "schema_version_key": _EXTRACT_SCHEMA_VERSION_KEY,
        "ship_files_discovered": len(source_images),
        "ship_files_considered": len(image_output_pairs),
        "ships_processed": success_count,
        "ships_skipped": ships_skipped,
        "files_failed": files_failed,
        "limit": limit,
        "sample_outputs": sample_outputs,
        "exit_code": 0 if files_failed == 0 else 2,
    }
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the extraction CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_extract(
            input_paths=args.inputs,
            output_dir=args.output_dir,
            verbose=args.verbose,
            workers=args.workers,
            executor=args.executor,
            limit=args.limit,
        )
    except FileNotFoundError:
        return 1

    return int(manifest.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
