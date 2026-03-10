"""Extract embedded ship JSON payloads from local `.ship.png` files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
from typing import Sequence

from common.files import iter_ship_png_files, output_name_for_ship_png
from common.logging import configure_logging
from common.cosmoteer import parse_ship_png


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
    return parser


def _extract_single(source_image_path: str, output_json_path: str) -> tuple[bool, str, str | None]:
    """Extract one embedded ship payload into a JSON file."""

    source_path = Path(source_image_path)
    destination_path = Path(output_json_path)

    try:
        ship_data = parse_ship_png(source_path)
        destination_path.write_text(
            json.dumps(ship_data, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        return True, str(source_path), None
    except Exception as exc:  # noqa: BLE001
        return False, str(source_path), repr(exc)


def run_extract(
    input_paths: Sequence[str | Path],
    output_dir: str | Path = "extracted_ship_data",
    verbose: bool = False,
) -> int:
    """Extract ship JSON files from local ship images.

    Args:
        input_paths: File and directory inputs containing `.ship.png` files
        output_dir: Destination directory for extracted JSON payloads
        verbose: When True, emit per-file progress logging

    Returns:
        Process exit code compatible with the legacy extractor:
        `0` on full success, `2` when any files fail to parse, `1` on setup failure
    """

    configure_logging(verbose)

    resolved_input_paths = [Path(path) for path in input_paths]
    missing_inputs = [path for path in resolved_input_paths if not path.exists()]
    if missing_inputs:
        for missing_input in missing_inputs:
            logging.error("Input path does not exist: %s", missing_input)
        return 1

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_images = list(iter_ship_png_files(resolved_input_paths))
    logging.info(
        "Found %d supported ship PNG file(s) across %d input path(s)",
        len(source_images),
        len(resolved_input_paths),
    )

    success_count = 0
    failure_count = 0
    worker_count = min(8, max(1, os.cpu_count() or 1), max(1, len(source_images)))
    logging.info("Using %d worker(s)", worker_count)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_source = {
            executor.submit(
                _extract_single,
                str(source_image_path),
                str(output_path / output_name_for_ship_png(source_image_path)),
            ): source_image_path
            for source_image_path in source_images
        }

        completed_count = 0
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
                    len(source_images),
                    source_image_path,
                    exc,
                )
                continue

            if ok:
                success_count += 1
                if verbose or completed_count == len(source_images) or completed_count % 100 == 0:
                    logging.info(
                        "[%d/%d] Wrote %s",
                        completed_count,
                        len(source_images),
                        output_path / output_name_for_ship_png(Path(source_path_text)),
                    )
            else:
                failure_count += 1
                logging.error(
                    "[%d/%d] Failed to parse %s: %s",
                    completed_count,
                    len(source_images),
                    source_path_text,
                    error_text,
                )

    logging.info(
        "Done. Parsed successfully: %d | Failed: %d",
        success_count,
        failure_count,
    )
    return 0 if failure_count == 0 else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the extraction CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_extract(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
