#!/usr/bin/env python3
"""Parse downloaded ship PNG files and write matching JSON outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ship_parser import parse_ship_png


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ship JSON payloads from downloaded ship PNG files (.ship.png and .ship__msg<digits>.png)."
    )
    parser.add_argument(
        "--input-dir",
        default="downloaded_ships",
        help="Directory containing downloaded ship PNG files (default: downloaded_ships)",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_ship_data",
        help="Directory to write .ship.json files (default: extracted_ship_data)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def is_supported_ship_png(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".ship.png") or (
        name.endswith(".png") and ".ship__msg" in name
    )


def output_name_for(source: Path) -> str:
    name = source.name
    if name.lower().endswith(".png"):
        return f"{name[:-len('.png')]}.json"
    return f"{name}.json"


def _extract_single(image_path: str, output_path: str) -> tuple[bool, str, str | None]:
    source = Path(image_path)
    destination = Path(output_path)

    try:
        ship_data = parse_ship_png(source)
        destination.write_text(
            json.dumps(ship_data, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        return True, str(source), None
    except Exception as exc:  # noqa: BLE001
        return False, str(source), repr(exc)


def run_extract(
    input_dir: str | Path = "downloaded_ships",
    output_dir: str | Path = "extracted_ship_data",
    verbose: bool = False,
) -> int:
    configure_logging(verbose)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logging.error("Input directory does not exist: %s", input_path)
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in input_path.rglob("*") if path.is_file() and is_supported_ship_png(path)
    )

    logging.info(
        "Found %d supported ship PNG file(s) under %s (.ship.png and .ship__msg<digits>.png)",
        len(images),
        input_path,
    )

    success = 0
    failures = 0

    worker_count = min(8, max(1, os.cpu_count() or 1), len(images))
    logging.info("Using %d worker(s)", worker_count)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_paths = {
            executor.submit(_extract_single, str(image_path), str(output_path / output_name_for(image_path))): image_path
            for image_path in images
        }

        completed = 0
        for future in as_completed(future_to_paths):
            completed += 1
            image_path = future_to_paths[future]

            try:
                ok, source, error = future.result()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logging.exception("[%d/%d] Worker crashed for %s: %s", completed, len(images), image_path, exc)
                continue

            if ok:
                success += 1
                if verbose or completed == len(images) or completed % 100 == 0:
                    logging.info("[%d/%d] Wrote %s", completed, len(images), output_path / output_name_for(Path(source)))
            else:
                failures += 1
                logging.error("[%d/%d] Failed to parse %s: %s", completed, len(images), source, error)

    logging.info("Done. Parsed successfully: %d | Failed: %d", success, failures)
    return 0 if failures == 0 else 2


def main() -> int:
    args = parse_args()
    return run_extract(input_dir=args.input_dir, output_dir=args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
