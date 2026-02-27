#!/usr/bin/env python3
"""Parse downloaded `.ship.png` files and write matching `.ship.json` outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
        description="Extract ship JSON payloads from downloaded .ship.png files."
    )
    parser.add_argument(
        "--input-dir",
        default="downloaded_ships",
        help="Directory containing downloaded .ship.png files (default: downloaded_ships)",
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


def output_name_for(source: Path) -> str:
    name = source.name
    if name.lower().endswith(".ship.png"):
        base = name[: -len(".ship.png")]
    else:
        base = source.stem
    return f"{base}.ship.json"


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
        path for path in input_path.rglob("*") if path.is_file() and path.name.lower().endswith(".ship.png")
    )

    logging.info("Found %d .ship.png file(s) under %s", len(images), input_path)

    success = 0
    failures = 0

    for index, image_path in enumerate(images, start=1):
        image_output_path = output_path / output_name_for(image_path)
        logging.info("[%d/%d] Parsing %s", index, len(images), image_path)

        try:
            ship_data = parse_ship_png(image_path)
            image_output_path.write_text(
                json.dumps(ship_data, indent=2, sort_keys=True, ensure_ascii=True),
                encoding="utf-8",
            )
            success += 1
            logging.info("Wrote %s", image_output_path)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logging.exception("Failed to parse %s: %s", image_path, exc)

    logging.info("Done. Parsed successfully: %d | Failed: %d", success, failures)
    return 0 if failures == 0 else 2


def main() -> int:
    args = parse_args()
    return run_extract(input_dir=args.input_dir, output_dir=args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
