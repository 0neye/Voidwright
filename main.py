#!/usr/bin/env python3
"""Run the full ship data pipeline: download images, then extract ship JSON."""

from __future__ import annotations

import argparse
import logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ship pipeline: download .ship.png files and extract .ship.json data."
    )
    parser.add_argument(
        "--download-output-dir",
        default="downloaded_ships",
        help="Directory where downloaded .ship.png files are saved (default: downloaded_ships)",
    )
    parser.add_argument(
        "--extract-output-dir",
        default="extracted_ship_data",
        help="Directory where extracted .ship.json files are written (default: extracted_ship_data)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the Discord download step",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip the extraction step",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.skip_download and args.skip_extract:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            force=True,
        )
        logging.info("Both steps were skipped; nothing to do.")
        return 0

    if not args.skip_download:
        from scripts.download_ship_images import run_download

        download_code = run_download(output_dir=args.download_output_dir, verbose=args.verbose)
        if download_code != 0:
            return download_code

    if not args.skip_extract:
        from scripts.extract_ship_data import run_extract

        extract_code = run_extract(
            input_dir=args.download_output_dir,
            output_dir=args.extract_output_dir,
            verbose=args.verbose,
        )
        if extract_code != 0:
            return extract_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
