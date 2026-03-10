"""Corpus-derived door-rule generation for preprocessed ship data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .door_rules_engine import Thresholds, infer_rules_from_corpus


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for door-rule inference."""

    parser = argparse.ArgumentParser(
        description="Infer reusable door-placement rules from the canonical ship corpus."
    )
    parser.add_argument(
        "--input-dir",
        default="extracted_ship_data_canonical",
        help="Canonical deduped corpus directory",
    )
    parser.add_argument(
        "--output",
        default="out/preprocessing/door-placement-rules.v2.json",
        help="Machine-readable output rules file",
    )
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--min-side-ratio", type=float, default=0.02)
    parser.add_argument("--min-pair-observations", type=int, default=2)
    parser.add_argument("--min-pair-ratio", type=float, default=0.02)
    return parser


def run_infer_door_rules(
    input_dir: str | Path = "extracted_ship_data_canonical",
    output_path: str | Path = "out/preprocessing/door-placement-rules.v2.json",
    min_side_observations: int = 2,
    min_side_ratio: float = 0.02,
    min_pair_observations: int = 2,
    min_pair_ratio: float = 0.02,
) -> dict:
    """Infer machine-readable door rules from a canonical ship corpus."""

    return infer_rules_from_corpus(
        input_dir=Path(input_dir),
        output_path=Path(output_path),
        thresholds=Thresholds(
            min_side_observations=min_side_observations,
            min_side_ratio=min_side_ratio,
            min_pair_observations=min_pair_observations,
            min_pair_ratio=min_pair_ratio,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the door-rule inference CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_infer_door_rules(
        input_dir=args.input_dir,
        output_path=args.output,
        min_side_observations=args.min_side_observations,
        min_side_ratio=args.min_side_ratio,
        min_pair_observations=args.min_pair_observations,
        min_pair_ratio=args.min_pair_ratio,
    )
    print(json.dumps(payload["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
