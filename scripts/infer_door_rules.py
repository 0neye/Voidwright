#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generators.markov.door_rules import Thresholds, infer_rules_from_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer reusable door-placement rules from the canonical ship corpus.")
    parser.add_argument("--input-dir", default="extracted_ship_data_canonical", help="Canonical deduped corpus directory")
    parser.add_argument(
        "--output",
        default="generators/markov/data/door-placement-rules.v2.json",
        help="Machine-readable output rules file",
    )
    parser.add_argument("--min-side-observations", type=int, default=2)
    parser.add_argument("--min-side-ratio", type=float, default=0.02)
    parser.add_argument("--min-pair-observations", type=int, default=2)
    parser.add_argument("--min-pair-ratio", type=float, default=0.02)
    args = parser.parse_args()

    payload = infer_rules_from_corpus(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        thresholds=Thresholds(
            min_side_observations=args.min_side_observations,
            min_side_ratio=args.min_side_ratio,
            min_pair_observations=args.min_pair_observations,
            min_pair_ratio=args.min_pair_ratio,
        ),
    )
    print(json.dumps(payload["validation"], indent=2))


if __name__ == "__main__":
    main()
