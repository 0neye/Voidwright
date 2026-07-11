"""CLI entrypoint for the corpus filter pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from corpus.filter import run_filter, validate_corpus_has_expansion
from corpus.rules.max_size import MaxSizeRule
from corpus.rules.require_crew_rooms import RequireCrewRoomsRule
from corpus.rules.require_reachable_reactor import RequireReachableReactorRule
from corpus.rules.vanilla_only import VanillaOnlyRule

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the corpus filter CLI parser."""
    parser = argparse.ArgumentParser(
        description="Filter a generated ship graph corpus by applying rule-based checks."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Input directory containing generated ship graph JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Output directory for accepted ship graph JSON files and manifest.",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        metavar="N",
        help="Reject ships with more than N parts.",
    )
    parser.add_argument(
        "--max-occupied-cells",
        type=int,
        default=None,
        metavar="N",
        help="Reject ships whose occupied 2x-cell count exceeds N.",
    )
    parser.add_argument(
        "--require-crew-rooms",
        action="store_true",
        default=False,
        help="Reject ships that contain no crew rooms.",
    )
    parser.add_argument(
        "--require-reachable-reactor",
        action="store_true",
        default=False,
        help=(
            "Reject ships with crew rooms but no reachable reactor. "
            "Requires expansion graphs in the input corpus."
        ),
    )
    parser.add_argument(
        "--vanilla-only",
        action="store_true",
        default=False,
        help="Reject ships that contain any non-vanilla (modded) part IDs.",
    )
    parser.add_argument(
        "--no-rejections-log",
        action="store_true",
        default=False,
        help="Skip writing rejections.jsonl even when ships are rejected.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    return parser


def _build_active_ruleset(args: argparse.Namespace) -> list:
    """Construct the ordered list of active rules from parsed CLI args."""
    rules = []
    if args.max_parts is not None or args.max_occupied_cells is not None:
        rules.append(
            MaxSizeRule(
                max_parts=args.max_parts,
                max_occupied_cells=args.max_occupied_cells,
            )
        )
    if args.require_crew_rooms:
        rules.append(RequireCrewRoomsRule())
    if args.require_reachable_reactor:
        rules.append(RequireReachableReactorRule())
    if args.vanilla_only:
        rules.append(VanillaOnlyRule())
    return rules


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus filter pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    rules = _build_active_ruleset(args)
    if not rules:
        print(
            "Warning: no filter rules enabled. All ships will be accepted.",
            file=sys.stderr,
        )

    # Fail fast if require_reachable_reactor is enabled but corpus lacks expansion.
    if args.require_reachable_reactor:
        try:
            validate_corpus_has_expansion(input_dir)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    result = run_filter(
        input_dir,
        output_dir,
        rules,
        write_rejections_log=not args.no_rejections_log,
    )

    kept_pct = (
        100 * result.ships_kept // result.ships_scanned
        if result.ships_scanned
        else 0
    )
    print(
        f"Scanned {result.ships_scanned} ships: "
        f"kept {result.ships_kept} ({kept_pct}%), "
        f"rejected {result.ships_rejected}."
    )
    if result.rejections_by_rule:
        for rule_name, count in result.rejections_by_rule.items():
            if count:
                print(f"  {rule_name}: {count} rejected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
