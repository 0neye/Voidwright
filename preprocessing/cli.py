"""CLI entrypoint for preprocessing workflows."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import canonicalize, door_rules, extract, graphs, pipeline

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level preprocessing CLI parser."""

    parser = argparse.ArgumentParser(
        description="Preprocessing utilities for local ship-image corpora."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run local .ship.png inputs through the full preprocessing pipeline",
    )
    for action in pipeline.build_parser()._actions[1:]:
        pipeline_parser._add_action(action)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract embedded ship JSON from local .ship.png inputs",
    )
    for action in extract.build_parser()._actions[1:]:
        extract_parser._add_action(action)

    canonicalize_parser = subparsers.add_parser(
        "canonicalize",
        help="Canonicalize and deduplicate extracted JSON artifacts",
    )
    for action in canonicalize.build_parser()._actions[1:]:
        canonicalize_parser._add_action(action)

    graphs_parser = subparsers.add_parser(
        "graphs",
        help="Generate graph JSON artifacts from canonical ship JSON",
    )
    for action in graphs.build_parser()._actions[1:]:
        graphs_parser._add_action(action)

    door_rules_parser = subparsers.add_parser(
        "door-rules",
        help="Infer reusable door rules from a canonical corpus",
    )
    for action in door_rules.build_parser()._actions[1:]:
        door_rules_parser._add_action(action)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch preprocessing commands."""

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    command_argv = raw_argv[1:]

    if args.command == "pipeline":
        return pipeline.main(command_argv)
    if args.command == "extract":
        return extract.main(command_argv)
    if args.command == "canonicalize":
        return canonicalize.main(command_argv)
    if args.command == "graphs":
        return graphs.main(command_argv)
    if args.command == "door-rules":
        return door_rules.main(command_argv)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
