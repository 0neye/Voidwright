"""CLI entrypoint for graph expansion."""

from __future__ import annotations

import argparse
from typing import Sequence

from graph_expansion.structural import EXPANSION_NAME, build_parser as build_expand_parser, run_from_args

__all__ = ["build_parser", "main"]



def build_parser() -> argparse.ArgumentParser:
    """Build the top-level graph expansion CLI parser.

    The canonical shape is now:
        graph-expansion expand --input-dir ... --output-dir ...

    For compatibility, a legacy positional pipeline name is still accepted:
        graph-expansion expand structural --input-dir ...
    """

    parser = argparse.ArgumentParser(
        description="Enrich preprocessing graph JSON with virtual nodes and cross-edges."
    )
    action_subparsers = parser.add_subparsers(dest="action", required=True)

    expand_subparser = action_subparsers.add_parser(
        "expand",
        help="Expand graph JSON artifacts using the structural pass pipeline",
    )
    expand_subparser.add_argument(
        "legacy_pipeline_name",
        nargs="?",
        help=f"Legacy optional pipeline name. Only {EXPANSION_NAME!r} is supported.",
    )
    build_expand_parser(expand_subparser)

    return parser



def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch graph expansion requests."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action == "expand":
        legacy_pipeline_name = getattr(args, "legacy_pipeline_name", None)
        if legacy_pipeline_name not in (None, EXPANSION_NAME):
            parser.error(
                f"Unknown graph expansion pipeline {legacy_pipeline_name!r}. Only {EXPANSION_NAME!r} is supported."
            )
        return run_from_args(args)

    parser.error(f"Unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
