"""CLI entrypoint for backend-agnostic graph expansion."""

from __future__ import annotations

import argparse
from typing import Sequence

from graph_expansion.router import get_expansion_backend, get_expansion_backends

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level graph expansion CLI parser."""

    parser = argparse.ArgumentParser(
        description="Enrich preprocessing graph JSON with virtual nodes and cross-edges."
    )
    action_subparsers = parser.add_subparsers(dest="action", required=True)

    expand_subparser = action_subparsers.add_parser(
        "expand",
        help="Expand graph JSON artifacts using a specific backend",
    )
    expand_backend_subparsers = expand_subparser.add_subparsers(dest="backend", required=True)

    for backend in get_expansion_backends().values():
        backend.register_expand_parser(expand_backend_subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch graph expansion requests to the selected backend."""

    parser = build_parser()
    args = parser.parse_args(argv)
    backend = get_expansion_backend(args.backend)

    if args.action == "expand":
        return backend.run_expand(args)

    parser.error(f"Unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
