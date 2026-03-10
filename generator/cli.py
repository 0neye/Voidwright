"""CLI entrypoint for backend-agnostic ship generation."""

from __future__ import annotations

import argparse
from typing import Sequence

from generator.router import get_generator_backend, get_generator_backends


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level generator CLI parser."""

    parser = argparse.ArgumentParser(
        description="Generate encoded ship files from trained backend-specific models."
    )
    action_subparsers = parser.add_subparsers(dest="action", required=True)

    generate_parser = action_subparsers.add_parser(
        "generate",
        help="Generate ships with a specific backend",
    )
    backend_subparsers = generate_parser.add_subparsers(dest="backend", required=True)
    for backend in get_generator_backends().values():
        backend.register_generate_parser(backend_subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch generator requests to the selected backend."""

    parser = build_parser()
    args = parser.parse_args(argv)
    backend = get_generator_backend(args.backend)

    if args.action == "generate":
        return backend.run_generate(args)

    parser.error(f"Unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
