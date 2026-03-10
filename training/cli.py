"""CLI entrypoint for backend-agnostic training."""

from __future__ import annotations

import argparse
from typing import Sequence

from training.router import get_training_backend, get_training_backends


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level training CLI parser."""

    parser = argparse.ArgumentParser(
        description="Train backend-specific ship models from preprocessing outputs."
    )
    action_subparsers = parser.add_subparsers(dest="action", required=True)

    build_parser = action_subparsers.add_parser(
        "build",
        help="Build a model for a specific training backend",
    )
    validate_parser = action_subparsers.add_parser(
        "validate",
        help="Run backend-specific validation",
    )

    build_backend_subparsers = build_parser.add_subparsers(dest="backend", required=True)
    validate_backend_subparsers = validate_parser.add_subparsers(dest="backend", required=True)

    for backend in get_training_backends().values():
        backend.register_build_parser(build_backend_subparsers)
        backend.register_validate_parser(validate_backend_subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch training requests to the selected backend."""

    parser = build_parser()
    args = parser.parse_args(argv)
    backend = get_training_backend(args.backend)

    if args.action == "build":
        return backend.run_build(args)
    if args.action == "validate":
        return backend.run_validate(args)

    parser.error(f"Unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
