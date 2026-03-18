"""CLI entrypoint for static ship visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from visualizer.icons import load_part_icon_library
from visualizer.router import get_static_backend, get_static_backends
from visualizer.static_render import SUBCELL_SIZE, load_ship_for_visualization

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level visualizer CLI parser."""
    parser = argparse.ArgumentParser(
        description="Render static visualizations of ship graph structures."
    )
    action_subparsers = parser.add_subparsers(dest="action", required=True)

    render_parser = action_subparsers.add_parser(
        "render",
        help="Render a static visualization using a specific backend",
    )
    backend_subparsers = render_parser.add_subparsers(dest="backend", required=True)

    for backend in get_static_backends().values():
        backend_parser = backend_subparsers.add_parser(
            backend.name,
            help=f"Render {backend.name} visualization",
        )
        backend_parser.add_argument(
            "--input",
            dest="inputs",
            type=Path,
            metavar="FILE",
            nargs="+",
            required=True,
            help="One or more .ship.png files to visualize",
        )
        backend_parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help=f"Output directory for rendered PNGs (default: {backend.default_output_dir})",
        )
        backend_parser.add_argument(
            "--icons-root",
            type=Path,
            default=None,
            help="Optional path to Cosmoteer Terran part icons root (Data/ships/terran)",
        )
        backend_parser.add_argument(
            "--game-root",
            type=Path,
            default=None,
            help="Optional path to Cosmoteer install root",
        )
        backend.register_parser(backend_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch visualization requests to the selected backend."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action == "render":
        backend = get_static_backend(args.backend)
        output_dir = args.output_dir if args.output_dir is not None else Path(backend.default_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / "_work"

        try:
            icon_library = load_part_icon_library(
                icons_root=args.icons_root,
                game_root=args.game_root,
                cell_size=SUBCELL_SIZE * 2,
            )
        except FileNotFoundError as exc:
            print(f"[visualizer] ERROR: {exc}")
            return 1

        for input_path in args.inputs:
            if not input_path.exists():
                print(f"[visualizer] WARNING: input file not found, skipping: {input_path}")
                continue
            try:
                expanded_data, flip_map = load_ship_for_visualization(input_path, work_dir)
            except Exception as exc:
                print(f"[visualizer] WARNING: failed to load {input_path.name}: {exc}")
                continue
            try:
                ship_name = input_path.name
                output_path = backend.render_ship(
                    ship_name, expanded_data, flip_map, output_dir, icon_library, args
                )
                print(output_path)
            except Exception as exc:
                print(f"[visualizer] WARNING: render failed for {input_path.name}: {exc}")
                continue

        return 0

    parser.error(f"Unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
