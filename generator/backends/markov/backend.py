"""Markov backend adapter for the generic generator module."""

from __future__ import annotations

import argparse
import orjson
from pathlib import Path

from markov.inputs import (
    load_allowlist,
    load_requirements,
    load_seed_parts_from_json,
    load_seed_parts_from_png,
)
from generator.base import GeneratorBackend, add_visualization_arguments

__all__ = ["MarkovGeneratorBackend"]
from generator.backends.markov.export import export_ship_png
from markov.model import (
    GenerationConfig,
    RelativeMarkovModel,
    iter_vanilla_parts_from_ship,
)
from visualizer import (
    VisualizationRecorder,
    ensure_ffmpeg_available,
    load_part_icon_library,
    render_events_to_mp4,
)


class MarkovGeneratorBackend(GeneratorBackend):
    """Generator adapter that preserves the existing Markov sampling behavior."""

    name = "markov"

    def register_generate_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the Markov generation parser."""

        parser = backend_subparsers.add_parser(
            self.name,
            help="Generate encoded ship files with the Markov backend",
        )
        parser.add_argument(
            "--model",
            type=Path,
            default=Path("models/markov/markov-model.v2.json"),
            help="Model artifact to load",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Directory where encoded .ship.png files will be written",
        )
        parser.add_argument(
            "--json-output-dir",
            type=Path,
            default=None,
            help="Optional directory for generated JSON diagnostics",
        )
        parser.add_argument("--count", type=int, default=1)
        parser.add_argument("--max-parts", type=int, default=250)
        parser.add_argument("--max-attempts", type=int, default=3000)
        parser.add_argument("--max-resample-per-step", type=int, default=32)
        parser.add_argument("--bounds-min-x", type=int, default=-64)
        parser.add_argument("--bounds-max-x", type=int, default=64)
        parser.add_argument("--bounds-min-y", type=int, default=-64)
        parser.add_argument("--bounds-max-y", type=int, default=64)
        parser.add_argument("--seed", type=int, default=1337)
        parser.add_argument(
            "--no-validate",
            action="store_true",
            default=False,
            help="Skip roundtrip PNG validation during export",
        )
        parser.add_argument(
            "--allowlist",
            nargs="+",
            metavar="PART_ID",
            default=None,
            help="Restrict generation to these part IDs",
        )
        parser.add_argument(
            "--allowlist-file",
            type=Path,
            default=None,
            help="Path to a file containing one allowed part ID per line",
        )
        parser.add_argument(
            "--mirror-symmetry",
            action="store_true",
            default=False,
            help="Enable strict left-right mirror symmetry during generation",
        )
        parser.add_argument(
            "--require",
            nargs=2,
            metavar=("PART_ID", "COUNT"),
            action="append",
            default=None,
            help="Require at least COUNT copies of PART_ID in the final ship",
        )
        parser.add_argument(
            "--requirements-file",
            type=Path,
            default=None,
            help="Optional JSON or line-based requirements file",
        )
        parser.add_argument(
            "--seed-json",
            type=Path,
            default=None,
            help="Optional generated or extracted ship JSON to seed generation",
        )
        parser.add_argument(
            "--seed-png",
            type=Path,
            default=None,
            help="Optional .ship.png file to seed generation",
        )
        add_visualization_arguments(parser)

    def run_generate(self, args: argparse.Namespace) -> int:
        """Run Markov generation and export encoded ship files."""

        if args.seed_json is not None and args.seed_png is not None:
            print("[generator:markov] ERROR: --seed-json and --seed-png are mutually exclusive")
            return 1

        model = RelativeMarkovModel.load(args.model)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.json_output_dir is not None:
            args.json_output_dir.mkdir(parents=True, exist_ok=True)
        visualizations_output_dir = args.output_dir / "visualizations"
        icon_library = None
        if args.visualize:
            try:
                ensure_ffmpeg_available()
                icon_library = load_part_icon_library(
                    icons_root=args.icons_root,
                    game_root=args.game_root,
                )
                visualizations_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"[generator:markov] ERROR: unable to initialize visualization: {exc}")
                return 1

        allowlist = load_allowlist(args.allowlist, args.allowlist_file)
        requirements = load_requirements(args.require, args.requirements_file)

        seed_parts = None
        if args.seed_json is not None:
            seed_parts = load_seed_parts_from_json(args.seed_json)
        elif args.seed_png is not None:
            seed_parts = load_seed_parts_from_png(args.seed_png, iter_vanilla_parts_from_ship)

        for sample_index in range(args.count):
            visualization_recorder = (
                VisualizationRecorder(sample_index)
                if args.visualize and icon_library is not None
                else None
            )
            generation_config = GenerationConfig(
                max_parts=args.max_parts,
                max_attempts=args.max_attempts,
                max_resample_per_step=args.max_resample_per_step,
                bounds_min_x=args.bounds_min_x,
                bounds_max_x=args.bounds_max_x,
                bounds_min_y=args.bounds_min_y,
                bounds_max_y=args.bounds_max_y,
                rng_seed=args.seed + sample_index,
                part_allowlist=allowlist,
                mirror_symmetry=args.mirror_symmetry,
                part_requirements=requirements,
            )
            try:
                payload = model.generate(
                    generation_config,
                    seed_parts=seed_parts,
                    event_sink=visualization_recorder,
                )
            except RuntimeError as exc:
                print(f"[generator:markov] sample-{sample_index:03d} FAILED: {exc}")
                continue

            if args.json_output_dir is not None:
                json_output_path = args.json_output_dir / f"sample-{sample_index:03d}.json"
                json_output_path.write_text(
                    orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode() + "\n",
                    encoding="utf-8",
                )

            png_output_path = args.output_dir / f"sample-{sample_index:03d}.ship.png"
            export_result = export_ship_png(
                payload,
                png_output_path,
                name=f"gen-{sample_index:03d}",
                validate=not args.no_validate,
            )

            stats = payload["stats"]
            export_status = (
                "OK"
                if export_result.get("valid") is True
                else ("SKIPPED" if export_result.get("valid") is None else "WARN")
            )
            print(
                "[generator:markov] "
                f"sample-{sample_index:03d}: stop={stats['stop_reason']} "
                f"parts={stats['parts_generated']} export={export_status} "
                f"output={png_output_path}"
            )

            if visualization_recorder is not None and icon_library is not None:
                mp4_output_path = visualizations_output_dir / f"sample-{sample_index:03d}.mp4"
                try:
                    render_events_to_mp4(
                        visualization_recorder.events,
                        mp4_output_path,
                        icon_library=icon_library,
                        fps=args.visualization_fps,
                    )
                except Exception as exc:
                    print(
                        "[generator:markov] "
                        f"sample-{sample_index:03d} visualization FAILED: {exc}"
                    )
                    return 1

        return 0
