"""Markov backend adapter for the generic training module."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson

from markov.inputs import load_allowlist
from markov.model import (
    TrainingConfig,
    build_model_from_corpus,
    build_model_from_graph_corpus,
    validate_relative_placement_assumptions,
)
from training.base import TrainingBackend

__all__ = ["MarkovTrainingBackend"]


class MarkovTrainingBackend(TrainingBackend):
    """Backend adapter that preserves the existing Markov training behavior."""

    name = "markov"

    def register_build_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the Markov build parser."""

        parser = backend_subparsers.add_parser(
            self.name,
            help="Train the Markov backend from graph or canonical corpora",
        )
        parser.add_argument(
            "--graph-input-dir",
            type=Path,
            default=None,
            help="Directory containing preprocessing graph JSON artifacts",
        )
        parser.add_argument(
            "--input-dir",
            type=Path,
            default=None,
            help="Optional canonical ship JSON directory for legacy/raw training",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("models/markov/markov-model.v2.json"),
            help="Path where the trained model JSON will be written",
        )
        parser.add_argument("--markov-order", type=int, default=2)
        parser.add_argument("--min-parts-per-ship", type=int, default=2)
        parser.add_argument("--max-parts-per-ship", type=int, default=5000)
        parser.add_argument("--anchor-window", type=int, default=128)
        parser.add_argument(
            "--validation-output",
            type=Path,
            default=None,
            help="Optional path for coordinate validation output when --input-dir is available",
        )
        parser.add_argument(
            "--allowlist",
            nargs="+",
            metavar="PART_ID",
            default=None,
            help="Restrict training to these part IDs",
        )
        parser.add_argument(
            "--allowlist-file",
            type=Path,
            default=None,
            help="Path to a file containing one allowed part ID per line",
        )

    def register_validate_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        """Register the Markov validation parser."""

        parser = backend_subparsers.add_parser(
            self.name,
            help="Validate Markov coordinate assumptions against canonical ship JSON",
        )
        parser.add_argument(
            "--input-dir",
            type=Path,
            required=True,
            help="Canonical ship JSON directory used for validation",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("models/markov/coordinate-validation.v2.json"),
            help="Validation report output path",
        )
        parser.add_argument("--sample-limit", type=int, default=None)

    def run_build(self, args: argparse.Namespace) -> int:
        """Run Markov training using graph or canonical corpora."""

        if args.graph_input_dir is not None and args.input_dir is not None:
            print("[training:markov] ERROR: --graph-input-dir and --input-dir are mutually exclusive")
            return 1
        if args.graph_input_dir is None and args.input_dir is None:
            print("[training:markov] ERROR: one of --graph-input-dir or --input-dir is required")
            return 1

        allowlist = load_allowlist(args.allowlist, args.allowlist_file)
        training_config = TrainingConfig(
            markov_order=args.markov_order,
            min_parts_per_ship=args.min_parts_per_ship,
            max_parts_per_ship=args.max_parts_per_ship,
            anchor_window=args.anchor_window,
            part_allowlist=allowlist,
        )

        if args.graph_input_dir is not None:
            model = build_model_from_graph_corpus(args.graph_input_dir, training_config)
            print(f"[training:markov] training from graph corpus: {args.graph_input_dir}")
        else:
            model = build_model_from_corpus(args.input_dir, training_config)
            print(f"[training:markov] training from canonical corpus: {args.input_dir}")

        model.save(args.output)
        print(f"[training:markov] model saved to {args.output}")

        if args.validation_output is not None:
            if args.input_dir is None:
                print(
                    "[training:markov] NOTE: --validation-output requires --input-dir and is skipped for graph-only training"
                )
            else:
                validation_payload = validate_relative_placement_assumptions(args.input_dir)
                args.validation_output.parent.mkdir(parents=True, exist_ok=True)
                args.validation_output.write_text(
                    orjson.dumps(
                        validation_payload,
                        option=orjson.OPT_INDENT_2,
                    ).decode()
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[training:markov] validation written to {args.validation_output}")
        return 0

    def run_validate(self, args: argparse.Namespace) -> int:
        """Run Markov coordinate validation."""

        validation_payload = validate_relative_placement_assumptions(
            args.input_dir,
            sample_limit=args.sample_limit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            orjson.dumps(
                validation_payload,
                option=orjson.OPT_INDENT_2,
            ).decode()
            + "\n",
            encoding="utf-8",
        )
        print(f"[training:markov] validation written to {args.output}")
        return 0
