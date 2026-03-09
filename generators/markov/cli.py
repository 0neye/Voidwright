from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .model import (
    GenerationConfig,
    RelativeMarkovModel,
    TrainingConfig,
    build_model_from_corpus,
    build_model_from_graph_corpus,
    iter_vanilla_parts_from_ship,
    validate_relative_placement_assumptions,
)


# ── allowlist helpers ─────────────────────────────────────────────────────────


def _load_allowlist(allowlist_arg: Optional[list], allowlist_file_arg: Optional[Path]) -> Optional[frozenset]:
    """Combine --allowlist and --allowlist-file into a frozenset, or return None."""
    ids: set[str] = set()
    if allowlist_arg:
        ids.update(a.strip() for a in allowlist_arg if a.strip())
    if allowlist_file_arg is not None:
        text = allowlist_file_arg.read_text()
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # Support JSON array or plain line-delimited
                if line.startswith("["):
                    ids.update(json.loads(line))
                else:
                    ids.add(line)
    return frozenset(ids) if ids else None


# ── requirements helpers ───────────────────────────────────────────────────────


def _load_requirements(require_arg: Optional[list], requirements_file_arg: Optional[Path]) -> Optional[dict]:
    """Parse part requirements into {part_id: min_count}.

    --require accepts repeated ``PART_ID COUNT`` pairs.
    --requirements-file accepts a JSON object ``{"part_id": count, ...}``
    or a plain-text file with one ``PART_ID COUNT`` per line (# comments ok).
    """
    reqs: dict[str, int] = {}
    if require_arg:
        # Each entry is [part_id, count_str]
        for part_id, count_str in require_arg:
            count = int(count_str)
            if count <= 0:
                raise ValueError(f"requirement count must be > 0, got {count} for {part_id}")
            reqs[part_id.strip()] = max(reqs.get(part_id.strip(), 0), count)
    if requirements_file_arg is not None:
        text = requirements_file_arg.read_text().strip()
        if text.startswith("{"):
            data = json.loads(text)
            for pid, cnt in data.items():
                reqs[pid.strip()] = max(reqs.get(pid.strip(), 0), int(cnt))
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"requirements file: expected 'PART_ID COUNT', got: {line!r}")
                pid, cnt = parts
                reqs[pid.strip()] = max(reqs.get(pid.strip(), 0), int(cnt))
    return reqs if reqs else None


# ── seed helpers ──────────────────────────────────────────────────────────────


def _load_seed_parts_from_json(path: Path) -> list:
    """Load seed parts from a generated ship JSON or Cosmoteer ship JSON."""
    data = json.loads(path.read_text())
    # Generated format: {"parts": [{part_id, rotation, x, y}, ...]}
    if "parts" in data and isinstance(data["parts"], list):
        raw = data["parts"]
        if raw and isinstance(raw[0], dict) and "part_id" in raw[0]:
            return raw  # already in our format
        # Cosmoteer format nested under "parts" key (unlikely but handle gracefully)
        return [
            {"part_id": p["ID"], "rotation": int(p.get("Rotation", 0)), "x": int(p["Location"][0]), "y": int(p["Location"][1])}
            for p in raw
            if isinstance(p, dict) and "ID" in p and "Location" in p
        ]
    # Cosmoteer extracted format: {"Parts": [{ID, Location, Rotation}, ...]}
    if "Parts" in data:
        return [
            {"part_id": p["ID"], "rotation": int(p.get("Rotation", 0)), "x": int(p["Location"][0]), "y": int(p["Location"][1])}
            for p in data["Parts"]
            if isinstance(p, dict) and "ID" in p and isinstance(p.get("Location"), list) and len(p["Location"]) == 2
        ]
    raise ValueError(f"Could not parse seed parts from {path}: unrecognized format (expected 'parts' or 'Parts' key)")


def _load_seed_parts_from_png(path: Path) -> list:
    """Load seed parts from a .ship.png by parsing its embedded payload."""
    import sys
    import os
    # Ensure ship_parser is importable
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ship_parser.cosmoteer_ship_parser import parse_ship_png
    ship_data = parse_ship_png(path)
    parts = iter_vanilla_parts_from_ship(ship_data)
    return [{"part_id": p.part_id, "rotation": p.rotation, "x": p.x, "y": p.y} for p in parts]


# ── argument parser ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or sample the first-pass vanilla-only relative Markov ship generator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build ──
    build = subparsers.add_parser("build", help="Train/build Markov artifacts from the canonical corpus.")
    build.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with extracted raw ship JSON files (legacy training path).",
    )
    build.add_argument(
        "--graph-input-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory with pre-generated ship graph JSON files "
            "(produced by scripts/generate_ship_graphs.py). "
            "Uses BFS-based touching-anchor ordering for better structural coherence. "
            "Mutually exclusive with --input-dir."
        ),
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--markov-order", type=int, default=2)
    build.add_argument("--min-parts-per-ship", type=int, default=2)
    build.add_argument("--max-parts-per-ship", type=int, default=5000)
    build.add_argument("--anchor-window", type=int, default=128)
    build.add_argument("--validation-output", type=Path, default=None)
    build.add_argument(
        "--allowlist",
        nargs="+",
        metavar="PART_ID",
        default=None,
        help=(
            "Restrict training to only these part IDs (space-separated). "
            "Parts not in this list are ignored during training."
        ),
    )
    build.add_argument(
        "--allowlist-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to a file containing part IDs to allow, one per line. "
            "Lines starting with # are ignored. "
            "Can be combined with --allowlist."
        ),
    )

    # ── generate ──
    generate = subparsers.add_parser("generate", help="Generate sample ships from a built artifact.")
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--max-parts", type=int, default=250)
    generate.add_argument("--max-attempts", type=int, default=3000)
    generate.add_argument("--max-resample-per-step", type=int, default=32)
    generate.add_argument("--bounds-min-x", type=int, default=-64)
    generate.add_argument("--bounds-max-x", type=int, default=64)
    generate.add_argument("--bounds-min-y", type=int, default=-64)
    generate.add_argument("--bounds-max-y", type=int, default=64)
    generate.add_argument("--seed", type=int, default=1337)
    generate.add_argument(
        "--export-png-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "If set, also export each generated ship as a .ship.png to this directory. "
            "Each sample-NNN.json produces a sample-NNN.ship.png."
        ),
    )
    generate.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip roundtrip validation when exporting PNGs (faster).",
    )
    generate.add_argument(
        "--allowlist",
        nargs="+",
        metavar="PART_ID",
        default=None,
        help=(
            "Restrict generation to only these part IDs (space-separated). "
            "Transitions sampling tokens not in this list are skipped."
        ),
    )
    generate.add_argument(
        "--allowlist-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a file of allowed part IDs, one per line. Can be combined with --allowlist.",
    )
    generate.add_argument(
        "--mirror-symmetry",
        action="store_true",
        default=False,
        help=(
            "Enforce left-right mirror symmetry across the ship centerline. "
            "The axis sits between grid columns -1 and 0 (x = -0.5). "
            "Primary parts are placed on the left half (x ≤ -1) and automatically "
            "mirrored to the right half (x ≥ 0). "
            "--max-parts counts the combined total (primary + mirrors)."
        ),
    )
    generate.add_argument(
        "--require",
        nargs=2,
        metavar=("PART_ID", "COUNT"),
        action="append",
        default=None,
        help=(
            "Require at least COUNT of PART_ID somewhere on the final ship "
            "(total, counting both primary and mirror halves). "
            "May be repeated: --require cosmoteer.control_room 1 --require cosmoteer.reactor_small 2"
        ),
    )
    generate.add_argument(
        "--requirements-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to a requirements file. Accepted formats: "
            'JSON object {"part_id": count, ...} or plain text with one "PART_ID COUNT" per line '
            "(# comments allowed). Combined with --require."
        ),
    )
    generate.add_argument(
        "--seed-json",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to an existing ship JSON to use as a seed layout. "
            "Accepts generated JSON (with 'parts' key) or Cosmoteer extracted JSON (with 'Parts' key). "
            "Non-vanilla and geometry-unknown parts are silently skipped."
        ),
    )
    generate.add_argument(
        "--seed-png",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Path to a .ship.png to use as a seed layout. "
            "The embedded ship payload is parsed and vanilla parts are extracted. "
            "Cannot be combined with --seed-json."
        ),
    )

    # ── export ──
    export = subparsers.add_parser(
        "export",
        help="Export already-generated sample JSON files to .ship.png files.",
    )
    export.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing sample-NNN.json files produced by 'generate'.",
    )
    export.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write .ship.png files into.",
    )
    export.add_argument(
        "--name-prefix",
        type=str,
        default="gen",
        help="Prefix for embedded ship names (default: 'gen').",
    )
    export.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip roundtrip validation (faster).",
    )
    export.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write a JSON export report to this file.",
    )

    # ── validate ──
    validate = subparsers.add_parser(
        "validate",
        help="Validate coordinate assumptions against the real canonical corpus.",
    )
    validate.add_argument("--input-dir", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--sample-limit", type=int, default=None)

    return parser


# ── command handlers ───────────────────────────────────────────────────────────


def cmd_build(args: argparse.Namespace) -> int:
    graph_input_dir = getattr(args, "graph_input_dir", None)
    input_dir = getattr(args, "input_dir", None)

    if graph_input_dir is not None and input_dir is not None:
        print("[build] ERROR: --graph-input-dir and --input-dir are mutually exclusive")
        return 1
    if graph_input_dir is None and input_dir is None:
        print("[build] ERROR: one of --input-dir or --graph-input-dir is required")
        return 1

    allowlist = _load_allowlist(args.allowlist, args.allowlist_file)
    if allowlist is not None:
        print(f"[build] allowlist active: {len(allowlist)} part IDs")

    training_config = TrainingConfig(
        markov_order=args.markov_order,
        min_parts_per_ship=args.min_parts_per_ship,
        max_parts_per_ship=args.max_parts_per_ship,
        anchor_window=args.anchor_window,
        part_allowlist=allowlist,
    )

    if graph_input_dir is not None:
        print(f"[build] training from graph corpus: {graph_input_dir}")
        model = build_model_from_graph_corpus(graph_input_dir, training_config)
    else:
        print(f"[build] training from raw ship corpus: {input_dir}")
        model = build_model_from_corpus(input_dir, training_config)

    model.save(args.output)
    stats = model.payload.get("stats", {})
    touching_frac = (
        stats.get("touching_transitions", 0) / stats.get("transition_tokens", 1)
        if stats.get("transition_tokens", 0) > 0 else 0.0
    )
    print(
        f"[build] model saved to {args.output}  "
        f"ships={stats.get('ships_used','?')}  "
        f"tokens={stats.get('transition_tokens','?')}  "
        f"touching={touching_frac:.1%}"
    )
    if args.validation_output is not None:
        if input_dir is not None:
            payload = validate_relative_placement_assumptions(input_dir)
            args.validation_output.parent.mkdir(parents=True, exist_ok=True)
            args.validation_output.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"[build] validation written to {args.validation_output}")
        else:
            print("[build] NOTE: --validation-output is only supported with --input-dir (skipped for graph corpus)")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    model = RelativeMarkovModel.load(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    allowlist = _load_allowlist(args.allowlist, args.allowlist_file)
    if allowlist is not None:
        print(f"[generate] allowlist active: {len(allowlist)} part IDs")
    if args.mirror_symmetry:
        print("[generate] mirror symmetry: ON  (axis at x = -0.5, primary half x ≤ -1)")

    # Load requirements
    requirements = _load_requirements(getattr(args, "require", None), getattr(args, "requirements_file", None))
    if requirements:
        print(f"[generate] requirements active: {len(requirements)} part(s)")
        for pid, cnt in sorted(requirements.items()):
            print(f"[generate]   {pid}: at least {cnt}")

    # Load seed parts (JSON takes priority over PNG; only one may be set)
    seed_parts = None
    seed_json = getattr(args, "seed_json", None)
    seed_png = getattr(args, "seed_png", None)
    if seed_json is not None and seed_png is not None:
        print("[generate] ERROR: --seed-json and --seed-png are mutually exclusive")
        return 1
    if seed_json is not None:
        seed_parts = _load_seed_parts_from_json(seed_json)
        print(f"[generate] seed: loaded {len(seed_parts)} parts from {seed_json.name}")
    elif seed_png is not None:
        seed_parts = _load_seed_parts_from_png(seed_png)
        print(f"[generate] seed: loaded {len(seed_parts)} vanilla parts from {seed_png.name}")

    export_dir = args.export_png_dir
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        from .export import export_ship_png as _export_ship_png

    for idx in range(args.count):
        config = GenerationConfig(
            max_parts=args.max_parts,
            max_attempts=args.max_attempts,
            max_resample_per_step=args.max_resample_per_step,
            bounds_min_x=args.bounds_min_x,
            bounds_max_x=args.bounds_max_x,
            bounds_min_y=args.bounds_min_y,
            bounds_max_y=args.bounds_max_y,
            rng_seed=args.seed + idx,
            part_allowlist=allowlist,
            mirror_symmetry=args.mirror_symmetry,
            part_requirements=requirements,
        )
        try:
            payload = model.generate(config, seed_parts=seed_parts)
        except RuntimeError as exc:
            print(f"[generate] sample-{idx:03d} FAILED: {exc}")
            continue

        out_path = args.output / f"sample-{idx:03d}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

        # Build status line
        stats = payload["stats"]
        mirror_info = ""
        if args.mirror_symmetry:
            ms = stats.get("mirror", {})
            mirror_info = f"  primary={ms.get('primary_parts','?')} mirror={ms.get('mirror_parts','?')}"
        req_info = ""
        if requirements:
            rs = stats.get("requirements", {})
            sat = rs.get("satisfied", "?")
            req_info = f"  reqs={'OK' if sat else 'UNMET'}"
        seed_info = ""
        if seed_parts is not None:
            ss = stats.get("seed", {})
            seed_info = f"  seed_placed={ss.get('seed_parts_placed','?')}"
        print(f"[generate] sample-{idx:03d}: {stats['parts_generated']} parts ({stats['stop_reason']}){mirror_info}{req_info}{seed_info}")

        # Print requirements progress if any unmet
        if requirements:
            rs = stats.get("requirements", {})
            if not rs.get("satisfied"):
                for pid, prog in rs.get("progress", {}).items():
                    if not prog.get("satisfied"):
                        print(f"[generate]   UNMET {pid}: need {prog['required']}, got {prog['actual']}")

        if export_dir is not None:
            png_path = export_dir / f"sample-{idx:03d}.ship.png"
            try:
                result = _export_ship_png(
                    payload,
                    png_path,
                    name=f"gen-{idx:03d}",
                    validate=not args.no_validate,
                )
                status = "OK" if result.get("valid") else ("?" if result.get("valid") is None else "WARN")
                rt = result.get("roundtrip", {})
                print(
                    f"[export]  sample-{idx:03d}.ship.png [{status}] "
                    f"parts_in={rt.get('parts_in','?')} parts_out={rt.get('parts_out','?')} "
                    f"size={result.get('parts_exported','?')}pts "
                    f"png={rt.get('png_bytes','?')}B"
                )
                if rt.get("warnings"):
                    for w in rt["warnings"]:
                        print(f"[export]    WARN: {w}")
            except Exception as exc:
                print(f"[export]  sample-{idx:03d}.ship.png FAILED: {exc}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_batch

    results = export_batch(
        args.input_dir,
        args.output_dir,
        validate=not args.no_validate,
        name_prefix=args.name_prefix,
    )

    ok_count = sum(1 for r in results if r.get("valid") is True)
    warn_count = sum(1 for r in results if r.get("valid") is False)
    err_count = sum(1 for r in results if "error" in r)
    print(f"[export] {len(results)} files processed: {ok_count} OK, {warn_count} validation issues, {err_count} errors")

    for r in results:
        src = Path(r.get("source", "?")).name
        if "error" in r:
            print(f"  ERROR {src}: {r['error']}")
        else:
            status = "OK" if r.get("valid") else ("?" if r.get("valid") is None else "WARN")
            rt = r.get("roundtrip", {})
            print(
                f"  [{status}] {src} → {Path(r.get('output_path','?')).name} "
                f"parts_in={rt.get('parts_in','?')} parts_out={rt.get('parts_out','?')}"
            )
            for w in (rt.get("warnings") or []):
                print(f"    WARN: {w}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[export] report written to {args.report}")

    return 0 if err_count == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    payload = validate_relative_placement_assumptions(args.input_dir, sample_limit=args.sample_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build":
        return cmd_build(args)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "export":
        return cmd_export(args)
    if args.command == "validate":
        return cmd_validate(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
