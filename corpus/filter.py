"""Corpus filter pipeline: load graph JSON files, apply rules, write output."""

from __future__ import annotations

import itertools
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import orjson

from corpus.context import CorpusContext
from corpus.rules.base import CorpusRule, RuleResult
from graph_expansion.context import EXPANSION_GRAPH_NAME

__all__ = [
    "FilterResult",
    "RejectionRecord",
    "run_filter",
    "validate_corpus_has_expansion",
]

log = logging.getLogger(__name__)

_SKIP_FILENAMES = frozenset({"manifest.json", "rejections.jsonl"})


def _is_graph_json(path: Path) -> bool:
    return path.suffix == ".json" and path.name not in _SKIP_FILENAMES and not path.name.startswith(".")


@dataclass(slots=True)
class RejectionRecord:
    """Record of a single rejected ship."""

    file: str
    ship_name: str
    author: str
    reasons: list[dict[str, str]]


@dataclass(slots=True)
class FilterResult:
    """Summary of a completed filter run."""

    ships_scanned: int = 0
    ships_kept: int = 0
    ships_rejected: int = 0
    rejections_by_rule: dict[str, int] = field(default_factory=dict)
    rejections: list[RejectionRecord] = field(default_factory=list)


def validate_corpus_has_expansion(
    input_dir: Path,
    *,
    sample_size: int = 5,
) -> None:
    """Raise RuntimeError if sampled files in *input_dir* lack expansion graphs.

    Called at startup when ``require_reachable_reactor`` is enabled so that the
    error surfaces before any work is done.
    """
    sample = list(
        itertools.islice(
            (p for p in input_dir.iterdir() if _is_graph_json(p)),
            sample_size,
        )
    )
    if not sample:
        raise RuntimeError(
            "require_reachable_reactor rule is enabled but the input directory "
            f"contains no graph JSON files: {input_dir}"
        )
    missing = []
    for path in sample:
        try:
            payload = orjson.loads(path.read_bytes())
        except Exception:
            continue
        if EXPANSION_GRAPH_NAME not in payload.get("graphs", {}):
            missing.append(path.name)
    if missing:
        raise RuntimeError(
            "require_reachable_reactor rule is enabled but the following sampled "
            f"files lack expansion graphs: {missing!r}. "
            "Run graph-expansion expand on the corpus first."
        )


def run_filter(
    input_dir: Path,
    output_dir: Path,
    rules: Sequence[CorpusRule],
    *,
    write_rejections_log: bool = True,
) -> FilterResult:
    """Apply *rules* to every graph JSON file in *input_dir*.

    Accepted files are copied verbatim to *output_dir*.  A ``manifest.json`` is
    always written.  ``rejections.jsonl`` is written when *write_rejections_log*
    is True and at least one ship was rejected.

    Returns a :class:`FilterResult` summarising the run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result = FilterResult()
    for rule in rules:
        result.rejections_by_rule[rule.name] = 0

    # Sort for deterministic ordering.
    json_files = sorted(p for p in input_dir.iterdir() if _is_graph_json(p))

    for path in json_files:
        result.ships_scanned += 1
        try:
            payload: dict[str, Any] = orjson.loads(path.read_bytes())
        except Exception as exc:
            log.warning("Skipping %s: failed to parse JSON: %s", path.name, exc)
            continue

        context = CorpusContext(path, payload)
        failures: list[dict[str, str]] = []
        for rule in rules:
            rule_result: RuleResult = rule.evaluate(context)
            if not rule_result.passed:
                failures.append(
                    {"rule": rule.name, "message": rule_result.message or ""}
                )
                result.rejections_by_rule[rule.name] += 1

        if failures:
            result.ships_rejected += 1
            result.rejections.append(
                RejectionRecord(
                    file=path.name,
                    ship_name=context.ship_name,
                    author=context.author,
                    reasons=failures,
                )
            )
            log.debug("Rejected %s: %s", path.name, failures)
        else:
            result.ships_kept += 1
            shutil.copy2(path, output_dir / path.name)

    # Write manifest
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "ships_scanned": result.ships_scanned,
        "ships_kept": result.ships_kept,
        "ships_rejected": result.ships_rejected,
        "active_rules": [{"name": r.name, "version": r.version} for r in rules],
        "rejections_by_rule": result.rejections_by_rule,
    }
    (output_dir / "manifest.json").write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
    )

    if write_rejections_log and result.rejections:
        lines = [
            orjson.dumps(
                {
                    "file": rec.file,
                    "ship_name": rec.ship_name,
                    "author": rec.author,
                    "reasons": rec.reasons,
                }
            )
            for rec in result.rejections
        ]
        (output_dir / "rejections.jsonl").write_bytes(b"\n".join(lines) + b"\n")

    return result
