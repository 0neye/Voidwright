"""Corpus filter pipeline: load graph JSON files, apply rules, write output."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import orjson

from corpus.context import CorpusContext
from corpus.rules.base import CorpusRule, RuleResult

__all__ = [
    "FilterResult",
    "RejectionRecord",
    "run_filter",
    "validate_corpus_has_expansion",
]

log = logging.getLogger(__name__)

_SKIP_FILENAMES = frozenset({"manifest.json", "rejections.jsonl"})


def _is_graph_json(path: Path) -> bool:
    return path.suffix == ".json" and path.name not in _SKIP_FILENAMES


class RejectionRecord:
    """Record of a single rejected ship."""

    __slots__ = ("file", "ship_name", "author", "reasons")

    def __init__(
        self,
        file: str,
        ship_name: str,
        author: str,
        reasons: list[dict[str, str]],
    ) -> None:
        self.file = file
        self.ship_name = ship_name
        self.author = author
        self.reasons = reasons


class FilterResult:
    """Summary of a completed filter run."""

    __slots__ = (
        "ships_scanned",
        "ships_kept",
        "ships_rejected",
        "rejections_by_rule",
        "rejections",
    )

    def __init__(self) -> None:
        self.ships_scanned: int = 0
        self.ships_kept: int = 0
        self.ships_rejected: int = 0
        self.rejections_by_rule: dict[str, int] = {}
        self.rejections: list[RejectionRecord] = []


def validate_corpus_has_expansion(
    input_dir: Path,
    *,
    sample_size: int = 5,
) -> None:
    """Raise RuntimeError if sampled files in *input_dir* lack expansion graphs.

    Called at startup when ``require_reachable_reactor`` is enabled so that the
    error surfaces before any work is done.
    """
    json_files = sorted(p for p in input_dir.iterdir() if _is_graph_json(p))
    sample = json_files[:sample_size]
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
        if "X_expansion_structural" not in payload.get("graphs", {}):
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
    always written.  An optional ``rejections.jsonl`` is written when
    *write_rejections_log* is True.

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
                result.rejections_by_rule[rule.name] = (
                    result.rejections_by_rule.get(rule.name, 0) + 1
                )

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
        "schema_version": 1,
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
        lines = []
        for rec in result.rejections:
            lines.append(
                orjson.dumps(
                    {
                        "file": rec.file,
                        "ship_name": rec.ship_name,
                        "author": rec.author,
                        "reasons": rec.reasons,
                    }
                )
            )
        (output_dir / "rejections.jsonl").write_bytes(b"\n".join(lines) + b"\n")

    return result
