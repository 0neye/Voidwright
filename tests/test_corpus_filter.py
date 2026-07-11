"""Tests for the corpus filtering pipeline and rules."""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import pytest

from corpus.context import CorpusContext
from corpus.filter import FilterResult, run_filter, validate_corpus_has_expansion
from corpus.rules.max_size import MaxSizeRule
from corpus.rules.require_crew_rooms import RequireCrewRoomsRule
from corpus.rules.require_reachable_reactor import RequireReachableReactorRule

# ---------------------------------------------------------------------------
# Synthetic graph payload helpers
# ---------------------------------------------------------------------------

_STRUCTURAL_GRAPH = "A_structural_part_graph"
_EXPANSION_GRAPH = "X_expansion_structural"


def _make_payload(
    *,
    ship_name: str = "TestShip",
    author: str = "TestAuthor",
    parts: int = 10,
    occupied_cells: int = 40,
    traversable_cells: int = 20,
    part_nodes: list[dict] | None = None,
    crew_access_reactor_edges: int | None = None,
) -> dict:
    """Build a minimal synthetic graph payload for tests."""
    if part_nodes is None:
        part_nodes = [
            {"kind": "part", "part_id": f"cosmoteer.corridor_{i}", "id": i}
            for i in range(parts)
        ]

    payload: dict = {
        "ship_name": ship_name,
        "author": author,
        "graphs": {
            _STRUCTURAL_GRAPH: {
                "summary": {
                    "parts": parts,
                    "occupied_cells": occupied_cells,
                    "traversable_cells": traversable_cells,
                },
                "nodes": part_nodes,
                "edges": [],
            }
        },
    }
    if crew_access_reactor_edges is not None:
        payload["graphs"][_EXPANSION_GRAPH] = {
            "summary": {
                "crew_access_reactor_edges": crew_access_reactor_edges,
            },
            "nodes": [],
            "edges": [],
        }
    return payload


def _make_crew_room_nodes(count: int) -> list[dict]:
    return [
        {"kind": "part", "part_id": "cosmoteer.crew_quarters", "id": i}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# CorpusContext tests
# ---------------------------------------------------------------------------


class TestCorpusContext:
    def _ctx(self, payload: dict, path: Path | None = None) -> CorpusContext:
        return CorpusContext(path or Path("test.json"), payload)

    def test_ship_name(self) -> None:
        ctx = self._ctx(_make_payload(ship_name="MyShip"))
        assert ctx.ship_name == "MyShip"

    def test_author(self) -> None:
        ctx = self._ctx(_make_payload(author="Alice"))
        assert ctx.author == "Alice"

    def test_part_count(self) -> None:
        ctx = self._ctx(_make_payload(parts=42))
        assert ctx.part_count == 42

    def test_occupied_cells(self) -> None:
        ctx = self._ctx(_make_payload(occupied_cells=99))
        assert ctx.occupied_cells == 99

    def test_traversable_cells(self) -> None:
        ctx = self._ctx(_make_payload(traversable_cells=15))
        assert ctx.traversable_cells == 15

    def test_crew_room_count_none(self) -> None:
        ctx = self._ctx(_make_payload(part_nodes=[
            {"kind": "part", "part_id": "cosmoteer.reactor", "id": 0},
        ]))
        assert ctx.crew_room_count == 0

    def test_crew_room_count_some(self) -> None:
        ctx = self._ctx(_make_payload(part_nodes=_make_crew_room_nodes(3)))
        assert ctx.crew_room_count == 3

    def test_has_expansion_graph_false(self) -> None:
        ctx = self._ctx(_make_payload())
        assert not ctx.has_expansion_graph

    def test_has_expansion_graph_true(self) -> None:
        ctx = self._ctx(_make_payload(crew_access_reactor_edges=2))
        assert ctx.has_expansion_graph

    def test_crew_access_reactor_edges(self) -> None:
        ctx = self._ctx(_make_payload(crew_access_reactor_edges=5))
        assert ctx.crew_access_reactor_edges == 5

    def test_crew_access_reactor_edges_missing_expansion(self) -> None:
        ctx = self._ctx(_make_payload())
        assert ctx.crew_access_reactor_edges == 0


# ---------------------------------------------------------------------------
# MaxSizeRule tests
# ---------------------------------------------------------------------------


class TestMaxSizeRule:
    def test_pass_no_thresholds(self) -> None:
        rule = MaxSizeRule()
        ctx = CorpusContext(Path("x.json"), _make_payload(parts=9999, occupied_cells=99999))
        result = rule.evaluate(ctx)
        assert result.passed

    def test_pass_under_max_parts(self) -> None:
        rule = MaxSizeRule(max_parts=100)
        ctx = CorpusContext(Path("x.json"), _make_payload(parts=50))
        assert rule.evaluate(ctx).passed

    def test_fail_over_max_parts(self) -> None:
        rule = MaxSizeRule(max_parts=100)
        ctx = CorpusContext(Path("x.json"), _make_payload(parts=150))
        result = rule.evaluate(ctx)
        assert not result.passed
        assert "150" in result.message
        assert "100" in result.message

    def test_pass_under_max_occupied_cells(self) -> None:
        rule = MaxSizeRule(max_occupied_cells=500)
        ctx = CorpusContext(Path("x.json"), _make_payload(occupied_cells=300))
        assert rule.evaluate(ctx).passed

    def test_fail_over_max_occupied_cells(self) -> None:
        rule = MaxSizeRule(max_occupied_cells=500)
        ctx = CorpusContext(Path("x.json"), _make_payload(occupied_cells=600))
        result = rule.evaluate(ctx)
        assert not result.passed
        assert "600" in result.message

    def test_parts_checked_before_cells(self) -> None:
        # Both thresholds exceeded; parts message expected first.
        rule = MaxSizeRule(max_parts=10, max_occupied_cells=50)
        ctx = CorpusContext(
            Path("x.json"), _make_payload(parts=20, occupied_cells=100)
        )
        result = rule.evaluate(ctx)
        assert not result.passed
        assert "max_parts" in result.message


# ---------------------------------------------------------------------------
# RequireCrewRoomsRule tests
# ---------------------------------------------------------------------------


class TestRequireCrewRoomsRule:
    def test_pass_with_crew_rooms(self) -> None:
        rule = RequireCrewRoomsRule()
        ctx = CorpusContext(
            Path("x.json"),
            _make_payload(part_nodes=_make_crew_room_nodes(1)),
        )
        assert rule.evaluate(ctx).passed

    def test_fail_without_crew_rooms(self) -> None:
        rule = RequireCrewRoomsRule()
        ctx = CorpusContext(
            Path("x.json"),
            _make_payload(part_nodes=[
                {"kind": "part", "part_id": "cosmoteer.reactor", "id": 0}
            ]),
        )
        result = rule.evaluate(ctx)
        assert not result.passed
        assert "crew rooms" in result.message


# ---------------------------------------------------------------------------
# RequireReachableReactorRule tests
# ---------------------------------------------------------------------------


class TestRequireReachableReactorRule:
    def test_pass_crew_rooms_and_reactor(self) -> None:
        rule = RequireReachableReactorRule()
        ctx = CorpusContext(
            Path("x.json"),
            _make_payload(
                part_nodes=_make_crew_room_nodes(1),
                crew_access_reactor_edges=2,
            ),
        )
        assert rule.evaluate(ctx).passed

    def test_fail_crew_rooms_no_reactor(self) -> None:
        rule = RequireReachableReactorRule()
        ctx = CorpusContext(
            Path("x.json"),
            _make_payload(
                part_nodes=_make_crew_room_nodes(1),
                crew_access_reactor_edges=0,
            ),
        )
        result = rule.evaluate(ctx)
        assert not result.passed
        assert "reactor" in result.message

    def test_pass_no_crew_rooms_no_reactor(self) -> None:
        # No crew rooms -> rule does not apply.
        rule = RequireReachableReactorRule()
        ctx = CorpusContext(
            Path("x.json"),
            _make_payload(
                part_nodes=[{"kind": "part", "part_id": "cosmoteer.reactor", "id": 0}],
                crew_access_reactor_edges=0,
            ),
        )
        assert rule.evaluate(ctx).passed

    def test_raises_if_no_expansion_graph(self) -> None:
        rule = RequireReachableReactorRule()
        ctx = CorpusContext(
            Path("x.json"),
            _make_payload(part_nodes=_make_crew_room_nodes(1)),
        )
        with pytest.raises(RuntimeError, match="expansion graph missing"):
            rule.evaluate(ctx)


# ---------------------------------------------------------------------------
# Pipeline (run_filter) tests
# ---------------------------------------------------------------------------


class TestRunFilter:
    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_bytes(orjson.dumps(payload))

    def test_accepts_all_when_no_rules(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        self._write_json(input_dir / "ship1.json", _make_payload(parts=5))
        self._write_json(input_dir / "ship2.json", _make_payload(parts=10))

        result = run_filter(input_dir, output_dir, rules=[])

        assert result.ships_scanned == 2
        assert result.ships_kept == 2
        assert result.ships_rejected == 0
        assert (output_dir / "ship1.json").exists()
        assert (output_dir / "ship2.json").exists()

    def test_rejects_over_size(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        self._write_json(input_dir / "small.json", _make_payload(parts=5))
        self._write_json(input_dir / "large.json", _make_payload(parts=50))

        result = run_filter(input_dir, output_dir, [MaxSizeRule(max_parts=10)])

        assert result.ships_kept == 1
        assert result.ships_rejected == 1
        assert (output_dir / "small.json").exists()
        assert not (output_dir / "large.json").exists()

    def test_manifest_written(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        self._write_json(input_dir / "ship.json", _make_payload(parts=5))

        run_filter(input_dir, output_dir, [MaxSizeRule(max_parts=10)])

        manifest = orjson.loads((output_dir / "manifest.json").read_bytes())
        assert manifest["schema_version"] == 2
        assert manifest["ships_scanned"] == 1
        assert manifest["ships_kept"] == 1
        assert manifest["ships_rejected"] == 0
        assert manifest["active_rules"] == [{"name": "max_size", "version": 1}]

    def test_rejections_jsonl_written(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        self._write_json(input_dir / "big.json", _make_payload(parts=999))

        run_filter(input_dir, output_dir, [MaxSizeRule(max_parts=10)])

        log_path = output_dir / "rejections.jsonl"
        assert log_path.exists()
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        rec = orjson.loads(lines[0])
        assert rec["file"] == "big.json"
        assert len(rec["reasons"]) == 1

    def test_no_rejections_log_flag(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        self._write_json(input_dir / "big.json", _make_payload(parts=999))

        run_filter(
            input_dir, output_dir, [MaxSizeRule(max_parts=10)],
            write_rejections_log=False,
        )

        assert not (output_dir / "rejections.jsonl").exists()

    def test_skips_manifest_json_in_input(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        # manifest.json in input should be skipped, not treated as a ship
        (input_dir / "manifest.json").write_bytes(orjson.dumps({"schema_version": 1}))
        self._write_json(input_dir / "ship.json", _make_payload(parts=5))

        result = run_filter(input_dir, output_dir, [])
        assert result.ships_scanned == 1

    def test_rejections_by_rule_counts(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        self._write_json(
            input_dir / "no_crew.json",
            _make_payload(
                part_nodes=[{"kind": "part", "part_id": "cosmoteer.reactor", "id": 0}]
            ),
        )
        self._write_json(
            input_dir / "big_with_crew.json",
            _make_payload(parts=999, part_nodes=_make_crew_room_nodes(1)),
        )

        result = run_filter(
            input_dir,
            output_dir,
            [MaxSizeRule(max_parts=10), RequireCrewRoomsRule()],
        )

        assert result.rejections_by_rule["max_size"] == 1
        assert result.rejections_by_rule["require_crew_rooms"] == 1


# ---------------------------------------------------------------------------
# validate_corpus_has_expansion tests
# ---------------------------------------------------------------------------


class TestValidateCorpusHasExpansion:
    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_bytes(orjson.dumps(payload))

    def test_passes_when_expansion_present(self, tmp_path: Path) -> None:
        self._write_json(
            tmp_path / "ship.json",
            _make_payload(crew_access_reactor_edges=1),
        )
        # Should not raise.
        validate_corpus_has_expansion(tmp_path)

    def test_fails_when_expansion_absent(self, tmp_path: Path) -> None:
        self._write_json(tmp_path / "ship.json", _make_payload())
        with pytest.raises(RuntimeError, match="expansion graph"):
            validate_corpus_has_expansion(tmp_path)

    def test_fails_on_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="no graph JSON files"):
            validate_corpus_has_expansion(tmp_path)
