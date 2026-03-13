"""Regression tests for Markov training payload builders."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from markov.training import build_payload_from_graph_corpus
from markov.types import TrainingConfig


def test_build_payload_from_graph_corpus_skips_hidden_sentinel_files() -> None:
    """build_payload_from_graph_corpus should skip .pipeline-version.json sentinel files.

    Regression test for: https://github.com/cosmoteer/voidwright/issues/...
    Python 3.10+ glob("*.json") matches hidden files, but only manifest.json
    was excluded. The .pipeline-version.json sentinel was incorrectly picked up
    as a graph corpus file, creating an empty ship entry.
    """

    with TemporaryDirectory() as tmpdir:
        graph_dir = Path(tmpdir)

        # Create a valid graph corpus file (with 2+ parts to meet default min_parts_per_ship)
        valid_graph = {
            "graphs": {
                "A_structural_part_graph": {
                    "nodes": [
                        {
                            "id": "0",
                            "part_id": "cosmoteer.corridor",
                            "rotation": 0,
                            "x": 0,
                            "y": 0,
                        },
                        {
                            "id": "1",
                            "part_id": "cosmoteer.reactor",
                            "rotation": 0,
                            "x": 1,
                            "y": 0,
                        },
                    ],
                    "edges": [{"from": "0", "to": "1"}],
                }
            }
        }
        (graph_dir / "ship1.json").write_text(json.dumps(valid_graph))

        # Create a .pipeline-version.json sentinel (should be ignored)
        sentinel = {"graph_schema_version": 5}
        (graph_dir / ".pipeline-version.json").write_text(json.dumps(sentinel))

        # Create manifest.json (should be ignored)
        manifest = {"files": ["ship1.json"]}
        (graph_dir / "manifest.json").write_text(json.dumps(manifest))

        # Build the model
        config = TrainingConfig()
        payload = build_payload_from_graph_corpus(graph_dir, config)

        # Should see exactly 1 ship (the valid graph), not 2
        # If .pipeline-version.json was picked up, it would create an empty ship
        # (since it has no "graphs" key)
        assert payload["stats"]["ships_seen"] == 1, (
            f"Expected to see 1 ship, but saw {payload['stats']['ships_seen']}. "
            "The .pipeline-version.json sentinel was likely picked up."
        )


def test_build_payload_from_graph_corpus_skips_manifest_json() -> None:
    """build_payload_from_graph_corpus should skip manifest.json files."""

    with TemporaryDirectory() as tmpdir:
        graph_dir = Path(tmpdir)

        # Create a valid graph corpus file (with 2+ parts to meet default min_parts_per_ship)
        valid_graph = {
            "graphs": {
                "A_structural_part_graph": {
                    "nodes": [
                        {
                            "id": "0",
                            "part_id": "cosmoteer.corridor",
                            "rotation": 0,
                            "x": 0,
                            "y": 0,
                        },
                        {
                            "id": "1",
                            "part_id": "cosmoteer.reactor",
                            "rotation": 0,
                            "x": 1,
                            "y": 0,
                        },
                    ],
                    "edges": [{"from": "0", "to": "1"}],
                }
            }
        }
        (graph_dir / "ship1.json").write_text(json.dumps(valid_graph))

        # Create manifest.json (should be ignored)
        manifest = {"files": ["ship1.json"]}
        (graph_dir / "manifest.json").write_text(json.dumps(manifest))

        # Build the model
        config = TrainingConfig()
        payload = build_payload_from_graph_corpus(graph_dir, config)

        # Should see exactly 1 ship (the valid graph), not 2
        assert payload["stats"]["ships_seen"] == 1
