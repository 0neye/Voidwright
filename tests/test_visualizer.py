"""Tests for generation visualization, icon discovery, and video output."""

from __future__ import annotations

import json
import shutil
from io import StringIO
from pathlib import Path

import pytest
from PIL import Image

import generator.cli
import main
from common.cosmoteer_install import (
    find_cosmoteer_install_root,
    iter_steam_library_paths,
    parse_steam_libraryfolders_vdf,
    resolve_terran_part_icons_root,
)
from markov.generation import WeightedSampler
from markov.model import END_TOKEN, GenerationConfig, RelativeMarkovModel, RelativePlacementToken
from markov.state import history_symbol
from visualizer import (
    VisualizationPart,
    VisualizationRecorder,
    load_part_icon_library,
    render_events_to_mp4,
)
from visualizer.renderer import render_visualization_frames


def _build_minimal_model_payload() -> dict:
    """Build a tiny deterministic model payload for visualization tests."""

    root_token = RelativePlacementToken(
        part_id="cosmoteer.armor_wedge",
        rotation=0,
        anchor_part_id="__ROOT__",
        anchor_rotation=0,
        dx=0,
        dy=0,
    ).as_key()
    invalid_air_contact = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_wedge",
        anchor_rotation=0,
        dx=0,
        dy=-1,
    ).as_key()
    valid_structural_contact = RelativePlacementToken(
        part_id="cosmoteer.armor",
        rotation=0,
        anchor_part_id="cosmoteer.armor_wedge",
        anchor_rotation=0,
        dx=1,
        dy=0,
    ).as_key()

    root_state = history_symbol(root_token)
    second_state = history_symbol(valid_structural_contact)

    return {
        "schema_version": 2,
        "config": {"markov_order": 1},
        "start_counts": {root_token: 1},
        "transition_counts": {
            root_state: {
                invalid_air_contact: 1,
                valid_structural_contact: 1,
            },
            second_state: {END_TOKEN: 1},
        },
        "part_frequency": {
            "cosmoteer.armor_wedge": 1,
            "cosmoteer.armor": 2,
        },
    }


def _write_test_icons(root: Path) -> Path:
    """Create a minimal Terran-style icon directory for visualization tests."""

    for part_folder, color in {
        "armor": (200, 120, 110, 255),
        "armor_wedge": (110, 180, 220, 255),
    }.items():
        icon_dir = root / part_folder
        icon_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGBA", (128, 128), color)
        image.save(icon_dir / "icon.png")
    return root


def _write_asymmetric_icon(root: Path, part_folder: str) -> Path:
    """Create a clearly asymmetric icon for transform-order assertions."""

    icon_dir = root / part_folder
    icon_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (20, 40), (0, 0, 0, 0))
    for x in range(0, 10):
        for y in range(0, 40):
            image.putpixel((x, y), (255, 0, 0, 255))
    for x in range(10, 20):
        for y in range(0, 40):
            image.putpixel((x, y), (0, 0, 255, 255))
    image.save(icon_dir / "icon.png")
    return root


def test_parse_steam_libraryfolders_vdf_reads_all_paths() -> None:
    """Steam library parsing should decode escaped library paths."""

    raw_text = """
"libraryfolders"
{
    "0"
    {
        "path"      "C:\\\\Program Files (x86)\\\\Steam"
    }
    "1"
    {
        "path"      "F:\\\\SteamLibrary"
    }
}
"""

    parsed_paths = parse_steam_libraryfolders_vdf(raw_text)

    assert parsed_paths == (
        Path(r"C:\Program Files (x86)\Steam"),
        Path(r"F:\SteamLibrary"),
    )


def test_iter_steam_library_paths_reads_libraryfolders_file(tmp_path: Path) -> None:
    """Library iteration should include the install path and additional Steam libraries."""

    steam_root = tmp_path / "Steam"
    steamapps_root = steam_root / "steamapps"
    steamapps_root.mkdir(parents=True)
    (steamapps_root / "libraryfolders.vdf").write_text(
        """
"libraryfolders"
{
    "0" { "path" "C:\\\\Steam" }
    "1" { "path" "D:\\\\SteamLibrary" }
}
""".strip(),
        encoding="utf-8",
    )

    libraries = iter_steam_library_paths([steam_root])

    assert steam_root in libraries
    assert Path(r"C:\Steam") in libraries
    assert Path(r"D:\SteamLibrary") in libraries


def test_find_cosmoteer_install_root_uses_secondary_steam_library_drive(tmp_path: Path) -> None:
    """Install discovery should find Cosmoteer in a non-default Steam library."""

    steam_root = tmp_path / "C-drive-Steam"
    steamapps_root = steam_root / "steamapps"
    steamapps_root.mkdir(parents=True)
    remote_library = tmp_path / "F-drive-SteamLibrary"
    cosmoteer_root = remote_library / "steamapps" / "common" / "Cosmoteer"
    (cosmoteer_root / "Data" / "ships" / "terran").mkdir(parents=True)
    (steamapps_root / "libraryfolders.vdf").write_text(
        f"""
"libraryfolders"
{{
    "0" {{ "path" "{str(steam_root).replace('\\', '\\\\')}" }}
    "1" {{ "path" "{str(remote_library).replace('\\', '\\\\')}" }}
}}
""".strip(),
        encoding="utf-8",
    )

    discovered_root = find_cosmoteer_install_root([steam_root])

    assert discovered_root == cosmoteer_root


def test_resolve_terran_part_icons_root_uses_fallback_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback icon cache should be used when auto-discovery finds nothing."""

    fallback_root = _write_test_icons(tmp_path / "assets" / "local" / "cosmoteer-icons" / "terran")
    monkeypatch.setattr("common.cosmoteer_install.find_cosmoteer_install_root", lambda: None)

    resolved_root = resolve_terran_part_icons_root(fallback_root=fallback_root)

    assert resolved_root == fallback_root


def test_root_help_exposes_visualization_flags() -> None:
    """Root delegated help should show shared visualization flags on the Markov backend."""

    output_stream = StringIO()
    command_registry = main.build_domain_registry()
    root_parser = main.build_root_parser(command_registry)
    root_parser.prog = "main.py"

    exit_code = main.run_help_command(
        ["generator", "generate", "markov"],
        command_registry,
        root_parser,
        output_stream,
    )

    rendered_output = output_stream.getvalue()

    assert exit_code == 0
    assert "--visualize" in rendered_output
    assert "--icons-root" in rendered_output
    assert "--game-root" in rendered_output


def test_markov_generation_recorder_emits_rejected_attempt_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visualization recorder should receive accepted and rejected Markov events."""

    model = RelativeMarkovModel(_build_minimal_model_payload())
    recorder = VisualizationRecorder(sample_index=0)
    config = GenerationConfig(
        max_parts=10,
        max_attempts=10,
        max_resample_per_step=4,
        rng_seed=123,
    )

    root_token = next(iter(model.start_counts))
    invalid_air_contact = next(
        token_key
        for token_key in model.transition_counts[history_symbol(root_token)]
        if token_key != END_TOKEN and RelativePlacementToken.from_key(token_key).dx == 0
    )
    valid_structural_contact = next(
        token_key
        for token_key in model.transition_counts[history_symbol(root_token)]
        if token_key != END_TOKEN and RelativePlacementToken.from_key(token_key).dx == 1
    )
    sampled_tokens = iter([root_token, invalid_air_contact, valid_structural_contact, END_TOKEN])

    def _sample_in_order(counter: dict, _rng) -> str:
        sampled_token = next(sampled_tokens)
        assert sampled_token in counter
        return sampled_token

    monkeypatch.setattr(WeightedSampler, "sample", staticmethod(_sample_in_order))
    payload = model.generate(config, event_sink=recorder)

    assert payload["stats"]["rejections"]["structural"] == 1
    assert [event.kind for event in recorder.events] == [
        "sample_started",
        "part_placed",
        "attempt_rejected",
        "part_placed",
        "sample_finished",
    ]
    assert recorder.events[2].metadata["reason"] == "structural"


def test_renderer_builds_frames_from_recorded_events(tmp_path: Path) -> None:
    """Renderer should build a non-empty frame sequence from recorded events."""

    icons_root = _write_test_icons(tmp_path / "icons")
    icon_library = load_part_icon_library(icons_root=icons_root)
    recorder = VisualizationRecorder(sample_index=0)
    recorder.sample_started(config={"max_parts": 10}, seeded=False)
    recorder.part_placed(
        part=VisualizationPart(part_id="cosmoteer.armor_wedge", rotation=0, x=0, y=0),
        message="Accepted root placement",
        metadata={"placed_index": 0},
    )
    recorder.part_placed(
        part=VisualizationPart(part_id="cosmoteer.armor", rotation=0, x=1, y=0),
        message="Accepted placement",
        metadata={"placed_index": 1, "anchor_index": 0},
    )
    recorder.attempt_rejected(
        reason="bounds",
        part=VisualizationPart(part_id="cosmoteer.armor", rotation=0, x=4, y=4),
        message="Candidate rejected: left configured bounds",
        metadata={},
    )
    recorder.sample_finished(
        stats={"parts_generated": 2, "attempts": 3},
        stop_reason="end_token",
        message="Generation finished",
    )

    frames = render_visualization_frames(recorder.events, icon_library=icon_library)

    assert len(frames) == len(recorder.events) + 2
    assert frames[0].size[0] > 0
    assert frames[0].size[1] > 0


def test_renderer_handles_unknown_part_on_rejected_seed_event(tmp_path: Path) -> None:
    """Renderer should not crash when rejected seed geometry is unknown."""

    icons_root = _write_test_icons(tmp_path / "icons")
    icon_library = load_part_icon_library(icons_root=icons_root)
    recorder = VisualizationRecorder(sample_index=0)
    recorder.sample_started(config={"max_parts": 10}, seeded=True)
    recorder.attempt_rejected(
        reason="seed_geometry",
        part=VisualizationPart(part_id="modded.unknown_part", rotation=0, x=4, y=6),
        message="Seed part skipped: missing vanilla geometry",
        metadata={"is_seed": True},
    )
    recorder.sample_finished(
        stats={"parts_generated": 0, "attempts": 0},
        stop_reason="seed_rejected",
        message="Generation finished",
    )

    frames = render_visualization_frames(recorder.events, icon_library=icon_library)

    assert len(frames) == len(recorder.events) + 2
    assert frames[0].size[0] > 0
    assert frames[0].size[1] > 0


def test_icon_flip_x_is_applied_in_local_space_before_rotation(tmp_path: Path) -> None:
    """FlipX should act in part-local space instead of the final screen-space axis."""

    icons_root = _write_asymmetric_icon(tmp_path / "icons", "armor_2x1")
    icon_library = load_part_icon_library(icons_root=icons_root, cell_size=10)

    transformed_icon = icon_library.get_icon(
        "cosmoteer.armor_2x1",
        rotation=1,
        flip_x=True,
    )

    # Local-space horizontal flip followed by CW rotation places the original
    # red half on the bottom, not on the top.
    assert transformed_icon.getpixel((5, 5))[:3] == (0, 0, 255)
    assert transformed_icon.getpixel((5, transformed_icon.height - 5))[:3] == (255, 0, 0)


def test_half_cell_triangles_flip_after_rotation_for_visual_parity(tmp_path: Path) -> None:
    """Triangle icons should mirror after rotation to match the in-game sprite orientation."""

    icons_root = _write_asymmetric_icon(tmp_path / "icons", "armor_tri")
    icon_library = load_part_icon_library(icons_root=icons_root, cell_size=10)
    base_icon = Image.open(icons_root / "armor_tri" / "icon.png").convert("RGBA")

    transformed_icon = icon_library.get_icon(
        "cosmoteer.armor_tri",
        rotation=3,
        flip_x=True,
    )
    expected_icon = base_icon.rotate(-270, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    expected_icon = expected_icon.resize(transformed_icon.size, Image.Resampling.LANCZOS)

    assert list(transformed_icon.getdata()) == list(expected_icon.getdata())


def test_half_cell_triangles_keep_flip_y_in_local_space(tmp_path: Path) -> None:
    """Triangle FlipY should stay local-space even when FlipX parity is post-rotation."""

    icons_root = _write_asymmetric_icon(tmp_path / "icons", "armor_tri")
    icon_library = load_part_icon_library(icons_root=icons_root, cell_size=10)
    base_icon = Image.open(icons_root / "armor_tri" / "icon.png").convert("RGBA")

    transformed_icon = icon_library.get_icon(
        "cosmoteer.armor_tri",
        rotation=1,
        flip_y=True,
    )
    expected_icon = base_icon.transpose(Image.Transpose.FLIP_TOP_BOTTOM).rotate(-90, expand=True)
    expected_icon = expected_icon.resize(transformed_icon.size, Image.Resampling.LANCZOS)

    assert list(transformed_icon.getdata()) == list(expected_icon.getdata())


def test_half_cell_wedges_undo_saved_rotation_remap_before_flip(tmp_path: Path) -> None:
    """Flipped 1x1 wedge icons should render from the pre-remap rotation."""

    icons_root = _write_asymmetric_icon(tmp_path / "icons", "armor_wedge")
    icon_library = load_part_icon_library(icons_root=icons_root, cell_size=10)
    base_icon = Image.open(icons_root / "armor_wedge" / "icon.png").convert("RGBA")

    transformed_icon = icon_library.get_icon(
        "cosmoteer.armor_wedge",
        rotation=1,
        flip_x=True,
    )
    expected_icon = base_icon.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    expected_icon = expected_icon.resize(transformed_icon.size, Image.Resampling.LANCZOS)

    assert list(transformed_icon.getdata()) == list(expected_icon.getdata())


def test_half_cell_wedges_preserve_distinct_flipped_rotations(tmp_path: Path) -> None:
    """Different flipped wedge rotations should not collapse to one sprite."""

    icons_root = _write_asymmetric_icon(tmp_path / "icons", "armor_wedge")
    icon_library = load_part_icon_library(icons_root=icons_root, cell_size=10)
    base_icon = Image.open(icons_root / "armor_wedge" / "icon.png").convert("RGBA")

    transformed_icon = icon_library.get_icon(
        "cosmoteer.armor_wedge",
        rotation=3,
        flip_x=True,
    )
    expected_icon = base_icon.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(-180, expand=True)
    expected_icon = expected_icon.resize(transformed_icon.size, Image.Resampling.LANCZOS)

    assert list(transformed_icon.getdata()) == list(expected_icon.getdata())


def test_render_events_to_mp4_writes_video(tmp_path: Path) -> None:
    """Video writer should encode rendered frames into the sample MP4 path."""

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")

    icons_root = _write_test_icons(tmp_path / "icons")
    icon_library = load_part_icon_library(icons_root=icons_root)
    recorder = VisualizationRecorder(sample_index=0)
    recorder.sample_started(config={"max_parts": 10}, seeded=False)
    recorder.part_placed(
        part=VisualizationPart(part_id="cosmoteer.armor_wedge", rotation=0, x=0, y=0),
        message="Accepted root placement",
        metadata={"placed_index": 0},
    )
    recorder.sample_finished(
        stats={"parts_generated": 1, "attempts": 1},
        stop_reason="end_token",
        message="Generation finished",
    )

    output_path = tmp_path / "out" / "visualizations" / "sample-000.mp4"
    render_events_to_mp4(recorder.events, output_path, icon_library=icon_library)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generator_cli_visualize_writes_ship_png_and_mp4(tmp_path: Path) -> None:
    """The Markov CLI should write both the ship PNG and visualization MP4."""

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")

    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(_build_minimal_model_payload()) + "\n", encoding="utf-8")
    icons_root = _write_test_icons(tmp_path / "icons")
    output_dir = tmp_path / "generated"

    exit_code = generator.cli.main(
        [
            "generate",
            "markov",
            "--model",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
            "--seed",
            "123",
            "--no-validate",
            "--visualize",
            "--icons-root",
            str(icons_root),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "sample-000.ship.png").exists()
    assert (output_dir / "visualizations" / "sample-000.mp4").exists()
