"""MP4 encoding helpers for generation visualization videos."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .events import VisualizationEvent
from .icons import PartIconLibrary
from .renderer import render_visualization_frames

__all__ = ["ensure_ffmpeg_available", "render_events_to_mp4"]


def ensure_ffmpeg_available() -> str:
    """Return the ffmpeg executable path or raise a clear runtime error."""

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "Visualization requires ffmpeg on PATH. Install ffmpeg to use --visualize."
        )
    return ffmpeg_path


def render_events_to_mp4(
    events: list[VisualizationEvent],
    output_path: str | Path,
    *,
    icon_library: PartIconLibrary,
    fps: int = 6,
) -> Path:
    """Render recorded events into an MP4 video."""

    ffmpeg_path = ensure_ffmpeg_available()
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_frames = render_visualization_frames(events, icon_library=icon_library)
    if not rendered_frames:
        raise RuntimeError("No visualization frames were produced for the generated sample.")

    with tempfile.TemporaryDirectory(prefix="voidwright-visualizer-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for frame_index, frame in enumerate(rendered_frames):
            frame.save(temp_dir_path / f"frame-{frame_index:05d}.png", format="PNG")

        ffmpeg_command = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir_path / "frame-%05d.png"),
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            str(resolved_output_path),
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed while encoding the visualization video: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    return resolved_output_path
