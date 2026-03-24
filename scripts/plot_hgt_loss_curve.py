#!/usr/bin/env python3
"""Render a loss curve PNG from an HGT training log.

The script parses lines like:

    epoch 001/100  train_loss=2.4408  train_acc=0.5617  val_loss=0.8700  ...

The input log is provided explicitly. The default output path is derived from
the log's parent directory name, so any run folder can be plotted without code
changes.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["LossPoint", "parse_loss_log", "render_loss_curve", "main"]

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised only in a misconfigured env
    raise RuntimeError("Pillow is required: pip install Pillow") from exc

_EPOCH_RE = re.compile(r"^epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+(?P<body>.*)$")
_FIELD_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_]+)=(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)|nan|inf|-inf)"
)

@dataclass(frozen=True)
class LossPoint:
    """One parsed epoch from the HGT training log."""

    epoch: int
    train_loss: float
    val_loss: float


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _parse_value(raw: str) -> float:
    value = float(raw)
    return value


def parse_loss_log(log_path: Path) -> list[LossPoint]:
    """Parse a training log into ordered train/validation loss points."""

    points: list[LossPoint] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = _EPOCH_RE.match(line.strip())
        if match is None:
            continue

        fields = {
            field_match.group("key"): _parse_value(field_match.group("value"))
            for field_match in _FIELD_RE.finditer(match.group("body"))
        }
        train_loss = fields.get("train_loss")
        val_loss = fields.get("val_loss")
        if train_loss is None or val_loss is None:
            continue
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            continue
        points.append(
            LossPoint(
                epoch=int(match.group("epoch")),
                train_loss=train_loss,
                val_loss=val_loss,
            )
        )
    return points


def _nice_value(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _plot_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
    x_min: float,
    x_span: float,
    y_min: float,
    y_span: float,
    color: tuple[int, int, int, int],
    width: int = 4,
) -> None:
    if len(points) == 1:
        x, y = points[0]
        px = plot_left + round((x - x_min) / x_span * plot_width)
        py = plot_top + plot_height - round((y - y_min) / y_span * plot_height)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline=(255, 255, 255, 255))
        return

    scaled: list[tuple[int, int]] = []
    for x, y in points:
        px = plot_left + round((x - x_min) / x_span * plot_width)
        py = plot_top + plot_height - round((y - y_min) / y_span * plot_height)
        scaled.append((px, py))
    draw.line(scaled, fill=color, width=width, joint="curve")
    for px, py in scaled:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)


def render_loss_curve(
    points: list[LossPoint],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Render train and validation loss curves to *output_path*."""

    if not points:
        raise ValueError("No loss points were parsed from the training log")

    width, height = 1600, 960
    canvas = Image.new("RGBA", (width, height), (16, 18, 24, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = _font(30)
    body_font = _font(18)
    small_font = _font(15)

    margin_left = 100
    margin_right = 50
    margin_top = 120
    margin_bottom = 110
    plot_left = margin_left
    plot_top = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height

    epochs = [point.epoch for point in points]
    x_min = float(min(epochs))
    x_max = float(max(epochs))
    x_span = max(1.0, x_max - x_min)

    values = [point.train_loss for point in points] + [point.val_loss for point in points]
    y_min = min(values)
    y_max = max(values)
    padding = max(0.05 * (y_max - y_min), 0.02)
    y_min = max(0.0, y_min - padding)
    y_max = y_max + padding
    y_span = max(1e-6, y_max - y_min)

    # Title block.
    draw.text((48, 28), title, fill=(242, 246, 255, 255), font=title_font)
    best_val = min(points, key=lambda p: p.val_loss)
    last = points[-1]
    draw.text(
        (48, 70),
        (
            f"epochs={len(points)}  "
            f"train_loss={_nice_value(points[-1].train_loss)}  "
            f"val_loss={_nice_value(last.val_loss)}  "
            f"best_val={_nice_value(best_val.val_loss)} @ epoch {best_val.epoch}"
        ),
        fill=(188, 200, 223, 255),
        font=body_font,
    )

    # Legend.
    legend_x = width - 470
    legend_y = 34
    legend_items = [
        ("train loss", (255, 163, 97, 255)),
        ("val loss", (95, 200, 255, 255)),
    ]
    for label, color in legend_items:
        draw.rounded_rectangle(
            (legend_x, legend_y + 4, legend_x + 20, legend_y + 24),
            radius=5,
            fill=color,
        )
        draw.text((legend_x + 28, legend_y + 2), label, fill=(220, 228, 245, 255), font=body_font)
        legend_y += 28

    # Plot frame and grid.
    axis_color = (92, 102, 120, 255)
    grid_color = (44, 50, 62, 255)
    draw.rounded_rectangle(
        (plot_left - 2, plot_top - 2, plot_right + 2, plot_bottom + 2),
        radius=10,
        outline=(70, 78, 92, 255),
        width=2,
    )

    y_ticks = 6
    x_ticks = min(10, max(2, len(points) // 10))
    for i in range(y_ticks + 1):
        y = plot_top + round(plot_height * i / y_ticks)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        value = y_max - (y_max - y_min) * i / y_ticks
        label = _nice_value(value)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((plot_left - 12 - (bbox[2] - bbox[0]), y - 9), label, fill=(170, 180, 199, 255), font=small_font)
    for i in range(x_ticks + 1):
        x = plot_left + round(plot_width * i / x_ticks)
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
        epoch = round(x_min + (x_max - x_min) * i / x_ticks)
        label = str(epoch)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (bbox[2] - bbox[0]) // 2, plot_bottom + 10), label, fill=(170, 180, 199, 255), font=small_font)

    draw.text((plot_left, plot_bottom + 38), "epoch", fill=(170, 180, 199, 255), font=small_font)
    draw.text((22, plot_top + plot_height // 2 - 18), "loss", fill=(170, 180, 199, 255), font=small_font)

    # Axes.
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)

    train_points = [(float(point.epoch), point.train_loss) for point in points]
    val_points = [(float(point.epoch), point.val_loss) for point in points]

    _plot_line(
        draw,
        train_points,
        plot_left=plot_left,
        plot_top=plot_top,
        plot_width=plot_width,
        plot_height=plot_height,
        x_min=x_min,
        x_span=x_span,
        y_min=y_min,
        y_span=y_span,
        color=(255, 163, 97, 255),
    )
    _plot_line(
        draw,
        val_points,
        plot_left=plot_left,
        plot_top=plot_top,
        plot_width=plot_width,
        plot_height=plot_height,
        x_min=x_min,
        x_span=x_span,
        y_min=y_min,
        y_span=y_span,
        color=(95, 200, 255, 255),
    )

    # Best validation epoch marker.
    best_x = plot_left + round((best_val.epoch - x_min) / x_span * plot_width)
    best_y = plot_top + plot_height - round((best_val.val_loss - y_min) / y_span * plot_height)
    draw.ellipse((best_x - 7, best_y - 7, best_x + 7, best_y + 7), outline=(255, 255, 255, 255), width=2)
    draw.text(
        (min(best_x + 14, plot_right - 180), max(plot_top + 8, best_y - 20)),
        f"best val @ {best_val.epoch}",
        fill=(224, 232, 246, 255),
        font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    return output_path


def _default_output_path(input_path: Path) -> Path:
    model_name = input_path.parent.name or input_path.stem
    return Path("out/visualizations/loss-curves") / f"{model_name}-loss-curve.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Training log to plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: out/visualizations/loss-curves/<model>-loss-curve.png)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional chart title override",
    )
    args = parser.parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        parser.error(f"input log not found: {input_path}")

    points = parse_loss_log(input_path)
    if not points:
        parser.error(f"no loss points parsed from: {input_path}")

    output_path = args.output if args.output is not None else _default_output_path(input_path)
    title = args.title if args.title is not None else f"{input_path.parent.name} loss curve"
    rendered_path = render_loss_curve(points, output_path, title=title)
    print(rendered_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
