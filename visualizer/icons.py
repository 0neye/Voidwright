"""Part-icon loading helpers for generation visualization."""

from __future__ import annotations

from pathlib import Path

from common.cosmoteer_install import resolve_terran_part_icons_root
from common.geometry import (
    FLIP_H_PART_IDS,
    FLIP_H_ROTATE,
    PART_ID_ALIASES,
    load_vanilla_part_geometry,
)

__all__ = ["PartIconLibrary", "load_part_icon_library"]


def _require_pillow():
    """Import Pillow lazily so visualization stays optional until used."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Visualization requires Pillow. Install the 'Pillow' package to use --visualize."
        ) from exc
    return Image, ImageDraw


def _resolve_icon_transform(
    part_id: str,
    rotation: int,
    *,
    flip_x: bool,
    flip_y: bool,
) -> tuple[str, int, bool, bool]:
    """Resolve icon lookup ID plus local-space transform flags."""

    resolved_part_id = PART_ID_ALIASES.get(part_id, part_id)
    resolved_rotation = int(rotation) % 4
    resolved_flip_x = flip_x
    resolved_flip_y = flip_y

    # `_R` wedge variants are stored as the base part plus a local horizontal
    # flip. Keep that flip explicit for sprite rendering so asymmetric icons are
    # mirrored correctly instead of relying on geometry-only rotation remaps.
    if resolved_part_id in FLIP_H_PART_IDS:
        resolved_part_id = FLIP_H_PART_IDS[resolved_part_id]
        resolved_rotation = FLIP_H_ROTATE[resolved_rotation]
        resolved_flip_x = not resolved_flip_x

    return resolved_part_id, resolved_rotation, resolved_flip_x, resolved_flip_y


class PartIconLibrary:
    """Load and transform vanilla part icons for visualization frames."""

    def __init__(self, icons_root: Path, *, cell_size: int = 96):
        self.icons_root = Path(icons_root)
        self.cell_size = cell_size
        self.geometry_cache = load_vanilla_part_geometry()
        self._base_icon_cache: dict[str, object] = {}
        self._transformed_icon_cache: dict[tuple[str, int, bool, bool], object] = {}

    def _load_base_icon(self, part_id: str):
        Image, ImageDraw = _require_pillow()
        icon_path = self.icons_root / part_id.removeprefix("cosmoteer.") / "icon.png"
        if icon_path.exists():
            with Image.open(icon_path) as icon_image:
                return icon_image.convert("RGBA")

        geometry = self.geometry_cache.get(part_id)
        width = 1
        height = 1
        if geometry is not None:
            rotation_geometry = geometry.rotations.get(0) or next(iter(geometry.rotations.values()))
            width = rotation_geometry.width
            height = rotation_geometry.height
        fallback = Image.new(
            "RGBA",
            (max(1, width * self.cell_size), max(1, height * self.cell_size)),
            (60, 67, 82, 255),
        )
        draw = ImageDraw.Draw(fallback)
        draw.rectangle(
            (0, 0, fallback.width - 1, fallback.height - 1),
            outline=(255, 180, 80, 255),
            width=3,
        )
        draw.text((10, 10), part_id.removeprefix("cosmoteer.")[:18], fill=(255, 240, 210, 255))
        return fallback

    def get_icon(
        self,
        part_id: str,
        rotation: int,
        *,
        flip_x: bool = False,
        flip_y: bool = False,
    ):
        """Return a transformed RGBA icon for one placed part."""

        Image, _ImageDraw = _require_pillow()
        (
            resolved_part_id,
            resolved_rotation,
            resolved_flip_x,
            resolved_flip_y,
        ) = _resolve_icon_transform(
            part_id,
            rotation,
            flip_x=flip_x,
            flip_y=flip_y,
        )
        cache_key = (
            resolved_part_id,
            resolved_rotation,
            resolved_flip_x,
            resolved_flip_y,
        )
        if cache_key in self._transformed_icon_cache:
            return self._transformed_icon_cache[cache_key].copy()

        base_icon = self._base_icon_cache.get(resolved_part_id)
        if base_icon is None:
            base_icon = self._load_base_icon(resolved_part_id)
            self._base_icon_cache[resolved_part_id] = base_icon

        transformed_icon = base_icon.copy()
        # Cosmoteer-style FlipX / FlipY are part-local transforms, so apply them
        # before rotation instead of mirroring across the final screen axes.
        if resolved_flip_x:
            transformed_icon = transformed_icon.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if resolved_flip_y:
            transformed_icon = transformed_icon.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if resolved_rotation % 4:
            transformed_icon = transformed_icon.rotate(-90 * (resolved_rotation % 4), expand=True)

        geometry = self.geometry_cache.get(resolved_part_id)
        if geometry is not None:
            rotated_geometry = geometry.rotations.get(resolved_rotation) or next(iter(geometry.rotations.values()))
            target_size = (
                max(1, rotated_geometry.width * self.cell_size),
                max(1, rotated_geometry.height * self.cell_size),
            )
            transformed_icon = transformed_icon.resize(target_size, Image.Resampling.LANCZOS)

        self._transformed_icon_cache[cache_key] = transformed_icon
        return transformed_icon.copy()


def load_part_icon_library(
    *,
    icons_root: str | Path | None = None,
    game_root: str | Path | None = None,
    cell_size: int = 96,
) -> PartIconLibrary:
    """Resolve the icon root and build a reusable icon library."""

    resolved_icons_root = resolve_terran_part_icons_root(
        icons_root=icons_root,
        game_root=game_root,
    )
    return PartIconLibrary(resolved_icons_root, cell_size=cell_size)
