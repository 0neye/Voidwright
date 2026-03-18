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


_POST_ROTATION_FLIP_PART_IDS = frozenset(
    {
        "cosmoteer.armor_tri",
        "cosmoteer.structure_tri",
        "cosmoteer.armor_structure_hybrid_tri",
    }
)
_PRE_FLIP_ROTATION_REMAP_PART_IDS = frozenset(
    {
        "cosmoteer.armor_wedge",
        "cosmoteer.structure_wedge",
        "cosmoteer.armor_structure_hybrid_1x1",
    }
)
_FLIP_X_ROTATION_REMAP = {0: 1, 1: 0, 2: 3, 3: 2}


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

    if resolved_part_id in _PRE_FLIP_ROTATION_REMAP_PART_IDS and resolved_flip_x:
        # Mirrored 1x1 wedges are saved with a handedness-swapped rotation plus
        # FlipX. Undo that saved-rotation remap here so sprite transforms match
        # the same in-game orientation as generated/exported ship parts.
        resolved_rotation = _FLIP_X_ROTATION_REMAP[resolved_rotation]

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
        part_dir = self.icons_root / part_id.removeprefix("cosmoteer.")
        for icon_name in ("blueprints.png", "icon.png"):
            try:
                with Image.open(part_dir / icon_name) as icon_image:
                    return icon_image.convert("RGBA")
            except FileNotFoundError:
                pass

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
        try:
            from PIL import ImageFont
            _font = ImageFont.load_default(size=50)
        except TypeError:
            from PIL import ImageFont
            _font = ImageFont.load_default()
        draw.text((10, 10), part_id.removeprefix("cosmoteer.")[:18], fill=(255, 240, 210, 255), font=_font)
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
        use_post_rotation_flip_x = resolved_part_id in _POST_ROTATION_FLIP_PART_IDS
        if (
            not use_post_rotation_flip_x
            and resolved_part_id in _PRE_FLIP_ROTATION_REMAP_PART_IDS
            and resolved_flip_x
            and (resolved_rotation % 2 == 1)
        ):
            # Remapped 1x1 wedge mirrors saved as rotation 0/2 + FlipX become
            # odd rotations after `_FLIP_X_ROTATION_REMAP`. Applying FlipX
            # before rotation makes those placements look like a vertical flip
            # in screen space; applying FlipX after rotation matches ship-file
            # mirror parity in the visualizer.
            use_post_rotation_flip_x = True
        # Most vanilla sprites treat FlipX / FlipY as local-space transforms, so
        # apply them before rotation. Half-cell triangle icons are exported on a
        # padded canvas where the game-facing mirrored look matches rotating the
        # icon first and then flipping it horizontally, but FlipY remains a
        # local-space transform for those same parts.
        if not use_post_rotation_flip_x and resolved_flip_x:
            transformed_icon = transformed_icon.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if resolved_flip_y:
            transformed_icon = transformed_icon.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if resolved_rotation % 4:
            transformed_icon = transformed_icon.rotate(-90 * (resolved_rotation % 4), expand=True)
        if use_post_rotation_flip_x and resolved_flip_x:
            transformed_icon = transformed_icon.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

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
