"""Part-ID vocabulary registry for the HGT backend."""

from __future__ import annotations

from pathlib import Path
import orjson

# Re-export zone/weapon constants from their canonical sources so consumers of
# this module get a stable single import point without duplicating definitions.
from graph_expansion.passes.spatial_zones import ZONE_NAMES as ZONE_LABELS
from graph_expansion.passes.spatial_zones import ZONE_NAMES_ROTATED as ZONE_ROT_LABELS
from graph_expansion.passes.weapon_groups import WEAPON_TYPE_SUBSTRINGS as WEAPON_TYPES

__all__ = ["VocabRegistry", "WEAPON_TYPES", "ZONE_LABELS", "ZONE_ROT_LABELS"]

_MASK_TOKEN = "<mask>"
_UNK_TOKEN = "<unk>"

_GEOMETRY_PATH = Path(__file__).parents[3] / "common" / "data" / "vanilla_parts_full_geometry.json"

_SKIP_FILENAMES = frozenset({"manifest.json", "rejections.jsonl"})


class VocabRegistry:
    """Maps part_id strings to integer indices.

    Indices are assigned as: vanilla parts (0..V-1), modded parts (V..V+M-1),
    ``<unk>`` (V+M), ``<mask>`` (V+M+1). All lookups normalize to lowercase.

    The ``<mask>`` token is used by the training loop to replace masked part_id
    values; it is never a valid prediction target.
    """

    def __init__(self, part_ids: list[str], in_geometry: set[str]) -> None:
        # part_ids must end with [<unk>, <mask>]
        self._ids = part_ids
        self._in_geometry = in_geometry
        self._index: dict[str, int] = {pid: i for i, pid in enumerate(part_ids)}
        self.unk_idx: int = self._index[_UNK_TOKEN]
        self.mask_idx: int = self._index[_MASK_TOKEN]

    def __len__(self) -> int:
        """Total vocabulary size including <unk> and <mask>."""
        return len(self._ids)

    @property
    def num_classes(self) -> int:
        """Number of predictable classes (excludes the <mask> token)."""
        return self.mask_idx  # mask is last; everything before it is a valid class

    def encode(self, part_id: str) -> int:
        """Return the integer index for *part_id*, falling back to unk."""
        return self._index.get(part_id.lower(), self.unk_idx)

    def decode(self, idx: int) -> str:
        """Return the part_id string for *idx*."""
        return self._ids[idx]

    def is_vanilla(self, part_id: str) -> bool:
        return part_id.lower() in self._in_geometry

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "part_ids": self._ids,
            "in_geometry": sorted(self._in_geometry),
        }
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    @classmethod
    def load(cls, path: Path) -> "VocabRegistry":
        payload = orjson.loads(path.read_bytes())
        return cls(payload["part_ids"], set(payload["in_geometry"]))

    @classmethod
    def build_from_corpus(cls, input_dir: Path) -> "VocabRegistry":
        """Scan *input_dir* to discover all part_id values and build a registry.

        Vanilla parts (from the geometry JSON) are indexed first for stability,
        then modded parts (alphabetically), then ``<unk>``, then ``<mask>``.

        Virtual flipped-wedge tokens (e.g. ``cosmoteer.armor_1x2_wedge__flipped``)
        are injected alongside their base IDs so the model can predict flipped
        wedges as distinct part types rather than via a separate binary head.
        """
        from common.geometry import (
            FLIP_H_PART_IDS,
            FLIPPABLE_PART_IDS,
            FLIPPED_PART_ID_SUFFIX,
        )

        geo = orjson.loads(_GEOMETRY_PATH.read_bytes())
        base_vanilla: set[str] = {p["id"].lower() for p in geo["parts"]}
        # Add a virtual __flipped token for every flippable wedge base ID so
        # convert_graph() can encode flipped wedges as distinct vocabulary entries.
        flipped_virtual: set[str] = {
            pid.lower() + FLIPPED_PART_ID_SUFFIX for pid in FLIPPABLE_PART_IDS
        }
        vanilla_ids = sorted(base_vanilla | flipped_virtual)
        in_geometry: set[str] = set(vanilla_ids)
        # FlipH alias keys (e.g. cosmoteer.armor_1x2_wedge_R) are vanilla parts
        # stored under their raw IDs in graph JSON because normalize_part_id
        # cannot simultaneously remap rotation.  Include them as vanilla so
        # they don't inflate the modded count.
        flip_h_vanilla: set[str] = {k.lower() for k in FLIP_H_PART_IDS}

        modded: set[str] = set()
        for path in sorted(input_dir.iterdir()):
            if path.suffix != ".json" or path.name in _SKIP_FILENAMES:
                continue
            try:
                payload = orjson.loads(path.read_bytes())
            except Exception:
                continue
            for node in (
                payload.get("graphs", {})
                .get("A_structural_part_graph", {})
                .get("nodes", [])
            ):
                pid = node.get("part_id", "").lower()
                if pid and pid not in in_geometry and pid not in flip_h_vanilla:
                    modded.add(pid)

        part_ids = vanilla_ids + sorted(modded) + [_UNK_TOKEN, _MASK_TOKEN]
        return cls(part_ids, in_geometry)
