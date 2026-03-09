"""Cosmoteer `.ship.png` encoder – inverse of the extractor in cosmoteer_ship_parser.py.

The embedded ship payload is stored in the least-significant bits of the PNG's
RGB channels.  Encoding steps:
  1. Serialize the ship dict to the Cosmoteer object stream (binary).
  2. Gzip-compress the stream.
  3. Prepend the 9-byte ``COSMOSHIP`` header.
  4. Prepend a 4-byte big-endian payload-length prefix.
  5. Embed the resulting bytes into the LSBs of a freshly-generated carrier PNG.

Limitations (first-pass encoder):
- Only handles the simple value types actually present in generated vanilla ships:
  int32, string (LEB128-prefixed), coordinate pairs, RGBA color quads, bool.
- ``__bytes__`` passthrough values are re-encoded verbatim.
- Link nodes (``_type: link``) are encoded but are not expected in generated ships.
- Complex nested PartControlGroups / PartUIToggleStates / Decals are accepted but
  must already be expressed as properly typed Python dicts/lists.  Generated ships
  omit these fields entirely, so this is not a blocker.
- The carrier PNG is a freshly-synthesised solid-gray image – the thumbnail is
  cosmetic and meaningless.  The game reads the embedded payload only.
"""

from __future__ import annotations

import gzip
import io
import math
import struct
import zlib
from pathlib import Path
from typing import Any

# ── key-set constants (must match cosmoteer_ship_parser.py) ──────────────────

_INT32_KEYS = {
    "FlightDirection",
    "FormationOrder",
    "ID",
    "Key",
    "Max",
    "Min",
    "Orientation",
    "Rotation",
    "Version",
}

_STRING_KEYS = {
    "Author",
    "ComponentID",
    "Description",
    "ID",
    "IDString",
    "Name",
    "PartID",
    "RoofBaseTexture",
    "ShipRulesID",
    "Value",
}

_COLOR_KEYS = {
    "Color",
    "CrewUniformColor",
    "RoofBaseColor",
    "RoofDecalColor1",
    "RoofDecalColor2",
    "RoofDecalColor3",
}

_COORD_KEYS = {"Location", "Cell", "Key"}

COSMOSHIP_HEADER = b"COSMOSHIP"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ── varint / string encoding ──────────────────────────────────────────────────


def _write_varint(n: int) -> bytes:
    """Encode a non-negative integer using the Cosmoteer varint format.

    The decoder in cosmoteer_ship_parser.py determines byte-count from the low
    bits of the first byte:
        bit0 = 0          → 1 byte,  value = encoded >> 1
        bit0=1, bit1=0    → 2 bytes, value = assembled >> 2
        bit0=1, bit1=1, bit2=0 → 3 bytes, value = assembled >> 3
        bit0=1, bit1=1, bit2=1 → 4 bytes, value = assembled >> 3
    """
    if n < 0:
        raise ValueError(f"varint must be non-negative, got {n}")
    if n < (1 << 7):  # fits in 7 bits → 1 byte
        encoded = n << 1  # bit0 = 0
        return bytes([encoded])
    if n < (1 << 14):  # fits in 14 bits → 2 bytes
        encoded = (n << 2) | 1  # bit0=1, bit1=0
        return bytes([encoded & 0xFF, (encoded >> 8) & 0xFF])
    if n < (1 << 21):  # fits in 21 bits → 3 bytes
        encoded = (n << 3) | 3  # bit0=1, bit1=1, bit2=0
        return bytes([encoded & 0xFF, (encoded >> 8) & 0xFF, (encoded >> 16) & 0xFF])
    if n < (1 << 29):  # fits in 29 bits → 4 bytes
        encoded = (n << 3) | 7  # bit0=1, bit1=1, bit2=1
        return bytes([encoded & 0xFF, (encoded >> 8) & 0xFF, (encoded >> 16) & 0xFF, (encoded >> 24) & 0xFF])
    raise ValueError(f"varint overflow: {n} >= 2^29")


def _write_leb128_string(s: str) -> bytes:
    """Encode a string as LEB128-prefixed latin1 bytes (matches _read_length_prefixed_string)."""
    data = s.encode("latin1")
    length = len(data)
    prefix = bytearray()
    while True:
        byte = length & 0x7F
        length >>= 7
        if length:
            prefix.append(byte | 0x80)
        else:
            prefix.append(byte)
            break
    return bytes(prefix) + data


# ── value encoding (inverse of _convert_value_bytes) ─────────────────────────


def _encode_value_for_key(key: str, value: Any) -> bytes:
    """Return the raw bytes payload for a type-1 (data) node given key context.

    Mirrors the decoding priorities in _convert_value_bytes.
    """
    # __bytes__ passthrough
    if isinstance(value, dict) and "__bytes__" in value:
        return value["__bytes__"].encode("latin1")

    # Coordinate pairs: Location, Cell, or Key=[x,y]
    if key in _COORD_KEYS and isinstance(value, (list, tuple)) and len(value) == 2:
        return struct.pack("<ll", int(value[0]), int(value[1]))

    # FlipX / FlipY: single-byte bool
    if key in {"FlipX", "FlipY"}:
        return bytes([1 if value else 0])

    # Int32 keys (but only when value is int, not when it's a string or list)
    if key in _INT32_KEYS and isinstance(value, int) and not isinstance(value, bool):
        return struct.pack("<i", value)

    # Color keys: list of 4 hex strings → 16 bytes
    if key in _COLOR_KEYS:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return b"".join(bytes.fromhex(c) for c in value)
            except (ValueError, TypeError):
                return b"\x00" * 16
        return b"\x00" * 16

    # String keys: LEB128 prefix + latin1 bytes
    if key in _STRING_KEYS and isinstance(value, str):
        return _write_leb128_string(value)

    # DefaultAttackRotation: float
    if key == "DefaultAttackRotation" and isinstance(value, float):
        return struct.pack("<f", value)

    # DefaultAttackRadius: uint
    if key == "DefaultAttackRadius" and isinstance(value, int):
        return struct.pack("<I", value)

    # Value: bool (1 byte) or int (4 bytes uint)
    if key == "Value":
        if isinstance(value, bool):
            return bytes([1 if value else 0])
        if isinstance(value, int):
            return struct.pack("<I", value)

    # Generic fallbacks
    if isinstance(value, bool):
        return bytes([1 if value else 0])
    if isinstance(value, int):
        return struct.pack("<i", value)
    if isinstance(value, float):
        return struct.pack("<f", value)
    if isinstance(value, str):
        return _write_leb128_string(value)

    raise ValueError(f"Cannot encode primitive for key {key!r}: {value!r} ({type(value).__name__})")


# ── object encoder ────────────────────────────────────────────────────────────


def _encode_object(obj: Any, key_context: str = "") -> bytes:
    """Encode any Python value as a Cosmoteer binary object node.

    Node types (mirrors _decode_object in parser):
        0 – "Unset" sentinel
        1 – raw bytes payload (primitives, strings, coords, colors)
        2 – list of child nodes
        3 – dict (object) with keyed children
        4 – link reference or null link
        5 – null
    """
    if obj == "Unset":
        return b"\x00"

    if obj is None:
        return b"\x05"

    # Link reference
    if isinstance(obj, dict) and obj.get("_type") == "link":
        link_id = int(obj["_id"])
        return b"\x04\xff" + _write_varint(link_id)

    # __bytes__ passthrough → type 1
    if isinstance(obj, dict) and "__bytes__" in obj:
        data = obj["__bytes__"].encode("latin1")
        return b"\x01" + _write_varint(len(data)) + data

    # Regular dict → type 3
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            key_bytes = _write_leb128_string(k)
            child_bytes = _encode_object(v, key_context=k)
            parts.append(key_bytes + child_bytes)
        count_bytes = _write_varint(len(obj))
        return b"\x03" + count_bytes + b"".join(parts)

    # List: coord pair → type 1; color quad → type 1; real list → type 2
    if isinstance(obj, (list, tuple)):
        # Coordinate pair (2 ints, key is a coord key)
        if (
            key_context in _COORD_KEYS
            and len(obj) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in obj)
        ):
            data = struct.pack("<ll", int(obj[0]), int(obj[1]))
            return b"\x01" + _write_varint(len(data)) + data

        # Color quad (4 hex strings, key is a color key)
        if key_context in _COLOR_KEYS and len(obj) == 4 and all(isinstance(x, str) for x in obj):
            try:
                data = b"".join(bytes.fromhex(c) for c in obj)
                return b"\x01" + _write_varint(len(data)) + data
            except (ValueError, TypeError):
                pass

        # Generic list → type 2
        items = [_encode_object(item, key_context=key_context) for item in obj]
        count_bytes = _write_varint(len(items))
        return b"\x02" + count_bytes + b"".join(items)

    # Primitive → type 1
    data = _encode_value_for_key(key_context, obj)
    return b"\x01" + _write_varint(len(data)) + data


# ── ship serialisation ────────────────────────────────────────────────────────


def encode_ship_data(ship_dict: dict) -> bytes:
    """Serialise *ship_dict* to a gzip-compressed Cosmoteer object stream.

    The returned bytes can be prepended with the COSMOSHIP header and a 4-byte
    big-endian length to form the payload that gets LSB-embedded into a PNG.
    """
    obj_bytes = _encode_object(ship_dict)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(obj_bytes)
    return buf.getvalue()


# ── PNG generation ────────────────────────────────────────────────────────────


def _min_image_dims(payload_size: int) -> tuple[int, int]:
    """Return the smallest square (width, height) that can hold *payload_size* bytes.

    The extractor capacity formula is: ``(width * height * 3 // 8) - 4``.
    We need capacity >= payload_size, so:
        width * height >= (payload_size + 4) * 8 / 3
    """
    needed_pixels = math.ceil((payload_size + 4) * 8 / 3)
    side = max(8, math.ceil(math.sqrt(needed_pixels)))
    # Round up to next multiple of 8 for alignment
    side = ((side + 7) // 8) * 8
    # Verify (in rare cases sqrt underestimates)
    while side * side * 3 // 8 - 4 < payload_size:
        side += 8
    return side, side


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def _build_png_from_rgb(width: int, height: int, rgb_bytes: bytes) -> bytes:
    """Build a minimal valid RGB (colour-type 2) PNG from raw RGB byte data."""
    assert len(rgb_bytes) == width * height * 3, (
        f"RGB length mismatch: expected {width * height * 3}, got {len(rgb_bytes)}"
    )
    # Build raw filtered image data: each row prefixed with filter-byte 0 (None)
    row_size = width * 3
    raw = bytearray(height * (row_size + 1))
    for y in range(height):
        raw[y * (row_size + 1)] = 0  # filter type None
        src = y * row_size
        raw[y * (row_size + 1) + 1 : y * (row_size + 1) + 1 + row_size] = rgb_bytes[src : src + row_size]
    idat_data = zlib.compress(bytes(raw), level=6)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr_data)
        + _png_chunk(b"IDAT", idat_data)
        + _png_chunk(b"IEND", b"")
    )


def _embed_payload_into_rgb(rgb: bytearray, payload_with_length: bytes) -> None:
    """Write *payload_with_length* bytes into the LSBs of *rgb* (in-place).

    Each byte of the payload occupies 8 consecutive RGB channel bytes (1 bit each),
    LSB-first – matching _read_embedded_byte in the parser.
    """
    for byte_index, byte_value in enumerate(payload_with_length):
        base = byte_index * 8
        for bit_index in range(8):
            bit = (byte_value >> bit_index) & 1
            rgb[base + bit_index] = (rgb[base + bit_index] & 0xFE) | bit


def create_ship_png_bytes(ship_dict: dict) -> bytes:
    """Encode *ship_dict* and return the bytes of a valid ``.ship.png`` file."""
    compressed = encode_ship_data(ship_dict)
    full_payload = COSMOSHIP_HEADER + compressed
    length_prefix = len(full_payload).to_bytes(4, "big")
    embedded = length_prefix + full_payload

    width, height = _min_image_dims(len(full_payload))

    # Solid-gray carrier: all channels = 0x80 (visually neutral, LSBs = 0)
    rgb = bytearray(b"\x80" * (width * height * 3))

    assert len(rgb) // 8 - 4 >= len(full_payload), (
        f"carrier too small: capacity {len(rgb) // 8 - 4} < payload {len(full_payload)}"
    )

    _embed_payload_into_rgb(rgb, embedded)
    return _build_png_from_rgb(width, height, bytes(rgb))


def write_ship_png(ship_dict: dict, output_path: str | Path) -> None:
    """Write *ship_dict* as a ``.ship.png`` file at *output_path*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(create_ship_png_bytes(ship_dict))
