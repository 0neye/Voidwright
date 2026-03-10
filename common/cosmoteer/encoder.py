"""Cosmoteer `.ship.png` encoder.

This is the inverse of the extractor in `common.cosmoteer.parser`.
"""

from __future__ import annotations

import gzip
import io
import math
import struct
import zlib
from pathlib import Path
from typing import Any

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


def _write_varint(n: int) -> bytes:
    """Encode a non-negative integer using the Cosmoteer varint format."""

    if n < 0:
        raise ValueError(f"varint must be non-negative, got {n}")
    if n < (1 << 7):
        encoded = n << 1
        return bytes([encoded])
    if n < (1 << 14):
        encoded = (n << 2) | 1
        return bytes([encoded & 0xFF, (encoded >> 8) & 0xFF])
    if n < (1 << 21):
        encoded = (n << 3) | 3
        return bytes([encoded & 0xFF, (encoded >> 8) & 0xFF, (encoded >> 16) & 0xFF])
    if n < (1 << 29):
        encoded = (n << 3) | 7
        return bytes(
            [
                encoded & 0xFF,
                (encoded >> 8) & 0xFF,
                (encoded >> 16) & 0xFF,
                (encoded >> 24) & 0xFF,
            ]
        )
    raise ValueError(f"varint overflow: {n} >= 2^29")


def _write_leb128_string(s: str) -> bytes:
    """Encode a string as LEB128-prefixed latin1 bytes."""

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


def _encode_value_for_key(key: str, value: Any) -> bytes:
    """Return the raw bytes payload for a primitive node given key context."""

    if isinstance(value, dict) and "__bytes__" in value:
        return value["__bytes__"].encode("latin1")

    if key in _COORD_KEYS and isinstance(value, (list, tuple)) and len(value) == 2:
        return struct.pack("<ll", int(value[0]), int(value[1]))

    if key in {"FlipX", "FlipY"}:
        return bytes([1 if value else 0])

    if key in _INT32_KEYS and isinstance(value, int) and not isinstance(value, bool):
        return struct.pack("<i", value)

    if key in _COLOR_KEYS:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return b"".join(bytes.fromhex(c) for c in value)
            except (ValueError, TypeError):
                return b"\x00" * 16
        return b"\x00" * 16

    if key in _STRING_KEYS and isinstance(value, str):
        return _write_leb128_string(value)

    if key == "DefaultAttackRotation" and isinstance(value, float):
        return struct.pack("<f", value)

    if key == "DefaultAttackRadius" and isinstance(value, int):
        return struct.pack("<I", value)

    if key == "Value":
        if isinstance(value, bool):
            return bytes([1 if value else 0])
        if isinstance(value, int):
            return struct.pack("<I", value)

    if isinstance(value, bool):
        return bytes([1 if value else 0])
    if isinstance(value, int):
        return struct.pack("<i", value)
    if isinstance(value, float):
        return struct.pack("<f", value)
    if isinstance(value, str):
        return _write_leb128_string(value)

    raise ValueError(f"Cannot encode primitive for key {key!r}: {value!r} ({type(value).__name__})")


def _encode_object(obj: Any, key_context: str = "") -> bytes:
    """Encode any Python value as a Cosmoteer binary object node."""

    if obj == "Unset":
        return b"\x00"

    if obj is None:
        return b"\x05"

    if isinstance(obj, dict) and obj.get("_type") == "link":
        link_id = int(obj["_id"])
        return b"\x04\xff" + _write_varint(link_id)

    if isinstance(obj, dict) and "__bytes__" in obj:
        data = obj["__bytes__"].encode("latin1")
        return b"\x01" + _write_varint(len(data)) + data

    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            key_bytes = _write_leb128_string(k)
            child_bytes = _encode_object(v, key_context=k)
            parts.append(key_bytes + child_bytes)
        count_bytes = _write_varint(len(obj))
        return b"\x03" + count_bytes + b"".join(parts)

    if isinstance(obj, (list, tuple)):
        if (
            key_context in _COORD_KEYS
            and len(obj) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in obj)
        ):
            data = struct.pack("<ll", int(obj[0]), int(obj[1]))
            return b"\x01" + _write_varint(len(data)) + data

        if key_context in _COLOR_KEYS and len(obj) == 4 and all(isinstance(x, str) for x in obj):
            try:
                data = b"".join(bytes.fromhex(c) for c in obj)
                return b"\x01" + _write_varint(len(data)) + data
            except (ValueError, TypeError):
                pass

        items = [_encode_object(item, key_context=key_context) for item in obj]
        count_bytes = _write_varint(len(items))
        return b"\x02" + count_bytes + b"".join(items)

    data = _encode_value_for_key(key_context, obj)
    return b"\x01" + _write_varint(len(data)) + data


def encode_ship_data(ship_dict: dict) -> bytes:
    """Serialize *ship_dict* to a gzip-compressed Cosmoteer object stream."""

    obj_bytes = _encode_object(ship_dict)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(obj_bytes)
    return buf.getvalue()


def _min_image_dims(payload_size: int) -> tuple[int, int]:
    """Return the smallest square image that can hold *payload_size* bytes."""

    needed_pixels = math.ceil((payload_size + 4) * 8 / 3)
    side = max(8, math.ceil(math.sqrt(needed_pixels)))
    side = ((side + 7) // 8) * 8
    while side * side * 3 // 8 - 4 < payload_size:
        side += 8
    return side, side


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build one PNG chunk with length and CRC."""

    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def _build_png_from_rgb(width: int, height: int, rgb_bytes: bytes) -> bytes:
    """Build a minimal valid RGB PNG from raw RGB byte data."""

    assert len(rgb_bytes) == width * height * 3, (
        f"RGB length mismatch: expected {width * height * 3}, got {len(rgb_bytes)}"
    )
    row_size = width * 3
    raw = bytearray(height * (row_size + 1))
    for y in range(height):
        raw[y * (row_size + 1)] = 0
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
    """Write *payload_with_length* bytes into the RGB least-significant bits."""

    for byte_index, byte_value in enumerate(payload_with_length):
        base = byte_index * 8
        for bit_index in range(8):
            bit = (byte_value >> bit_index) & 1
            rgb[base + bit_index] = (rgb[base + bit_index] & 0xFE) | bit


def create_ship_png_bytes(ship_dict: dict) -> bytes:
    """Encode *ship_dict* and return the bytes of a valid `.ship.png` file."""

    compressed = encode_ship_data(ship_dict)
    full_payload = COSMOSHIP_HEADER + compressed
    length_prefix = len(full_payload).to_bytes(4, "big")
    embedded = length_prefix + full_payload

    width, height = _min_image_dims(len(full_payload))
    rgb = bytearray(b"\x80" * (width * height * 3))

    assert len(rgb) // 8 - 4 >= len(full_payload), (
        f"carrier too small: capacity {len(rgb) // 8 - 4} < payload {len(full_payload)}"
    )

    _embed_payload_into_rgb(rgb, embedded)
    return _build_png_from_rgb(width, height, bytes(rgb))


def write_ship_png(ship_dict: dict, output_path: str | Path) -> None:
    """Write *ship_dict* as a `.ship.png` file at *output_path*."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(create_ship_png_bytes(ship_dict))
