"""Cosmoteer `.ship.png` extractor and object decoder.

The embedded ship payload is stored in the least-significant bits of the PNG's
RGB channels. The first four decoded bytes contain a big-endian payload length,
followed by a gzip-compressed Cosmoteer object stream.
"""

from __future__ import annotations

import gzip
import io
import struct
from pathlib import Path
from typing import Any, BinaryIO, Iterator

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COSMOSHIP_HEADER = b"COSMOSHIP"

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


def _iter_png_chunks(raw: bytes) -> Iterator[tuple[bytes, bytes]]:
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError("File is not a valid PNG")

    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        crc_end = end + 4

        if crc_end > len(raw):
            raise ValueError("PNG chunk extends beyond file length")

        yield chunk_type, raw[start:end]
        offset = crc_end

        if chunk_type == b"IEND":
            break


def _paeth_predictor(left: int, up: int, upper_left: int) -> int:
    initial = left + up - upper_left
    distance_left = abs(initial - left)
    distance_up = abs(initial - up)
    distance_upper_left = abs(initial - upper_left)

    if distance_left <= distance_up and distance_left <= distance_upper_left:
        return left
    if distance_up <= distance_upper_left:
        return up
    return upper_left


def _decode_png_rgb_bytes(raw: bytes) -> bytes:
    if Image is not None:
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA")

            pixel_bytes = image.tobytes()
            if image.mode == "RGB":
                return pixel_bytes

            rgb_bytes = bytearray((len(pixel_bytes) // 4) * 3)
            rgb_bytes[0::3] = pixel_bytes[0::4]
            rgb_bytes[1::3] = pixel_bytes[1::4]
            rgb_bytes[2::3] = pixel_bytes[2::4]
            return bytes(rgb_bytes)

    width = height = bit_depth = color_type = interlace_method = None
    idat_data = bytearray()

    for chunk_type, chunk_data in _iter_png_chunks(raw):
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression_method, filter_method, interlace_method = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression_method != 0 or filter_method != 0:
                raise ValueError("Unsupported PNG compression or filter method")
        elif chunk_type == b"IDAT":
            idat_data.extend(chunk_data)

    if width is None or height is None or bit_depth is None or color_type is None or interlace_method is None:
        raise ValueError("PNG is missing IHDR metadata")
    if interlace_method != 0:
        raise ValueError("Interlaced PNG files are not supported")
    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth: {bit_depth}")

    channels_by_type = {2: 3, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported PNG color type: {color_type}")

    channels = channels_by_type[color_type]
    bytes_per_pixel = channels
    row_length = width * bytes_per_pixel
    inflated = zlib_decompress(bytes(idat_data))
    expected_size = height * (row_length + 1)
    if len(inflated) != expected_size:
        raise ValueError("Unexpected decompressed PNG data length")

    previous_row = bytearray(row_length)
    offset = 0
    rgb_bytes = bytearray(width * height * 3)
    write_offset = 0

    for _ in range(height):
        filter_type = inflated[offset]
        offset += 1
        row = bytearray(inflated[offset : offset + row_length])
        offset += row_length

        if filter_type == 1:
            for index in range(row_length):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                row[index] = (row[index] + left) & 0xFF
        elif filter_type == 2:
            for index in range(row_length):
                row[index] = (row[index] + previous_row[index]) & 0xFF
        elif filter_type == 3:
            for index in range(row_length):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                up = previous_row[index]
                row[index] = (row[index] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for index in range(row_length):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                up = previous_row[index]
                upper_left = previous_row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                row[index] = (row[index] + _paeth_predictor(left, up, upper_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")

        for pixel_offset in range(0, row_length, bytes_per_pixel):
            rgb_bytes[write_offset : write_offset + 3] = row[pixel_offset : pixel_offset + 3]
            write_offset += 3

        previous_row = row

    return bytes(rgb_bytes)


def zlib_decompress(data: bytes) -> bytes:
    import zlib

    return zlib.decompress(data)


def _read_embedded_byte(rgb_bytes: bytes, byte_offset: int) -> int:
    start = byte_offset * 8
    end = start + 8
    if end > len(rgb_bytes):
        raise ValueError("Embedded payload exceeds PNG capacity")

    value = 0
    for bit_index, channel in enumerate(rgb_bytes[start:end]):
        value |= (channel & 1) << bit_index
    return value


def _extract_embedded_payload(png_bytes: bytes) -> bytes:
    rgb_bytes = _decode_png_rgb_bytes(png_bytes)
    payload_length = int.from_bytes(bytes(_read_embedded_byte(rgb_bytes, index) for index in range(4)), "big")

    max_payload_length = (len(rgb_bytes) // 8) - 4
    if payload_length < 0 or payload_length > max_payload_length:
        raise ValueError(f"Invalid embedded payload length: {payload_length}")

    payload = bytes(
        _read_embedded_byte(rgb_bytes, index)
        for index in range(4, 4 + payload_length)
    )

    if payload.startswith(COSMOSHIP_HEADER):
        payload = payload[len(COSMOSHIP_HEADER) :]

    return payload


def _read_varint(stream: BinaryIO) -> int:
    first = stream.read(1)
    if not first:
        raise ValueError("Unexpected end of data while reading integer")

    value = first[0]
    if value & 1 == 0:
        byte_count = 1
    elif value & 2 == 0:
        byte_count = 2
    elif value & 4 == 0:
        byte_count = 3
    else:
        byte_count = 4

    for index in range(1, byte_count):
        current = stream.read(1)
        if not current:
            raise ValueError("Unexpected end of data while reading integer")
        value |= current[0] << (index * 8)

    return value >> min(byte_count, 3)


def _read_length_prefixed_string(stream: BinaryIO) -> str:
    length = 0
    shift = 0

    while True:
        current = stream.read(1)
        if not current:
            raise ValueError("Unexpected end of data while reading string length")
        byte_value = current[0]
        length |= (byte_value & 0x7F) << shift
        if byte_value & 0x80 == 0:
            break
        shift += 7

    data = stream.read(length)
    if len(data) != length:
        raise ValueError("Unexpected end of data while reading string")
    return data.decode("latin1")


def _convert_value_bytes(key: str, value: bytes) -> Any:
    if key == "DefaultAttackRotation" and len(value) == 4:
        return struct.unpack("<f", value)[0]

    if key == "DefaultAttackRadius" and len(value) == 4:
        return struct.unpack("<I", value)[0]

    if key == "Value" and len(value) == 1:
        return bool(value[0])

    if key == "Value" and len(value) == 4:
        return struct.unpack("<I", value)[0]

    if key in {"Location", "Cell", "Key"} and len(value) == 8:
        return list(struct.unpack("<ll", value))

    if key in {"FlipX", "FlipY"} and len(value) == 1:
        return bool(value[0])

    if key in _INT32_KEYS and len(value) == 4:
        return struct.unpack("<i", value)[0]

    if key in _STRING_KEYS:
        decoded = _try_decode_embedded_string(value)
        if decoded is not None:
            return decoded

    if key in _COLOR_KEYS and len(value) == 16:
        return [
            value[0:4].hex().upper(),
            value[4:8].hex().upper(),
            value[8:12].hex().upper(),
            value[12:16].hex().upper(),
        ]

    if len(value) == 0:
        return ""
    if len(value) == 1:
        return value[0]
    if len(value) == 4:
        return struct.unpack("<i", value)[0]

    return {"__bytes__": value.decode("latin1")}


def _try_decode_embedded_string(value: bytes) -> str | None:
    stream = io.BytesIO(value)
    try:
        decoded = _read_length_prefixed_string(stream)
    except ValueError:
        return None
    if stream.tell() != len(value):
        return None
    return decoded


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, bytes):
        decoded_string = _try_decode_embedded_string(value)
        if decoded_string is not None:
            return decoded_string
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return value[0]
        if len(value) == 4:
            return struct.unpack("<i", value)[0]
        return {"__bytes__": value.decode("latin1")}

    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]

    if isinstance(value, dict):
        return {key: _normalize_for_json(item) for key, item in value.items()}

    return value


def _decode_object(stream: BinaryIO) -> Any:
    node_type_raw = stream.read(1)
    if not node_type_raw:
        raise ValueError("Unexpected end of data while reading object type")

    node_type = node_type_raw[0]

    if node_type == 0:
        return "Unset"

    if node_type == 1:
        payload_length = _read_varint(stream)
        payload = stream.read(payload_length)
        if len(payload) != payload_length:
            raise ValueError("Unexpected end of data while reading data payload")
        return payload

    if node_type == 2:
        item_count = _read_varint(stream)
        return [_decode_object(stream) for _ in range(item_count)]

    if node_type == 3:
        child_count = _read_varint(stream)
        output: dict[str, Any] = {}
        for _ in range(child_count):
            key = _read_length_prefixed_string(stream)
            value = _decode_object(stream)
            if isinstance(value, bytes):
                output[key] = _convert_value_bytes(key, value)
            else:
                output[key] = value
        return output

    if node_type == 4:
        link_type_raw = stream.read(1)
        if not link_type_raw:
            raise ValueError("Unexpected end of data while reading link type")
        link_type = link_type_raw[0]
        if link_type == 255:
            return {"_type": "link", "_id": _read_varint(stream)}
        if link_type == 254:
            return None
        raise ValueError(f"Unknown link type: {link_type}")

    if node_type == 5:
        return None

    raise ValueError(f"Unsupported Cosmoteer object type: {node_type}")


def _decode_ship_data(payload: bytes) -> Any:
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed_stream:
        decompressed = compressed_stream.read()

    return _normalize_for_json(_decode_object(io.BytesIO(decompressed)))


def parse_ship_png(path: str | Path) -> Any:
    """Parse a `.ship.png` file and return decoded ship data."""

    source = Path(path)
    payload = _extract_embedded_payload(source.read_bytes())
    return _decode_ship_data(payload)
