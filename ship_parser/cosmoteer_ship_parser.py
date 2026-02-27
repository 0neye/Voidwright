"""Minimal Cosmoteer `.ship.png` payload parser.

Attribution:
This implementation is adapted to match the ship-payload extraction approach used by
community tooling such as `franklin050187/cosmo-api`.

The parser reads PNG text chunks (`tEXt`, `zTXt`, `iTXt`) and attempts to decode ship
JSON payloads using plain JSON text and common compressed/base64 encodings.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class PngChunk:
    chunk_type: bytes
    data: bytes


def _iter_png_chunks(raw: bytes) -> Iterator[PngChunk]:
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError("File is not a valid PNG")

    offset = len(PNG_SIGNATURE)
    raw_len = len(raw)

    while offset + 8 <= raw_len:
        length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        crc_end = end + 4

        if crc_end > raw_len:
            raise ValueError("PNG chunk extends beyond file length")

        yield PngChunk(chunk_type=chunk_type, data=raw[start:end])

        offset = crc_end
        if chunk_type == b"IEND":
            return


def _parse_text_chunk(chunk: PngChunk) -> list[bytes]:
    data = chunk.data
    outputs: list[bytes] = []

    if chunk.chunk_type == b"tEXt":
        if b"\x00" not in data:
            return outputs
        _, text = data.split(b"\x00", 1)
        outputs.append(text)

    elif chunk.chunk_type == b"zTXt":
        if b"\x00" not in data:
            return outputs
        _, rest = data.split(b"\x00", 1)
        if len(rest) < 2:
            return outputs
        compression_method = rest[0]
        compressed = rest[1:]
        if compression_method != 0:
            return outputs
        try:
            outputs.append(zlib.decompress(compressed))
        except zlib.error:
            return outputs

    elif chunk.chunk_type == b"iTXt":
        parts = data.split(b"\x00", 5)
        if len(parts) != 6:
            return outputs

        compression_flag = parts[1]
        compression_method = parts[2]
        text = parts[5]

        if compression_flag == b"\x01" and compression_method == b"\x00":
            try:
                outputs.append(zlib.decompress(text))
            except zlib.error:
                return outputs
        else:
            outputs.append(text)

    return outputs


def _try_json_from_text(raw: bytes) -> Any | None:
    text = raw.decode("utf-8", errors="ignore").strip()
    if not text:
        return None

    for candidate in _candidate_payloads(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _candidate_payloads(text: str) -> Iterable[str]:
    yield text

    compact = "".join(text.split())
    if compact != text:
        yield compact

    b64_inputs = [text, compact]
    seen: set[str] = set()

    for candidate in b64_inputs:
        if candidate in seen:
            continue
        seen.add(candidate)

        for alt in _decode_base64_variants(candidate):
            yield alt


def _decode_base64_variants(payload: str) -> Iterable[str]:
    normalized = payload + "=" * (-len(payload) % 4)

    variants = [normalized]
    if "-" in normalized or "_" in normalized:
        variants.append(normalized.replace("-", "+").replace("_", "/"))

    for variant in variants:
        try:
            decoded = base64.b64decode(variant, validate=False)
        except Exception:
            continue

        for text in _bytes_to_text_variants(decoded):
            yield text


def _bytes_to_text_variants(raw: bytes) -> Iterable[str]:
    attempts = [raw]

    try:
        attempts.append(zlib.decompress(raw))
    except zlib.error:
        pass

    for blob in attempts:
        try:
            yield blob.decode("utf-8")
        except UnicodeDecodeError:
            continue


def parse_ship_png(path: str | Path) -> Any:
    """Parse a `.ship.png` file and return decoded JSON ship data."""
    path = Path(path)
    data = path.read_bytes()

    for chunk in _iter_png_chunks(data):
        if chunk.chunk_type not in {b"tEXt", b"zTXt", b"iTXt"}:
            continue

        for text_blob in _parse_text_chunk(chunk):
            parsed = _try_json_from_text(text_blob)
            if parsed is not None:
                return parsed

    raise ValueError(f"No ship JSON payload found in {path}")
