"""Patch the Author field of a .ship.png while preserving the original image.

Re-embeds the modified payload into the original PNG pixel data rather than
generating a new minimal image, so the ship screenshot is preserved exactly.

Usage:
    python scripts/patch_ship_author.py <input.ship.png> <output.ship.png> <author>
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.cosmoteer.encoder import COSMOSHIP_HEADER, encode_ship_data
from common.cosmoteer.parser import parse_ship_png

__all__ = ["patch_author"]

try:
    from PIL import Image
except ImportError:
    Image = None


def patch_author(
    input_path: str | Path,
    output_path: str | Path,
    new_author: str,
) -> bool:
    """Patch the Author field in a .ship.png, preserving the original image pixels.

    Reads the original PNG, modifies only the Author field in the embedded ship
    payload, then re-embeds the payload into the original image's LSBs and saves.

    Returns True if the author was changed, False if it already matched.
    Writes the output file either way.
    """
    if Image is None:
        raise RuntimeError("Pillow is required: pip install Pillow")

    input_path = Path(input_path)
    output_path = Path(output_path)

    original_bytes = input_path.read_bytes()
    ship_data = parse_ship_png(input_path)

    if ship_data.get("Author") == new_author:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(original_bytes)
        return False

    ship_data["Author"] = new_author

    compressed = encode_ship_data(ship_data)
    full_payload = COSMOSHIP_HEADER + compressed
    length_prefix = len(full_payload).to_bytes(4, "big")
    embedded = length_prefix + full_payload

    with Image.open(io.BytesIO(original_bytes)) as img:
        mode = img.mode
        if mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
            mode = "RGBA"

        width, height = img.size
        pixel_bytes = bytearray(img.tobytes())
        bpp = 4 if mode == "RGBA" else 3

        # 3 RGB channels per pixel, 8 payload bits per channel byte, minus 4-byte header
        capacity = (width * height * 3) // 8 - 4
        if len(full_payload) > capacity:
            raise ValueError(
                f"{input_path.name}: payload {len(full_payload)} bytes "
                f"exceeds image capacity {capacity} bytes"
            )

        # Embed payload bytes into LSBs of RGB channels; alpha channel is untouched.
        # Bit layout matches parser._read_embedded_byte and encoder._embed_payload_into_rgb:
        # payload byte N occupies rgb_bytes[N*8 .. N*8+8], one bit per channel byte.
        for byte_index, byte_value in enumerate(embedded):
            for bit_index in range(8):
                rgb_pos = byte_index * 8 + bit_index
                rgb_pixel = rgb_pos // 3
                rgb_channel = rgb_pos % 3   # 0=R, 1=G, 2=B
                byte_pos = rgb_pixel * bpp + rgb_channel
                bit = (byte_value >> bit_index) & 1
                pixel_bytes[byte_pos] = (pixel_bytes[byte_pos] & 0xFE) | bit

        result_img = Image.frombytes(mode, (width, height), bytes(pixel_bytes))
        out_buf = io.BytesIO()
        result_img.save(out_buf, format="PNG")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(out_buf.getvalue())
    return True


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    parser.add_argument("input", help="Input .ship.png file")
    parser.add_argument("output", help="Output .ship.png file")
    parser.add_argument("author", help="New author name to set")
    args = parser.parse_args(argv)

    changed = patch_author(args.input, args.output, args.author)
    status = "patched" if changed else "unchanged"
    print(f"[{status}] {args.input}  Author={args.author!r}")


if __name__ == "__main__":
    main()
