# ship-diffusion

Utilities for:
- downloading every accessible Discord `.ship.png` attachment from a target guild, and
- extracting ship JSON data from downloaded files.

## Files

- `scripts/download_ship_images.py`: connects to Discord and downloads matching files.
- `scripts/extract_ship_data.py`: parses downloaded `.ship.png` files into JSON.
- `ship_parser/cosmoteer_ship_parser.py`: vendored/adapted minimal PNG payload parser (with attribution).

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure `.env` contains:

```env
DISCORD_BOT_TOKEN=your_bot_token_here
```

## Script 1: Download `.ship.png` images from Discord

This script targets guild ID `546229904488923141` and scans all text channels where the bot has `View Channel` and `Read Message History` permissions.

```bash
python scripts/download_ship_images.py --output-dir downloaded_ships --verbose
```

Behavior:
- Enumerates all accessible text channels in the guild.
- Walks complete channel history with `history(limit=None, oldest_first=True)` (discord.py handles pagination/rate limits).
- Downloads only attachments ending in `.ship.png`.
- Preserves original filenames; if a filename already exists locally, appends `__msg<message_id>`.
- Logs periodic progress.

## Script 2: Extract JSON ship data from downloaded images

```bash
python scripts/extract_ship_data.py --input-dir downloaded_ships --output-dir extracted_ship_data --verbose
```

Behavior:
- Recursively finds `*.ship.png` in the input directory.
- Parses embedded ship payload from PNG text chunks.
- Writes one JSON file per input using the same base name:
  - `foo.ship.png` -> `foo.ship.json`

## Attribution

`ship_parser/cosmoteer_ship_parser.py` is an adapted minimal parser implementation intended to match the extraction approach used by the `franklin050187/cosmo-api` project, while keeping only the code needed for this repository.
