# ship-diffusion

Utilities for:
- downloading every accessible Discord `.ship.png` attachment from a target guild, and
- extracting ship JSON data from downloaded files.

## Files

- `main.py`: top-level pipeline runner that can execute download then extract.
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

## Pipeline runner

Run the full pipeline (download first, then extract):

```bash
python main.py --download-output-dir downloaded_ships --extract-output-dir extracted_ship_data --verbose
```

Available flags:
- `--download-output-dir` (default: `downloaded_ships`)
- `--extract-output-dir` (default: `extracted_ship_data`)
- `--skip-download`
- `--skip-extract`
- `--verbose`

Notes:
- Extraction input is the download output directory.
- `.env` token loading is only used by the download step.

## Script 1: Download `.ship.png` images from Discord

This script targets guild ID `546229904488923141` and scans all accessible text channels, their threads (active + archived), and forum threads (active + archived) where the bot can read history.

```bash
python scripts/download_ship_images.py --output-dir downloaded_ships --verbose
```

Behavior:
- Enumerates all accessible text channels in the guild.
- Enumerates active and archived threads for those text channels, and active/archived threads in accessible forum channels.
- Walks complete history for each channel/thread with `history(limit=None, oldest_first=True)` (discord.py handles pagination/rate limits).
- Downloads only attachments ending in `.ship.png`.
- Preserves original filenames; if a filename already exists locally, appends `__msg<message_id>`.
- Logs periodic progress.

Permissions/intents notes:
- Keep `Server Members Intent` disabled; this script only requires `Guilds` and `Messages` intents.
- The bot needs `View Channel` and `Read Message History` in each text/forum context to scan messages.
- For private threads, the bot can only scan archived threads it has joined.

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
