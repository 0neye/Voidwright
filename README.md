# ship-generator

Utilities for:
- downloading every accessible Discord `.ship.png` attachment from a target guild,
- extracting ship JSON data from downloaded files,
- canonicalizing the extracted corpus,
- inferring reusable door-placement rules from the canonical deduped corpus, and
- building a first-pass vanilla-only relative-placement Markov ship generator.

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

This script targets guild ID `546229904488923141` and scans only configured channel IDs. By default, it uses this built-in allowlist:

- `546229904488923145`
- `546947839008440330`
- `546907635149045775`
- `1297445027835936779`
- `1101149194498089051`
- `546329445032787987`
- `546327169014431746`
- `546327605738209291`
- `546333653605679104`
- `546333675906662400`
- `1318750623302418452`
- `561981489357651980`
- `1240185912525324300`
- `546555689464627212`

```bash
python scripts/download_ship_images.py --output-dir downloaded_ships --verbose
```

Optional channel flags:

```bash
# Repeat --channel-id as needed
python scripts/download_ship_images.py --channel-id 546229904488923145 --channel-id 546947839008440330

# Or load IDs from file (one ID per line; blank lines and # comments allowed)
python scripts/download_ship_images.py --channels-file channel_ids.txt

# Combine both
python scripts/download_ship_images.py --channel-id 546229904488923145 --channels-file channel_ids.txt
```

Behavior:
- Resolves each configured ID in the guild and handles:
  - `TextChannel`: scans the channel itself plus active + archived threads.
  - `ForumChannel`: scans active + archived forum threads.
  - `Thread`: scans the thread directly.
  - Missing/inaccessible IDs are logged as warnings and skipped.
- Walks history with `history(limit=None, oldest_first=True, after=...)` to support resume.
- Downloads only attachments ending in `.ship.png`.
- Preserves original filenames; if a filename already exists locally, appends `__msg<message_id>`.
- Persists resume state to `downloaded_ships/state.json` (or `<output-dir>/state.json`) with per-target `last_message_id` checkpoints and tracked downloaded filenames/attachment IDs.
- Saves state frequently during scanning and after downloads, so restart/resume continues near the last processed message.
- Retries transient network/history/download failures with exponential backoff (e.g. DNS/socket/aiohttp/discord HTTP errors).
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
- Recursively finds both `*.ship.png` and `*.ship__msg<digits>.png` in the input directory.
- Parses embedded ship payload from PNG text chunks.
- Writes one JSON file per input by preserving the full PNG basename and swapping only the final `.png` for `.json`:
  - `foo.ship.png` -> `foo.ship.json`
  - `foo.ship__msg123.png` -> `foo.ship__msg123.json`

## Door rule inference

Infer reusable door-placement rules from the canonical corpus only:

```bash
python scripts/infer_door_rules.py \
  --input-dir extracted_ship_data_canonical \
  --output generators/markov/data/door-placement-rules.v1.json
```

This step streams one canonical ship JSON at a time, derives observed door placements relative to neighboring part footprints/rotations, saves a machine-readable rules file for later runtime filtering, and validates the inferred rules back against the same canonical corpus.

## First-pass vanilla-only Markov generator

Build the relative-placement Markov artifact from the canonical corpus only:

```bash
python scripts/build_markov_generator.py build \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/markov-model.v2.json \
  --validation-output out/markov/coordinate-validation.v2.json
```

Generate sample layouts from a built artifact:

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-v2 \
  --count 5
```

Validate the relative-coordinate assumption against the real canonical corpus:

```bash
python scripts/build_markov_generator.py validate \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/coordinate-validation.v2.json
```

See `generators/markov/README.md` for model details, limitations, and artifact conventions.

## Attribution

`ship_parser/cosmoteer_ship_parser.py` is an adapted minimal parser implementation intended to match the extraction approach used by the `franklin050187/cosmo-api` project, while keeping only the code needed for this repository.
