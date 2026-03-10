# Pipeline And Artifacts

## Repository purpose

This repository builds a ship corpus and a first-pass generator around Cosmoteer `.ship.png` files.
The main workflow is:

1. Download visible ship attachments from a Discord guild
2. Extract embedded ship JSON payloads from those PNGs
3. Canonicalize and dedupe the extracted corpus
4. Infer reusable door-placement rules from the canonical corpus
5. Build, validate, and sample a vanilla-only Markov ship generator

## Key entrypoints

- `main.py`
  - Runs the download step and then the extraction step
  - Supports `--skip-download`, `--skip-extract`, `--download-output-dir`, `--extract-output-dir`, and `--verbose`
- `scripts/download_ship_images.py`
  - Downloads `.ship.png` attachments from the configured guild and channel/thread targets
- `scripts/extract_ship_data.py`
  - Parses downloaded PNG payloads into `.ship.json` files
- `scripts/canonicalize_ship_json_corpus.py`
  - Dedupes extracted JSON by canonicalized content hash
- `scripts/infer_door_rules.py`
  - Rebuilds the machine-readable door validation artifact from the canonical corpus
- `scripts/build_markov_generator.py`
  - Thin wrapper around the Markov generator CLI
- `scripts/generate_ship_graphs.py`
  - Produces structural and cell-graph JSONs for extracted ships

## Download step

The Discord downloader is stateful and resumable.

- Target guild ID is hard-coded as `546229904488923141`
- If no explicit channel IDs are supplied, the script uses its built-in allowlist
- `TextChannel` targets include the channel plus active and archived threads
- `ForumChannel` targets include active and archived threads
- `Thread` targets are scanned directly
- Resume state is saved to `<output-dir>/state.json`
- Only attachments ending in `.ship.png` are downloaded
- Filename collisions are handled by appending `__msg<message_id>`
- The script retries transient Discord/network/history/download failures with backoff

Operational notes:

- The bot token is loaded from `.env` via `DISCORD_BOT_TOKEN`
- Required intents are `Guilds` and `Messages`
- Required permissions are `View Channel` and `Read Message History`
- Private archived threads are only visible if the bot joined them

## Extraction step

The extractor scans recursively for downloaded ship images and writes one JSON file per input image.

- Accepted inputs:
  - `*.ship.png`
  - `*.ship__msg<digits>.png`
- Output naming preserves the full PNG basename and swaps only the final `.png` for `.json`
- Extraction runs in a thread pool and writes sorted, indented JSON
- Exit code is `0` on full success and `2` if any files fail to parse
- Batch extraction continues past bad files instead of aborting the entire run

Extractor implementation notes:

- The parser reads LSB-embedded payload bytes from PNG RGB data
- It decodes the 4-byte payload length, strips an optional `COSMOSHIP` header, gzip-decompresses the payload, and decodes the Cosmoteer object stream
- There is an optional Pillow-backed path for PNG decoding plus a pure-Python fallback

Corpus snapshot preserved from the extraction and canonicalization workflow:

- extracted JSON corpus size: `15610`
- canonical deduped corpus size: `12913`
- duplicates removed: `2697`

## Canonicalization rules

Canonicalization is content-based, not filename-based.

- Every source JSON is parsed and normalized with stable recursive key ordering
- The canonicalized JSON bytes are hashed with SHA-256
- Deduplication is performed by content hash
- The script keeps only metadata during the scan and re-reads one representative file per content group when writing outputs

Canonical filename rules:

1. Prefer a filename that already exists without a `__msg<digits>` suffix
2. Otherwise strip the `__msg<digits>` suffix from a representative filename
3. If different content groups want the same canonical filename, keep the unsuffixed name for the lexicographically smallest hash
4. Suffix the others as `__dedup-<12 hex>`

Default canonicalization outputs:

- Canonical corpus directory: `extracted_ship_data_canonical`
- Machine-readable report: `out/ship_canonicalization_report.json`
- Human-readable report: `SHIP_CANONICALIZATION_REPORT.md`

Practical implication:

- Filename collisions are common enough that `__dedup-<12 hex>` names are a normal canonicalization outcome, not an exceptional failure mode

## Ship graph generation

`scripts/generate_ship_graphs.py` generates graph-oriented views of extracted ships.
See `docs/ship-graphs.md` for schema details and assumptions.

- Input defaults to `extracted_ship_data`
- Output defaults to `generated_ship_graphs`
- Produces per-ship JSON plus a `manifest.json`
- Includes:
  - structural part-touching edges
  - cell-level traversability graph
  - door edge validity counts
  - unknown part ID reporting when geometry has to fall back to heuristics

## Markov artifact conventions

Typical Markov artifacts live under `out/markov/`.

Common files:

- `out/markov/markov-model.v2.json`
- `out/markov/coordinate-validation.v2.json`
- `out/markov/samples-v2/sample-000.json`
- `out/markov/exported-ships/sample-000.ship.png`
- `out/markov/export-report.json`

The generator-specific reusable assets live under `generators/markov/`, especially:

- `generators/markov/model.py`
- `generators/markov/cli.py`
- `generators/markov/export.py`
- `generators/markov/door_rules.py`
- `generators/markov/data/`
