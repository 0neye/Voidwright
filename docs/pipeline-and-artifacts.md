# Pipeline And Artifacts

## Repository purpose

This repository builds a ship corpus and a first-pass generator around Cosmoteer `.ship.png` files.
The main workflow is:

1. Optionally download visible ship attachments with a standalone Discord script
2. Run the preprocessing module on local `.ship.png` inputs
3. Train a backend-specific model from preprocessing outputs
4. Generate finished encoded ship files from a trained model

## Key entrypoints

- `scripts/download_ship_images.py`
  - Standalone Discord acquisition utility for building a local `.ship.png` corpus
- `preprocessing/cli.py`
  - Local `.ship.png` -> extracted JSON -> canonical JSON -> graph JSON pipeline
- `training/cli.py`
  - Backend-agnostic training router
- `generator/cli.py`
  - Backend-agnostic generation router

## Download step

The Discord downloader is stateful and resumable, but it is no longer part of the core preprocessing contract.

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

The extractor scans recursively for local ship images and writes one JSON file per input image.

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

Corpus snapshot preserved from the historical extraction and canonicalization workflow:

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

`preprocessing/graphs.py` generates graph-oriented views of canonical ship JSON files.
See `docs/ship-graphs.md` for schema details and assumptions.

- The full preprocessing pipeline starts from local `.ship.png` files and ends here
- Final graph outputs are intended to feed the training module
- Intermediate extracted and canonical outputs can optionally be persisted
- Produces per-ship JSON plus a `manifest.json`
- Includes:
  - structural part-touching edges
  - cell-level traversability graph
  - door edge validity counts
  - unknown part ID reporting when geometry has to fall back to heuristics

## Markov artifact conventions

Preferred model artifacts now live under `models/markov/`.

Common files:

- `models/markov/markov-model.v2.json`
- `models/markov/coordinate-validation.v2.json`
- `out/generated-ships/sample-000.ship.png`

The shared and backend-specific implementation now lives primarily under:

- `markov/model.py`
- `markov/symmetry.py`
- `markov/inputs.py`
- `preprocessing/door_rules_engine.py`
- `generator/backends/markov/export.py`
- `common/cosmoteer/`
- `common/data/`
