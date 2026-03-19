# Pipeline And Artifacts

## Repository purpose

This repository builds a ship corpus and a first-pass generator around Cosmoteer `.ship.png` files.
The main workflow is:

1. Optionally download visible ship attachments with a standalone Discord script
2. Run the preprocessing module on local `.ship.png` inputs
3. Optionally filter the graph corpus with the corpus filter (`corpus/cli.py`)
4. Train a backend-specific model from preprocessing outputs
5. Generate finished encoded ship files from a trained model

## Key entrypoints

- `scripts/download_ship_images.py`
  - Standalone Discord acquisition utility for building a local `.ship.png` corpus
- `preprocessing/cli.py`
  - Local `.ship.png` -> extracted JSON -> canonical JSON -> graph JSON pipeline
- `training/cli.py`
  - Backend-agnostic training router
- `generator/cli.py`
  - Backend-agnostic generation router
- `corpus/cli.py`
  - Optional corpus filtering stage (rule-based accept/reject on graph JSON files)

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

Opt-in filtering:

- Pass `--opt-in-csv <path>` to a CSV listing exact `Author` names that should be kept
- Ships whose `Author` does not appear in the list are deleted immediately after download
- The CSV is also applied to any previously downloaded files at startup
- When the CSV is absent or produces an empty list, all ships are kept (filtering disabled)
- Use `scripts/patch_ship_author.py` to fix a blank `Author` field on an individual ship before preprocessing
- The pipeline CLI automatically maintains a stat-keyed author cache at `.ship-filter-cache.json`
  in the first input directory; unchanged files are skipped on subsequent runs without re-parsing

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
- Extraction uses hardware-agnostic parallel workers and defaults to a thread pool
- Exit code is `0` on full success and `2` if any files fail to parse
- Batch extraction continues past bad files instead of aborting the entire run
- `--workers` and `--executor {auto,thread,process}` can override the default concurrency behavior
- `auto` falls back to thread-based execution if process pools are unavailable in the current environment

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
- The scan/hash phase can run in parallel before the deterministic global dedupe and naming pass
- Canonical output writes can also run concurrently after final filenames are resolved
- `--workers` and `--executor {auto,thread,process}` can override the default concurrency behavior
- `auto` falls back to thread-based execution if process pools are unavailable in the current environment

Canonical filename rules:

1. Prefer a filename that already exists without a `__msg<digits>` suffix
2. Otherwise strip the `__msg<digits>` suffix from a representative filename
3. If different content groups want the same canonical filename, keep the unsuffixed name for the lexicographically smallest hash
4. Suffix the others as `__dedup-<12 hex>`

Default canonicalization outputs:

- Canonical corpus directory: `extracted_ship_data_canonical`
- Machine-readable report: `out/ship_canonicalization_report.json`
- Human-readable markdown report: only when `--report-md <path>` is provided

Practical implication:

- Filename collisions are common enough that `__dedup-<12 hex>` names are a normal canonicalization outcome, not an exceptional failure mode

## Ship graph generation

`preprocessing/graphs.py` generates graph-oriented views of canonical ship JSON files.
See `docs/ship-graphs.md` for schema details and assumptions.

- The full preprocessing pipeline starts from local `.ship.png` files and ends here
- Final graph outputs are intended to feed the training module
- Intermediate extracted and canonical outputs can optionally be persisted
- Produces per-ship JSON plus a `manifest.json`
- Graph generation can process ships in parallel, then reduce manifest counters in sorted filename order for deterministic outputs
- Ships that fail during parallel generation are skipped with a printed warning; one bad file does not abort the batch
- `--workers` and `--executor {auto,thread,process}` can override the default concurrency behavior
- `auto` falls back to thread-based execution if process pools are unavailable in the current environment
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

## Corpus filtering stage (optional)

`corpus/cli.py` is a shared, backend-agnostic filtering stage that sits between
graph preprocessing (or graph expansion) and model training.  It reads a
directory of generated ship graph JSON files, evaluates each ship against an
ordered set of pluggable rules, copies accepted ships verbatim to a new output
directory, and writes a `manifest.json` summary.

**Typical invocation:**

```bash
python main.py corpus filter \
  --input-dir generated_ship_graphs_canonical \
  --output-dir filtered_ship_graphs_canonical \
  --max-parts 300 \
  --require-crew-rooms \
  --require-reachable-reactor
```

Or directly:

```bash
python -m corpus.cli \
  --input-dir generated_ship_graphs_canonical \
  --output-dir filtered_ship_graphs_canonical \
  --max-parts 300 \
  --require-crew-rooms \
  --require-reachable-reactor
```

**Output artifacts:**

- `filtered_ship_graphs_canonical/` — accepted graph JSON files copied verbatim
- `filtered_ship_graphs_canonical/manifest.json` — counts, active rules, and
  per-rule rejection totals
- `filtered_ship_graphs_canonical/rejections.jsonl` — one record per rejected
  ship (suppressed with `--no-rejections-log`)

**Built-in rules:**

| Flag | Rule | Notes |
|---|---|---|
| `--max-parts N` | `max_size` | Reject ships with more than N parts |
| `--max-occupied-cells N` | `max_size` | Reject ships whose occupied 2x-cell count exceeds N |
| `--require-crew-rooms` | `require_crew_rooms` | Reject ships with no crew rooms |
| `--require-reachable-reactor` | `require_reachable_reactor` | Reject ships whose crew cannot reach any reactor; requires expansion graphs |

No rules are enabled by default.  The `require_reachable_reactor` rule requires
that the input corpus contains expansion graph data
(`X_expansion_structural`); the CLI will fail fast with a clear error if
expansion data is absent.

New rules can be added by creating a module under `corpus/rules/` that
implements `CorpusRule` from `corpus/rules/base.py` and wiring a CLI flag in
`corpus/cli.py`.
