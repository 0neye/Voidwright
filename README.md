# ship-generator

This repository is organized around three purpose-specific modules:

- `preprocessing/`
  - local `.ship.png` inputs -> extracted JSON -> canonical JSON -> graph JSON
- `training/`
  - backend-agnostic model training router with a Markov backend
- `generator/`
  - backend-agnostic runtime generation router with a Markov backend

Discord acquisition is intentionally separate from preprocessing and stays in `scripts/` as a miscellaneous operational utility.

## Documentation

Start with the curated docs in `docs/`:

- `docs/pipeline-and-artifacts.md`
- `docs/ship-graphs.md`
- `docs/markov-generator.md`
- `docs/generation-modes.md`
- `docs/door-rules.md`

Markov backend implementation notes now live in `docs/markov-generator.md` and
`generator/backends/markov/README.md`.

## Quick start

1. Create and activate a Python virtual environment
2. Install dependencies
3. Add your Discord bot token to `.env` only if you plan to use the standalone Discord download script

```bash
pip install -r requirements.txt
```

```env
DISCORD_BOT_TOKEN=your_bot_token_here
```

## Unified root CLI (new)

The repository now includes a root entry point at `main.py` that delegates to:

- `preprocessing`
- `training`
- `generator`

You can use it to discover commands, inspect help, and run an interactive REPL:

```bash
python main.py commands
python main.py help
python main.py help training build markov
python main.py repl
```

All existing package-level CLIs (`python -m preprocessing.cli`, `python -m training.cli`, and `python -m generator.cli`) remain supported.

## Main workflow

### 1. Optional Discord acquisition

Download visible Discord `.ship.png` attachments into a local folder:

```bash
python scripts/download_ship_images.py --output-dir downloaded_ships --verbose
```

This step is optional. Everything else starts from local `.ship.png` files or directories.

### 2. Run preprocessing

Run the full preprocessing pipeline from local ship images to canonical graph JSON outputs:

```bash
python main.py preprocessing pipeline downloaded_ships \
  --output-dir generated_ship_graphs_canonical \
  --write-extracted-dir extracted_ship_data \
  --write-canonical-dir extracted_ship_data_canonical \
  --verbose
```

You can also run the stages individually:

```bash
python main.py preprocessing extract downloaded_ships --output-dir extracted_ship_data
python main.py preprocessing canonicalize --input-dir extracted_ship_data --output-dir extracted_ship_data_canonical
python main.py preprocessing graphs --input-dir extracted_ship_data_canonical --output-dir generated_ship_graphs_canonical
python main.py preprocessing door-rules --input-dir extracted_ship_data_canonical
```

### 3. Train a model

Train the Markov backend from preprocessing outputs:

```bash
python main.py training build markov \
  --graph-input-dir generated_ship_graphs_canonical \
  --output models/markov/markov-model.v2.json
```

If you want the legacy raw-corpus validation pass too:

```bash
python main.py training build markov \
  --input-dir extracted_ship_data_canonical \
  --output models/markov/markov-model.v2.json \
  --validation-output models/markov/coordinate-validation.v2.json
```

### 4. Generate encoded ships

Generate finished `.ship.png` outputs with the backend-agnostic generator CLI:

```bash
python main.py generator generate markov \
  --model models/markov/markov-model.v2.json \
  --output-dir out/generated-ships \
  --count 5 \
  --seed 1337
```

Optional diagnostics:

- `--json-output-dir` writes the generated JSON payloads alongside the exported ships
- `--seed-json` or `--seed-png` seeds generation from an existing layout
- `--mirror-symmetry`, `--allowlist`, and `--requirements-file` preserve the existing Markov runtime options

## Module layout

- `common/` - shared helper modules plus shared Cosmoteer geometry and PNG parse/encode support
- `preprocessing/` - extraction, canonicalization, graph generation, and local pipeline orchestration
- `training/` - backend router and training adapters
- `generator/` - backend router and generation adapters
- `markov/` - shared Markov model, sampling, symmetry, and backend input helpers
- `scripts/` - miscellaneous operational utilities such as Discord acquisition
- `models/` - generated model artifacts

## Notes

- The extractor supports both `*.ship.png` and `*.ship__msg<digits>.png`
- Canonicalization is content-based and may produce `__dedup-<12 hex>` filenames when different ships want the same canonical name
- Export and parse paths preserve stored part `Location` values, including `FlipX` and `FlipY` metadata when present
- The current Markov backend is intentionally conservative and does not synthesize doors during generation
- Door validation is vanilla-first and treats many non-vanilla situations as intentionally unresolved

## Attribution

`common/cosmoteer/parser.py` is an adapted minimal parser implementation intended
to match the extraction approach used by the `franklin050187/cosmo-api` project,
while keeping only the code needed for this repository.
