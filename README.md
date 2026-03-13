# Voidwright

An open-source algorithmic ship generation project for the game Cosmoteer.

https://github.com/user-attachments/assets/aedbaf98-fc67-45d8-b9d7-7d9fe7f4a618


## Technical overview

This repository is organized around purpose-specific packages:

- `preprocessing/`
  - local `.ship.png` inputs -> centered-2x extracted JSON -> canonical JSON -> graph JSON
- `training/`
  - backend-agnostic model training router with a Markov backend
- `generator/`
  - backend-agnostic runtime generation router with a Markov backend
- `markov/`
  - shared Markov model, training, validation, and generation internals
- `ship_layout/`
  - shared structural geometry, connectivity, mirror symmetry, and placement validation — including the `PlacementValidator` API used by all generator backends

Discord acquisition is intentionally separate from preprocessing and stays in `scripts/` as a miscellaneous operational utility.

## Documentation

Start with the curated docs in `docs/`:

- `docs/pipeline-and-artifacts.md`
- `docs/ship-graphs.md`
- `docs/markov-generator.md`
- `docs/generation-modes.md`
- `docs/door-rules.md`
- `tests/README.md` (test strategy and expectations)

Markov backend implementation notes now live in `docs/markov-generator.md` and
`generator/backends/markov/README.md`.

## Quick start

1. Create and activate a Python virtual environment
2. Install dependencies
3. Add your Discord bot token to `.env` only if you plan to use the standalone Discord download script

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

If you need the Discord download script too, install the extra inside the same
virtual environment:

```bash
pip install -e .[scripts]
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

The preprocessing stages now support hardware-agnostic parallelism. By default,
each stage picks an `auto` executor mode and worker count:

- `extract` defaults to thread-based workers
- `canonicalize` defaults to a parallel scan/hash phase plus concurrent writes
- `graphs` defaults to parallel per-ship workers
- In restricted environments where process pools are unavailable, `auto` falls back to thread-based execution instead of failing

You can override these defaults when needed:

```bash
python main.py preprocessing pipeline downloaded_ships \
  --graph-workers 1 \
  --graph-executor thread \
  --canonicalize-workers 8
```

You can also run the stages individually:

```bash
python main.py preprocessing extract downloaded_ships --output-dir extracted_ship_data
python main.py preprocessing canonicalize --input-dir extracted_ship_data --output-dir extracted_ship_data_canonical
python main.py preprocessing graphs --input-dir extracted_ship_data_canonical --output-dir generated_ship_graphs_canonical
python main.py preprocessing door-rules --input-dir extracted_ship_data_canonical
```

Each individual preprocessing stage also accepts:

- `--workers <n>`
- `--executor {auto,thread,process}`

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
- `--visualization-fps` controls MP4 playback speed when `--visualize` is enabled

### 5. Visualize generation as MP4

The Markov generator can now render one MP4 per generated sample showing the
ship grow step by step, including accepted placements and renderable rejected
attempts:

```bash
python main.py generator generate markov \
  --model models/markov/markov-model.v2.json \
  --output-dir out/generated-ships \
  --count 3 \
  --seed 1337 \
  --visualize \
  --visualization-fps 30
```

Visualization outputs are written here:

- `.ship.png` files: `out/generated-ships/`
- `.mp4` videos: `out/generated-ships/visualizations/`

Icon discovery order:

1. `--icons-root` if you want to point directly at a Terran icon directory such as `Data/ships/terran`
2. `--game-root` if you want to point at a local Cosmoteer install root
3. Windows Steam auto-discovery via the Steam registry plus `steamapps/libraryfolders.vdf` or `config/libraryfolders.vdf`
4. Repo-local fallback cache under `assets/local/cosmoteer-icons/terran/`

Example with a manual override:

```bash
python main.py generator generate markov \
  --model models/markov/markov-model.v2.json \
  --output-dir out/generated-ships \
  --count 1 \
  --visualize \
  --game-root "F:/SteamLibrary/steamapps/common/Cosmoteer"
```

If auto-discovery fails, copy the Terran part folders containing `icon.png` into
`assets/local/cosmoteer-icons/terran/`. That folder is ignored by git on
purpose, so it can be used as a local cache without polluting the repo.

## Module layout

- `common/` - shared helper modules plus shared Cosmoteer geometry and PNG parse/encode support
- `preprocessing/` - extraction, canonicalization, graph generation, and local pipeline orchestration
- `training/` - backend router and training adapters
- `generator/` - backend router and generation adapters
- `markov/` - shared Markov model, sampling, and backend input helpers; `markov/symmetry.py` is a backward-compat shim over `ship_layout/symmetry.py`
- `visualizer/` - shared generation event recording, icon loading, frame rendering, and MP4 export
- `scripts/` - miscellaneous operational utilities such as Discord acquisition
- `models/` - generated model artifacts

## Notes

- The extractor supports both `*.ship.png` and `*.ship__msg<digits>.png`
- Canonicalization is content-based and may produce `__dedup-<12 hex>` filenames when different ships want the same canonical name
- Export and parse paths preserve stored part `Location` values, including `FlipX` and `FlipY` metadata when present
- Mirror mode and visualization both apply part-specific wedge/triangle handedness rules to keep mirrored output aligned with in-game orientation
- The current Markov backend is intentionally conservative and does not synthesize doors during generation
- Door validation is vanilla-first and treats many non-vanilla situations as intentionally unresolved

## Attribution

`common/cosmoteer/parser.py` is an adapted minimal parser implementation intended
to match the extraction approach used by the `franklin050187/cosmo-api` project,
while keeping only the code needed for this repository.
