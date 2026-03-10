# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo does

Generates Cosmoteer `.ship.png` files from a learned model. The full pipeline is:

1. (Optional) Download `.ship.png` files from Discord
2. Preprocess local images: extract embedded JSON → canonicalize → build ship graphs
3. Train a Markov model from graph outputs
4. Generate new encoded `.ship.png` files from the trained model

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Full preprocessing pipeline** (images → graph JSON):
```bash
python -m preprocessing.cli pipeline downloaded_ships \
  --output-dir generated_ship_graphs_canonical \
  --write-extracted-dir extracted_ship_data \
  --write-canonical-dir extracted_ship_data_canonical \
  --verbose
```

**Preprocessing stages individually:**
```bash
python -m preprocessing.cli extract downloaded_ships --output-dir extracted_ship_data
python -m preprocessing.cli canonicalize --input-dir extracted_ship_data --output-dir extracted_ship_data_canonical
python -m preprocessing.cli graphs --input-dir extracted_ship_data_canonical --output-dir generated_ship_graphs_canonical
python -m preprocessing.cli door-rules --input-dir extracted_ship_data_canonical
```

**Train a Markov model** (preferred — from graph corpus):
```bash
python -m training.cli build markov \
  --graph-input-dir generated_ship_graphs_canonical \
  --output models/markov/markov-model.v2.json
```

**Validate coordinate assumptions:**
```bash
python -m training.cli validate markov \
  --input-dir extracted_ship_data_canonical \
  --output models/markov/coordinate-validation.v2.json
```

**Generate ships:**
```bash
python -m generator.cli generate markov \
  --model models/markov/markov-model.v2.json \
  --output-dir out/generated-ships \
  --count 5 \
  --seed 1337
```

**Discord acquisition** (requires `DISCORD_BOT_TOKEN` in `.env`):
```bash
python scripts/download_ship_images.py --output-dir downloaded_ships --verbose
```

## Module architecture

The codebase is split into five purpose-specific packages, all invoked via `python -m <module>.cli`:

- **`preprocessing/`** — four-stage pipeline (extract → canonicalize → graphs → door-rules). Each stage is its own submodule with a `main(argv)` and `build_parser()`. `pipeline.py` orchestrates all stages.
- **`training/`** — backend-agnostic router. `router.py` resolves backend names; each backend under `training/backends/<name>/` registers its own CLI parser via `register_build_parser` / `register_validate_parser`.
- **`generator/`** — same router pattern as training. `generator/backends/markov/backend.py` wires CLI options; `generator/backends/markov/export.py` handles `.ship.png` encoding and roundtrip validation.
- **`markov/`** — shared Markov internals used by both training and generation: `model.py` (tokens, training, sampling), `symmetry.py` (mirror-mode logic), `inputs.py` (allowlist/seed loading).
- **`common/`** — geometry metadata (`geometry.py`), file helpers, logging, and `common/cosmoteer/` (parser and encoder for `.ship.png` LSB payloads). `common/data/vanilla-parts-from-game-files.json` is the authoritative part geometry source.

### Key data flow

```
.ship.png → parser → raw JSON → canonicalize → graph JSON → Markov model → generated JSON → encoder → .ship.png
```

### Adding a new backend

Register it in `training/router.py` and `generator/router.py` alongside the Markov backend. Implement `register_build_parser` / `run_build` (training) and `register_generate_parser` / `run_generate` (generator) following `MarkovTrainingBackend` / `MarkovGeneratorBackend` as templates.

## Important conventions

- **Geometry source of truth:** `common/data/vanilla-parts-from-game-files.json` via `common/geometry.py`. All vanilla part footprints, dimensions, and traversability come from here. Non-vanilla parts fall back to regex inference.
- **Model artifacts** live under `models/markov/`. Preferred artifact: `markov-model.v2.json` (built from graph corpus). Legacy `v1` artifacts used the raw canonical corpus.
- **Graph training is preferred** over the legacy `--input-dir` raw-corpus path. Use `--graph-input-dir` when building models.
- **Canonicalization is content-based.** Files may get `__dedup-<12 hex>` suffixes — this is normal, not a failure.
- **The Markov generator does not synthesize doors.** Door-rule logic in `preprocessing/door_rules.py` and `preprocessing/door_rules_engine.py` is for analysis and future passes only.
- **Mirror symmetry axis** is at `x = -0.5`. Left half: all footprint cells `x <= -1`; right half: `x >= 0`. Parts straddling the axis are rejected.
- **Token format:** `(part_id, rotation, anchor_part_id, anchor_rotation, dx, dy)`. Root tokens use `anchor_part_id = "__ROOT__"`. END token is `"__END__"`.
