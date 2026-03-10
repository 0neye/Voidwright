# ship-generator

This repository builds a Cosmoteer ship corpus and a first-pass vanilla-only ship generator.

It includes tooling for:

- downloading visible Discord `.ship.png` attachments
- extracting embedded ship JSON payloads from those PNGs
- canonicalizing and deduplicating the extracted corpus
- generating structural and cell-graph analysis artifacts
- inferring reusable vanilla-first door-placement rules
- building, validating, sampling, and exporting a relative-placement Markov generator

## Documentation

Start with the curated docs in `docs/`:

- `docs/pipeline-and-artifacts.md`
- `docs/ship-graphs.md`
- `docs/markov-generator.md`
- `docs/generation-modes.md`
- `docs/door-rules.md`

Generator-specific details and CLI examples also live in `generators/markov/README.md`.

## Quick start

1. Create and activate a Python virtual environment
2. Install dependencies
3. Add your Discord bot token to `.env` if you plan to use the download step

```bash
pip install -r requirements.txt
```

```env
DISCORD_BOT_TOKEN=your_bot_token_here
```

## Main workflow

### Download and extract

Run the top-level pipeline:

```bash
python main.py --download-output-dir downloaded_ships --extract-output-dir extracted_ship_data --verbose
```

Useful flags:

- `--download-output-dir`
- `--extract-output-dir`
- `--skip-download`
- `--skip-extract`
- `--verbose`

Notes:

- `main.py` runs download first and then extraction
- the download token is only needed for the Discord step
- extraction reads from the download output directory

### Canonicalize the extracted corpus

```bash
python scripts/canonicalize_ship_json_corpus.py \
  --input-dir extracted_ship_data \
  --output-dir extracted_ship_data_canonical \
  --report-json out/ship_canonicalization_report.json
```

### Generate ship graphs

```bash
python scripts/generate_ship_graphs.py \
  --input-dir extracted_ship_data_canonical \
  --output-dir generated_ship_graphs_canonical
```

### Infer door rules

```bash
python scripts/infer_door_rules.py \
  --input-dir extracted_ship_data_canonical \
  --output generators/markov/data/door-placement-rules.v2.json
```

### Build and sample the Markov generator

Build:

```bash
python scripts/build_markov_generator.py build \
  --input-dir extracted_ship_data_canonical \
  --output out/markov/markov-model.v2.json \
  --validation-output out/markov/coordinate-validation.v2.json
```

Generate:

```bash
python scripts/build_markov_generator.py generate \
  --model out/markov/markov-model.v2.json \
  --output out/markov/samples-v2 \
  --count 5 \
  --seed 1337
```

Export existing generated samples:

```bash
python scripts/build_markov_generator.py export \
  --input-dir out/markov/samples-v2 \
  --output-dir out/markov/exported-ships \
  --report out/markov/export-report.json
```

## Key entrypoints

- `main.py` - top-level download and extract runner
- `scripts/download_ship_images.py` - Discord ship downloader with resume support
- `scripts/extract_ship_data.py` - `.ship.png` to `.json` extractor
- `scripts/canonicalize_ship_json_corpus.py` - content-hash dedupe and canonical naming
- `scripts/generate_ship_graphs.py` - graph artifact generator
- `scripts/infer_door_rules.py` - canonical-corpus door-rule inference
- `scripts/build_markov_generator.py` - thin wrapper around the Markov CLI
- `ship_parser/cosmoteer_ship_parser.py` - adapted embedded-payload ship parser
- `ship_parser/cosmoteer_ship_encoder.py` - encoder used by export tooling

## Notes

- The extractor supports both `*.ship.png` and `*.ship__msg<digits>.png`
- Canonicalization is content-based and may produce `__dedup-<12 hex>` filenames when different ships want the same canonical name
- The current generator is intentionally conservative and does not yet synthesize doors during generation
- Door validation is vanilla-first and treats many non-vanilla situations as intentionally unresolved

## Attribution

`ship_parser/cosmoteer_ship_parser.py` is an adapted minimal parser implementation intended to match the extraction approach used by the `franklin050187/cosmo-api` project, while keeping only the code needed for this repository.
