# Extended Materials Databases

Place curated material databases that need Materials Project validation in this
directory. The `scripts/validate-extended-materials.py` helper (relative to the
`qulab-infinite/` root) scans every
`*.db.json` file (recursively) under `data/materials/` unless you pass explicit
paths.

Supported formats:

- JSON objects whose keys are material names. Metadata keys should be prefixed
  with `_` (for example `_metadata`). Each material object can optionally
  include `name`, but the loader will fall back to the key name when missing.
- JSON arrays of material objects.
- JSON Lines (`.jsonl`) files where each line is a separate material object.

Each material entry should expose at least one Materials Project identifier so
the validator can query the remote API:

- `mp_id`, `mpId`, `material_id`, `materialsProjectId`, or `materials_project_id`
- `provenance.url` containing `https://materialsproject.org/materials/mp-####`
- Any nested `references` entry whose link includes an `mp-####` token

The validator compares any overlapping properties (e.g., density, band gap,
formation energy) against the API response. Properties that do not exist in the
input entry are skipped automatically. Files are streamed with `ijson` so even
multi-gigabyte datasets can be processed without exhausting memory.

## Working with compressed dumps

Large curated exports ship as `.jsonl.gz` archives to keep the repo lighter. To
validate those datasets, decompress them into the same directory first:

```bash
cd qulab-infinite/data/materials
gunzip -c mp-summary.jsonl.gz > mp-summary.jsonl
```

Any `.gz` file is streamed transparently by the validator, but keeping a
decompressed copy makes quick spot-checks with tools like `head`/`rg` much
faster.

## Running the validator

Once the JSONL (or decompressed copy) is in place, call the helper from the
repo root and point it at the dataset. Results land in
`logs/materials/mp-validation-report.json` by default.

```bash
cd qulab-infinite
python scripts/validate-extended-materials.py data/materials/mp-summary.jsonl
```
