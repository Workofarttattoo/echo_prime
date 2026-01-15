# ech0 Golden Run Acceptance Criteria

The golden run is a deterministic, reproducible demo that validates the ech0 → Qulab → ech0 handoff. It does **not** claim real-world lab output; it proves orchestration, provenance logging, and repeatability.

## Pass Criteria

1. **Deterministic seed**: the run uses a fixed seed and records it.
2. **Provenance capture**: timestamp, repo commit hash, runtime versions, and platform metadata are recorded.
3. **Input parameters**: configuration parameters are stored alongside outputs.
4. **Artifact hashing**: payload and artifact hashes are computed and reported.
5. **Cross-check**: the ech0 report hash matches the computed payload hash.
6. **Qulab linkage**: ech0 report references the Qulab artifact, and its hashes verify.

## Run

```bash
cd visualizer
npm run ech0:golden-run -- --seed 1337
```

## Verify

```bash
cd visualizer
npm run ech0:golden-verify
```

## Investor Summary

```bash
cd visualizer
npm run ech0:golden-report
```

The summary is written alongside the golden run report in `logs/ech0-golden-runs/`.

## One-Page Brief (PDF-ready)

```bash
cd visualizer
npm run ech0:golden-onepager
```

Open the generated HTML in a browser and print to PDF for a shareable one-pager.

The verifier confirms required fields are present and hashes match the artifacts. Use the resulting report and verifier output in demos for investors or reviewers.
