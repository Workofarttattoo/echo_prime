# Community Benchmarking Plan and Status

Last audited: 2026-02-10

## Quick answer

No confirmed submission artifacts were found for LM Arena or other major community benchmark boards in this repository. The prior `online_benchmark_submission.py` workflow was mostly placeholder logic and did not implement reliable external onboarding.

## Priority community benchmark targets

These are the best external targets for broad model credibility and community visibility:

1. **LM Arena (LMSYS / LM Arena)**  
   - URL: <https://lmarena.ai/>
   - Why: strongest community signal for head-to-head chat preference.

2. **Hugging Face Open LLM Leaderboard**  
   - URL: <https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard>
   - Why: most referenced open leaderboard for reproducible public model comparisons.

3. **LiveBench**  
   - URL: <https://livebench.ai/>
   - Why: contamination-resistant benchmark with frequent refreshes.

4. **AlpacaEval 2.0**  
   - URL: <https://tatsu-lab.github.io/alpaca_eval/>
   - Why: common pairwise preference metric used in release reporting.

5. **LM Evaluation Harness compatibility track**  
   - URL: <https://github.com/EleutherAI/lm-evaluation-harness>
   - Why: baseline ecosystem standard used by many benchmark pipelines.

## New onboarding workflow

Generate a complete submission packet with one command:

```bash
python3 online_benchmark_submission.py --target all --announce
```

Optional explicit results file:

```bash
python3 online_benchmark_submission.py \
  --target all \
  --results-file path/to/results.json \
  --model-id ech0prime/ech0-prime-csa \
  --announce
```

The command writes a packet under:

```text
benchmark_results/community_submissions/<timestamp>_<model>/
```

Each packet includes:

- `submission_manifest.json`
- `targets/<target>/submission_payload.json`
- `targets/<target>/README.md` (checklist + action steps)
- `announcement_template.md` (if `--announce` is set)

## Execution checklist

- [ ] Confirm model card + public endpoint are ready.
- [ ] Generate fresh benchmark outputs and packet.
- [ ] Submit in target priority order (LM Arena first).
- [ ] Record confirmation links and listing URLs in this file.

## Status tracker

| Target | Status | Owner | Evidence Link |
|---|---|---|---|
| LM Arena | Pending submission | Unassigned | TBD |
| HF Open LLM Leaderboard | Pending submission | Unassigned | TBD |
| LiveBench | Pending submission | Unassigned | TBD |
| AlpacaEval 2.0 | Pending submission | Unassigned | TBD |
| LM Eval Harness Track | Pending run | Unassigned | TBD |
