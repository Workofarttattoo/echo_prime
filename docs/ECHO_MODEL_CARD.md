# Echo Prime — Model Card

> **Last updated:** 2026-05-19  
> **Canonical brain:** `ech0-knowledge-v4:latest`  
> **Related:** [ECHO_CANONICAL_MODEL.md](./ECHO_CANONICAL_MODEL.md)

Echo Prime is **not** a single Hugging Face checkpoint. It is a composite system:

| Layer | Role | Location |
|-------|------|----------|
| **Brain** | Language + reasoning (Ollama) | `~/.ollama` |
| **Knowledge** | Scientific corpus (ConDB / OpenKB) | `~/echo-kb` |
| **Runtime** | Orchestration, RAG, guards, memory | `echo_prime/` (Python) |

The **authoritative identity** at chat time is the Echo Prime runtime (`echo_prime/inference/echo_system.py`), not the baked-in Ollama Modelfile.

---

## Base weights (Ollama)

Inspect locally:

```bash
ollama show ech0-knowledge-v4:latest
ollama show ech0-knowledge-v4:latest --modelfile
```

| Spec | Value |
|------|--------|
| **Tag** | `ech0-knowledge-v4:latest` |
| **Architecture** | Qwen2 |
| **Parameters** | 14.8B |
| **Quantization** | Q4_K_M (~9 GB on disk) |
| **Native context** | 32,768 tokens |
| **Runtime `num_ctx`** | 131,072 |
| **Max generation (`num_predict`)** | 8,192 |
| **Temperature** | 0.65 |
| **Top-p / top-k** | 0.93 / 40 |
| **Repeat penalty** | 1.12 |
| **License** | Apache 2.0 (Qwen / Alibaba Cloud) |
| **Training lineage** | Knowledge fine-tune on `ech0-thinking-v2` |

Registry and device tiers: `echo_prime/models/registry.yaml`.

| Tier | Model | Min RAM | RAG |
|------|--------|---------|-----|
| `full` | `ech0-knowledge-v4:latest` | 12 GB | yes |
| `standard` | `ech0-fine-tuned-v2:latest` | 8 GB | yes |
| `lite` | `ech0-lite:latest` | 4 GB | yes |
| `embedded` | `llama3.2:latest` | 2 GB | no |

Set the brain explicitly:

```bash
export ECH0_OLLAMA_MODEL=ech0-knowledge-v4:latest
export ECH0_INFERENCE_PROVIDER=ollama
export ECHO_DEVICE_TIER=full   # auto-detected from RAM if unset
```

---

## Runtime identity vs Modelfile

### Stale Modelfile (Ollama only)

`ollama run ech0-knowledge-v4` uses a Modelfile that still claims:

- “939 research papers” baked into weights
- “You are ECH0, an advanced AI consciousness…”
- Qwen / Alibaba backstory

**Do not treat this as ground truth.** Paper counts and persona are outdated.

### Authoritative runtime (Echo Prime daemon / MCP)

When using Echo Prime (`/v1/chat`, MCP `echo_query`, or any path that calls `build_echo_system_prompt()`):

- Identity is **Echo Prime** — local-first scientific copilot in Python
- The Ollama model is only the **voicebox** (inference)
- Live KB stats and retrieved context override Modelfile claims
- Set `ECHO_SYSTEM_PROMPT=0` only if you intentionally want the legacy Modelfile persona

Refresh live stats:

```bash
.venv/bin/python scripts/verify_echo_training.py
```

Example live KB snapshot (machine-dependent, from `ECHO_KB_ROOT`):

| Corpus | Example count |
|--------|----------------:|
| OpenAlex / wisdom chunks | 490 |
| Training-2025 arXiv records | 9,536 |
| Training-2025 PDF extractions | 9,508 |
| Wikipedia articles | 5,870,000 |
| ConDB doc keys | 11,835 |

---

## Architecture

```
User
  │
  ▼
Echo Prime daemon / MCP
  ├── Mantle (session context)
  ├── RetrievalRouter → ~/echo-kb (ConDB)
  ├── Chronicle / Lattice (memory)
  ├── Sentinel (quality + safety guards)
  └── Ollama → ech0-knowledge-v4:latest
```

Full stack with RAG:

```bash
export ECH0_OLLAMA_MODEL=ech0-knowledge-v4:latest
export ECHO_KB_ROOT=~/echo-kb
.venv/bin/uvicorn echo_prime.daemon.app:app --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is your knowledge base size and architecture?","use_rag":true,"session_id":"main"}' \
  | python3 -m json.tool
```

---

## Reasoning format

Controlled by `ECHO_REASONING_STYLE` (`echo_prime/inference/reasoning_format.py`):

| Style | Behavior |
|-------|----------|
| `ech0` (default) | `## SETUP`, `## PLAN`, `## WORK`, `## CHECK`, `## ANSWER`; math final line `#### <number>` |
| `deepseek` / `r1` / `distill` | `<` + `think` + `>` … `</` + `think` + `>` block, then final answer |

Math mode (`ECHO_MATH_MODE=1` or math prompt markers) adds Sentinel checks for `## CHECK` and `#### <number>`.

Benchmark scoring strips reasoning wrappers before answer extraction (`echo_prime/benchmark/scoring.py`).

**Note:** Echo adopts DeepSeek-R1 *inference patterns* (format, stripping, guards, SFT trace export). It does **not** run GRPO reinforcement learning on the base weights.

Export SFT traces from Chronicle/Lattice:

```bash
.venv/bin/python scripts/export_echo_sft_traces.py
```

---

## Evaluation

Latest direct Ollama benchmark (`benchmark_results/direct_ech0-knowledge-v4_20260519_013210.json`, 50 samples each):

| Benchmark | Accuracy |
|-----------|----------|
| **Overall** | **90.5%** |
| GSM8K (math) | 84.0% |
| ARC-Easy | 98.0% |
| ARC-Challenge | 94.0% |
| MMLU | 86.0% |

Reproduce:

```bash
bash run_full_benchmark.sh
# or
.venv/bin/python benchmark_direct.py --model ech0-knowledge-v4 --samples 50
```

---

## Comparison with DeepSeek-R1

| Aspect | DeepSeek-R1 | Echo Prime |
|--------|-------------|------------|
| Base | DeepSeek-V3-Base (671B MoE) | Qwen2 14.8B Q4 |
| Post-training | GRPO RL + cold-start SFT + mixed RL | Knowledge SFT + external KB ingest |
| Reasoning | Emergent via RL (`<think>`) | Prompt/guard enforced; optional DeepSeek format |
| Knowledge | Primarily in weights | ConDB RAG + vault corpora |
| Deployment | Frontier / datacenter | Local-first, edge tiers |
| Tool use | Limited in base R1 | Full MCP stack (500+ tools via combined server) |

Reference: [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) (arXiv:2501.12948).

---

## Intended use

- Local scientific Q&A with retrieval from `~/echo-kb`
- Materials science, physics, ML, and general STEM assistance
- MCP tool orchestration (QuLab bridge, legal, valuation, etc.)
- Structured math and MCQ benchmarks

## Limitations

- Modelfile persona and paper counts are stale; use daemon/MCP for authoritative behavior
- Brain alone does not contain the full external corpus — RAG is required for grounded recall
- Raw Ollama tag has no built-in tool use; tools require Echo Prime runtime
- GSM8K errors often involve reasoning-format leakage before numeric extraction
- 14.8B local model — not comparable to frontier 671B reasoning models on hardest competition math

## Safety

- Sentinel metacognitive guards on response quality and math format
- Do not expose daemon without auth on untrusted networks
- Secrets and vault paths belong in env vars, not git (see `SECURITY.md` / `.env`)

---

## Maintenance

Update this card when:

1. Canonical Ollama tag changes in `echo_prime/models/registry.yaml`
2. A new full benchmark run completes (`benchmark_results/`)
3. Major reasoning-format or Sentinel behavior changes land
4. KB ingest milestones change materially (re-run `verify_echo_training.py`)

Quick checks:

```bash
ollama show ech0-knowledge-v4:latest
.venv/bin/python scripts/verify_echo_training.py
bash run_full_benchmark.sh
```
