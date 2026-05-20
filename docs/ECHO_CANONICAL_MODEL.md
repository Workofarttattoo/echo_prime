# Echo canonical model and training stack

See also: [ECHO_MODEL_CARD.md](./ECHO_MODEL_CARD.md) for the full model card (weights, benchmarks, limitations).

Echo’s intelligence is **two layers**, not one giant weight file:

| Layer | What it is | Where it lives |
|--------|------------|----------------|
| **Brain** | Ollama LLM (reasoning, language, PhD personas) | `~/.ollama` — **not in git** |
| **Knowledge** | Research papers + concepts (vectorless trees) | `~/echo-kb` (ConDB + OpenKB) |
| **Legacy memory** | ~25k-paper FAISS episodic/semantic | `memory_data/`, `knowledge_base/` in full repo |

GitHub ships **code + small memory snapshots**, not 4M paper JSON or multi‑GB GGUF files.

## Canonical model (use this going forward)

**`ech0-knowledge-v4:latest`**

- Installed Ollama tag on your Mac, built on **`ech0-thinking-v2`** (~9GB).
- This is the knowledge-trained Echo line; benchmarks reference it as `ech0-knowledge-v4`.
- Set once:

```bash
export ECH0_OLLAMA_MODEL=ech0-knowledge-v4:latest
export ECH0_INFERENCE_PROVIDER=ollama
```

Older names (`ech0-unified-14b`, `ech0-unified-14b-enhanced`) map to this tag in `echo_prime/models/registry.yaml`.

## Device tiers (strip down per hardware)

| Tier | Model | Typical device |
|------|--------|----------------|
| `full` | `ech0-knowledge-v4:latest` | Mac / workstation ≥12GB RAM |
| `standard` | `ech0-fine-tuned-v2:latest` | 8GB+ |
| `lite` | `ech0-lite:latest` | Pi / laptop 4GB (`ollama pull ech0-lite` if missing) |
| `embedded` | `llama3.2:latest` | servos / tiny MCUs via gateway |

```bash
export ECHO_DEVICE_TIER=lite   # auto-detected from RAM if unset
```

## OpenAlex stream (vault)

Echo pulled works from the OpenAlex API in parallel (`massive_openalex.log`) and wrote normalized JSON:

```
echo_wisdom/
  materials_science/openalex.json   # up to ~77k works per file
  openalex.json                     # root aggregate
  */crossref.json                   # parallel source dumps
```

Each record typically has: `title`, `abstract`, `authors`, `openalex_id`, `doi`, `year`, `citations`, `source`.

The vectorless ingest script streams these with **ijson** (no full-RAM load) into ConDB PageIndex trees.

## 4M papers vs what’s in git today

- **Goal:** millions of papers on the encrypted vault (`echo_wisdom/`).
- **In repo today:** ~25,300 papers integrated into `memory_data/episodic_wisdom.npy` (see `WISDOM_INTEGRATION_STATUS.md`).
- **Vectorless path:** ingest vault → `~/echo-kb` with `scripts/ingest_wisdom_to_vectorless.py`.

```bash
# After vault is mounted
.venv/bin/python scripts/verify_echo_training.py
.venv/bin/python scripts/ingest_wisdom_to_vectorless.py
```

## How to talk to Echo (not raw Qwen)

| You used | What you get |
|----------|----------------|
| `ollama run ech0-knowledge-v4` | Old **Modelfile** identity (~939 papers, Qwen backstory). **Not** the live KB. |
| Echo Prime **daemon** `/v1/chat` with `use_rag: true` | Python stack + ConDB retrieval + Mantle system prompt (authoritative). |

```bash
# Terminal A — Echo Prime (from echo_prime/)
export ECH0_OLLAMA_MODEL=ech0-knowledge-v4:latest
export ECHO_KB_ROOT=~/echo-kb
.venv/bin/uvicorn echo_prime.daemon.app:app --host 127.0.0.1 --port 8000

# Terminal B — chat with full RAG + correct identity
curl -s http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is your knowledge base size and architecture?","use_rag":true,"session_id":"main"}' \
  | python3 -m json.tool
```

Set `ECHO_SYSTEM_PROMPT=0` only if you need the legacy baked-in Modelfile persona.

## Verify

```bash
.venv/bin/python scripts/verify_echo_training.py
```
