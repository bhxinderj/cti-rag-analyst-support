# CTI-RAG Analyst Support

Retrieval-Augmented Generation (RAG) prototype for Cyber Threat Intelligence (CTI) analyst support.

**Master's Thesis Project** — FH Technikum Wien, Information Systems Management
Author: Joben Preet Bhinder | Advisor: Wolfgang Rogner, BSc. MSc

## Overview

This prototype enables natural-language querying over CTI data using a hybrid retrieval pipeline (BM25 + vector search) combined with a locally running LLM (Llama 3.1 via Ollama). It retrieves relevant vulnerability and threat intelligence context, generates source-grounded answers with citations, and supports systematic evaluation via the RAGAS framework.

### Data Sources
- **NVD/CVE** — National Vulnerability Database (CVSS >= 7.0, 2020–2025)
- **CISA KEV** — Known Exploited Vulnerabilities catalog
- **CISA Advisories** — Cybersecurity Advisories with narrative TTP/mitigation context (included in Setup B)
- **MISP** — CIRCL OSINT feed (threat events, IOCs, ATT&CK mappings)

Corpus snapshot semantics: documents **published on or before the configured
`snapshot_date`** (default `2025-12-31`) are included; later edits to in-range
documents are retained and the fetch date is documented in the thesis. The
modification timestamp is deliberately not a drop criterion.

### Key Features
- Hybrid retrieval: BM25 (lexical) + ChromaDB (vector) with Reciprocal Rank Fusion
- Cross-encoder reranking for precision
- Source-grounded generation with transparent citations and fuzzy alias resolution for citation normalization
- Baseline comparison mode (LLM without retrieval)
- Phase-2 templated pipeline: Query Router → deterministic Fact Bundles → L1/L2/L3 templates (VulnTriage, ThreatContext, CrossSourceCompare) with Severity Signal (Triage-Ampel)
- Ablation study support across BM25, vector, and hybrid candidate-generation modes under a shared reranking stage
- Three-axis evaluation: RAGAS (faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall), Field Coverage, and a Structured Rubric LLM-as-Judge
- Evidence-based abstention: questions the corpus cannot support are declined explicitly instead of answered speculatively (no model call, sub-second)
- **Extended Analysis mode** (optional, clearly flagged): hosted generation (Claude Haiku 4.5 via OpenRouter) with analyst-assist output (prioritized mitigations, Detection & Hunting starting points, Next Checks) and deeper retrieval — question and retrieved context leave the local machine only in this mode
- Conversation follow-ups via condense-then-retrieve: elliptical questions ("and how do I mitigate it?") are rewritten into standalone queries with entity carry-over; generation never sees chat history
- **MCP server**: the deterministic fact pipeline exposed as an `cti_query_tool` for MCP clients (no LLM involved, ~1–2 s per call)
- FastAPI backend and Streamlit demo UI with a Legacy ↔ Templated pipeline toggle
- Interactive terminal mode for ad-hoc queries

## Prerequisites

### Tested Environment
- **macOS** on Apple Silicon (M1/M2/M3/M4) — tested on M4, 24GB RAM
- **Python 3.12**
- **Ollama**
- **Git**

### Expected Compatibility
The prototype is designed to be portable and is expected to also run on **Linux** and **Windows**, provided the following are available:
- Python 3.12
- Ollama
- compatible PyTorch / sentence-transformers dependencies
- sufficient memory for local embedding and reranking models

Note:
- The project has been **tested on macOS Apple Silicon**
- Linux and Windows are **expected to work**, but were **not formally validated** in the current prototype stage

## Setup

### 1. Clone and create virtual environment

#### macOS / Linux

```bash
git clone https://github.com/bhxinderj/cti-rag-analyst-support.git
cd cti-rag-analyst-support
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
git clone https://github.com/bhxinderj/cti-rag-analyst-support.git
cd cti-rag-analyst-support
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Install and start Ollama

Install Ollama for your operating system:
- macOS, Linux, Windows: https://ollama.com/download

```bash
# Start Ollama service
ollama serve

# Pull the required model (in a separate terminal)
ollama pull llama3.1:8b-instruct-q5_K_M
```

### Quickstart via Makefile (macOS / Linux)

For a reproducible local setup with the expected cached models:

```bash
make bootstrap
make download
make index SETUP=both
make smoke
make run
```

Useful targets:
- `make warmup-models` — download/cache the embedding and reranker models locally
- `make index-a` / `make index-b` / `make index-both`
- `make run QUERY="What is CVE-2024-3094?" MODE=hybrid SETUP=b`
- `make evaluate MODE=hybrid SETUP=a`
- `make ablation`
- `make smoke`
- `make e2e-smoke`
- `make e2e-eval-smoke`
- `make test` — run the full offline test suite (all `tests/test_*.py` modules)
- `make test-retrieval` — run the narrow retrieval checks only
- `make api` — start the FastAPI backend on port 8000
- `make ui` — start the Streamlit demo UI on port 8501
- `make mcp-server` — start the MCP stdio server (deterministic `cti_query_tool`)
- `make mcp-demo` — run the MCP client demo (connect, list tools, one triage lookup)

Note:
- the `Makefile` assumes `python3.12`, `ollama`, and `make` are available
- the reranker now loads from the local Hugging Face cache for reproducible offline retrieval runs
- on a fresh machine, run `make warmup-models` once before retrieval or evaluation

### Quickstart via PowerShell (Windows)

For native Windows use, prefer the PowerShell entry points instead of the `Makefile`:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Setup Both
powershell -ExecutionPolicy Bypass -File .\start.ps1 -Setup B
```

What the scripts do:
- `setup.ps1` creates `.venv`, installs dependencies, pulls the configured Ollama model when available, warms up the embedding/reranker models, downloads CTI data, and rebuilds the indexes
- `start.ps1` starts the prototype in interactive mode

Useful options:
- `.\setup.ps1 -SkipDownload -SkipIndex` — reuse existing data and indexes
- `.\setup.ps1 -Setup A` — build Setup A indexes only
- `.\setup.ps1 -Setup B` — build Setup B indexes only
- `.\setup.ps1 -Setup Both` — build Setup A and Setup B indexes
- `.\setup.ps1 -MispMaxEvents 100`
- `.\start.ps1 -Setup A -Mode bm25`
- `.\start.ps1 -Setup B -Mode hybrid`
- `.\start.ps1 -Setup B -Mode hybrid -Question "What is CVE-2024-3094?"`

Note:
- `setup.ps1` expects Python 3.12 on PATH via `py -3.12`, `python3.12`, or `python`
- `setup.ps1` expects `ollama` on PATH if you want the LLM pulled automatically
- if native Windows dependency issues appear, WSL remains a reasonable fallback, but it is not the primary path documented for this prototype

### 3. Download CTI data

```bash
# Download all sources (NVD takes ~10 min, MISP ~5 min, CISA KEV ~5 sec)
python main.py download --source cisa_kev
python main.py download --source cisa_advisories
python main.py download --source nvd
python main.py download --source misp --misp-max-events 300

# Or download everything at once:
python main.py download --source all
```

`download --source all` now includes `cisa_advisories`, so Setup B can be reproduced from the documented CLI path.

### 4. Build search indexes

```bash
python main.py index --clear
```

For setup-specific indexes:

```bash
CTI_RAG_SETUP=a python main.py index --clear
CTI_RAG_SETUP=b python main.py index --clear
```

Setup semantics:
- **Setup A** — `NVD + CISA KEV + MISP`
- **Setup B** — `NVD + CISA KEV + MISP + CISA Advisories`

Baseline mode does not use retrieval and therefore does not require reindexing.

## Usage

### Interactive mode (recommended for exploration)

```bash
python main.py interactive
```

Commands inside interactive mode:
- Type any question and press Enter
- `/baseline` — toggle baseline mode (LLM without retrieval, for comparison)
- `/mode hybrid|bm25|vector` — switch retrieval strategy
- `/quit` — exit

### Single query

```bash
python main.py query "What is CVE-2021-44228 and how has it been exploited?"
python main.py query "What ransomware campaigns exploit CISA KEV vulnerabilities?" --mode bm25
CTI_RAG_SETUP=b python main.py query "Which ATT&CK techniques are described in CISA advisories?" --mode hybrid
```

### Baseline comparison (no retrieval)

```bash
python main.py baseline "What is CVE-2021-44228?"
```

### Phase-2 templated pipeline

The Phase-2 pipeline routes each question through a deterministic Query Router into one of three templates — `VulnTriage`, `ThreatContext`, or `CrossSourceCompare` — and renders the answer from a typed Fact Bundle with an inline Severity Signal (Triage-Ampel).

Enable it via the `--templated` flag on `evaluate` and `rubric`:

```bash
CTI_RAG_SETUP=b python main.py evaluate --mode hybrid --templated
CTI_RAG_SETUP=b python main.py rubric --live --mode hybrid --templated
```

The Streamlit UI (`make ui`) exposes a Legacy ↔ Templated toggle for interactive comparison. The FastAPI backend (`make api`) serves both pipelines via its query endpoint.

### Extended Analysis mode (hosted, clearly flagged)

The UI's "☁️ Extended Analysis" toggle switches generation to a hosted model
(Claude Haiku 4.5 via OpenRouter) over the otherwise unchanged fact pipeline,
with an analyst-assist instruction layer (prioritized mitigations, Detection &
Hunting starting points, Next Checks) and deeper retrieval (top-8). In this
mode — and only in this mode — the question and the retrieved context leave
the local machine; the UI states this explicitly. Requires `OPENROUTER_API_KEY`
in the API server environment. This mode is **not part of the evaluated
configuration**; the local-vs-hosted trade-off is quantified by the hosted
ablation row (V5) in the thesis. The API accepts `"extended": true` on
`/api/query`; `CTI_RAG_LLM_PROVIDER=openrouter` applies the same generator to
CLI runs without editing `settings.yaml`.

Follow-up questions are supported in all modes via condense-then-retrieve:
an elliptical follow-up is rewritten into a standalone query (local mode
condenses locally), shown in the UI as "Interpreted as: …". Generation never
receives conversation history.

### MCP server

The deterministic portion of the pipeline (router → hybrid retrieval →
fact bundle → severity → triage card) is exposed as an MCP stdio server with
one tool, `cti_query_tool` — no LLM involved, ~1–2 s per call:

```bash
make mcp-server   # start the stdio server (CTI_RAG_SETUP=b)
make mcp-demo     # client demo: connect, list tools, one triage lookup
```

### Run RAGAS evaluation

```bash
python main.py evaluate
CTI_RAG_SETUP=a python main.py evaluate --mode hybrid
CTI_RAG_SETUP=b python main.py evaluate --mode hybrid
CTI_RAG_SETUP=b python main.py evaluate --mode hybrid --templated
```

### Run the Structured Rubric evaluator

The Rubric evaluator is an LLM-as-Judge scoring layer orthogonal to RAGAS. It can either rescore an existing RAGAS artifact offline or run the pipeline live and score in one pass:

```bash
# Rescore an existing RAGAS run offline
python main.py rubric --from-ragas-artifact data/evaluation_results/setup_b/ragas_hybrid_templated_final_v1_r1_<ts>.json

# Or run live and score in one pass (Phase-2 templated)
CTI_RAG_SETUP=b python main.py rubric --live --mode hybrid --templated --name my_experiment
```

The judge model is configured in `configs/settings.yaml` under
`evaluation.ragas` (`eval_llm_provider` + `eval_llm_openrouter`; default
`openrouter:openai/gpt-4o-mini`) and is shared by the RAGAS and Rubric
evaluators. With the default provider, `evaluate` and `rubric` require
`OPENROUTER_API_KEY` in the environment.

### Reproducing the thesis evaluation (`_final_v1`)

The final evaluation series reported in the thesis was produced on the state
tagged **`evaluated-v1`**:

```bash
git checkout evaluated-v1
caffeinate -is bash scripts/run_final_evals.sh   # ~15 h: 3x RAGAS per row, rubric, field coverage
```

Supporting tooling:
- `scripts/eval_to_tex_table.py` — renders the artifact groups to LaTeX tables (`thesis/tables/`)
- `scripts/aggregate_final_evals.py` — mean±std, NaN-aware effective n, zero counts, per-template rubric join
- `requirements-evaluated-v1.lock` — exact package versions of the evaluation environment
- All artifacts of the series are committed under `data/evaluation_results/setup_b/*final_v1*`

### Run smoke checks

```bash
make smoke
```

The smoke target runs CLI help plus narrow repository tests that verify snapshot handling, setup-aware indexing, retrieval trace behavior, and evaluation artifact structure without requiring external services.

### Run a live local end-to-end smoke check

```bash
make e2e-smoke
```

This check is intentionally preflighted. It fails fast unless all of the following are already available locally:
- `ollama serve` is reachable
- the configured Ollama generation model exists locally
- Setup B retrieval artifacts exist locally
- embedding and reranker models exist in the local Hugging Face cache

If those prerequisites hold, it runs one real baseline query and one real RAG query against the local stack.

For a heavier one-sample local evaluation run on top of the same preflight:

```bash
make e2e-eval-smoke
```

`e2e-eval-smoke` intentionally uses a reduced smoke metric set (`answer_relevancy` by default) so the live evaluation path stays fast and less timeout-prone. It is a runtime sanity check, not a replacement for the full evaluation run.

### Run ablation study

```bash
python main.py ablation
```

The retrieval ablation compares three candidate-generation strategies with the same final reranking stage:
- `bm25` — BM25 candidate generation + shared reranker
- `vector` — vector candidate generation + shared reranker
- `hybrid` — BM25 + vector + RRF fusion + shared reranker

## Project Structure

```
cti-rag-analyst-support/
├── main.py                          # CLI entry point
├── configs/
│   ├── settings.yaml                # All configurable parameters
│   └── eval_queries.yaml            # Evaluation query set (RAGAS + Rubric)
├── src/cti_rag/
│   ├── ingestion/
│   │   ├── models.py                # CTIDocument Pydantic schema
│   │   ├── downloader.py            # NVD, CISA, MISP data download
│   │   ├── nvd_parser.py            # NVD JSON → CTIDocument
│   │   ├── cisa_parser.py           # CISA KEV/Advisory → CTIDocument
│   │   └── misp_parser.py           # MISP events → CTIDocument
│   ├── retrieval/
│   │   ├── indexer.py               # ChromaDB + BM25 index builder
│   │   └── hybrid_retriever.py      # Hybrid search + RRF + reranking
│   ├── rag/
│   │   ├── chain.py                 # RAG chain (legacy + templated) + generation-provider adapter
│   │   ├── prompts.py               # Legacy Phase-1 prompt templates
│   │   ├── router.py                # Phase-2 Query Router
│   │   ├── facts.py                 # Deterministic Fact Bundles + fuzzy alias resolver
│   │   ├── templates.py             # L1/L2/L3 templates (VulnTriage, ThreatContext, CrossSourceCompare)
│   │   ├── severity.py              # Severity Signal (Triage-Ampel)
│   │   ├── cpe.py                   # CPE parser for affected-product extraction
│   │   ├── extended.py              # Extended-Analysis mode (hosted analyst-assist, flagged)
│   │   └── conversation.py          # Condense-then-retrieve follow-up wrapper
│   ├── evaluation/
│   │   ├── ragas_eval.py            # RAGAS framework integration
│   │   ├── rubric_eval.py           # Structured Rubric LLM-as-Judge evaluator
│   │   ├── field_coverage.py        # Field-coverage evaluator
│   │   └── _judge_llm.py            # Judge-model adapter (Ollama / OpenAI / OpenRouter)
│   ├── api/
│   │   ├── server.py                # FastAPI backend
│   │   └── schemas.py               # Request/response models
│   ├── ui/
│   │   └── app.py                   # Streamlit demo UI (pipeline + Extended toggles)
│   └── mcp/
│       ├── server.py                # MCP stdio server (deterministic cti_query_tool)
│       └── client_demo.py           # Minimal MCP client (thesis transcript figure)
├── scripts/
│   ├── run_final_evals.sh           # _final_v1 batch runner (thesis evaluation series)
│   ├── eval_to_tex_table.py         # Artifacts → LaTeX tables (thesis/tables/)
│   └── aggregate_final_evals.py     # mean±std, effective n, per-template rubric join
├── tests/                           # Offline test suite (test_*.py) + smoke runners
├── data/
│   ├── raw/                         # Downloaded CTI data (not in git)
│   ├── processed/                   # Normalized documents
│   ├── indexes/                     # ChromaDB + BM25 indexes (setup_a / setup_b)
│   └── evaluation_results/          # RAGAS + Rubric artifacts (final series committed)
├── requirements.txt                 # Dependency ranges
└── requirements-evaluated-v1.lock   # Exact versions of the evaluation environment
```

## Configuration

All parameters are centralized in `configs/settings.yaml` for reproducibility:
- LLM model and inference settings
- Embedding model selection
- Embedding device selection (`cpu`, `cuda`, or `mps` depending on host system)
- Retrieval parameters (top-k, RRF constant, reranking)
- Data source filters (CVSS threshold, year range, global `snapshot_date` cutoff on publication/addition date)
- RAGAS evaluation metrics

Environment-based experiment setup:
- leave `CTI_RAG_SETUP` unset for the default single-index layout
- set `CTI_RAG_SETUP=a` for Setup A
- set `CTI_RAG_SETUP=b` for Setup B

## Runtime Assumptions

- Python `3.12` is the supported interpreter. `.python-version` pins the expected major/minor version for local env managers.
- `ollama serve` must be running at `http://localhost:11434` before `query`, `baseline`, `interactive`, or `evaluate`.
- Retrieval and evaluation expect the embedding model and reranker to be present locally. Run `make warmup-models` once on a fresh machine.
- Runtime retrieval/indexing now expects the embedding model to resolve from the local Hugging Face cache instead of silently attempting a network fetch.
- `download` and `warmup-models` require network access. Re-running `index`, `query`, `baseline`, and `make smoke` does not.
- `evaluate` and `rubric` use a hosted judge by default (`openrouter:openai/gpt-4o-mini`) and therefore require `OPENROUTER_API_KEY` and network access; the same key enables the optional Extended Analysis mode. Query/baseline/interactive runs stay fully local.
- The default embedding device in `configs/settings.yaml` is `mps`. On non-Apple-Silicon hosts, switch it to `cpu` or `cuda`.

BM25 persistence:
- BM25 artifacts are stored as transparent JSON inputs and reconstructed at load time
- setup-specific artifacts live under `data/indexes/setup_a/` and `data/indexes/setup_b/`

Recommended embedding device values:
- **Apple Silicon macOS**: `mps`
- **Linux/Windows with NVIDIA GPU**: `cuda`
- **CPU-only systems**: `cpu`

Important:
- `mps` should only be used on supported Apple Silicon systems
- on unsupported systems, use `cpu` or `cuda` instead

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| LLM (generation, primary) | Llama 3.1 8B (Q5_K_M) via Ollama — fully local |
| LLM (judge, evaluation only) | gpt-4o-mini via OpenRouter |
| LLM (Extended mode / hosted ablation) | Claude Haiku 4.5 via OpenRouter — optional, flagged |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector Store | ChromaDB (embedded, local) |
| Lexical Search | rank_bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Orchestration | LangChain |
| Evaluation | RAGAS + Structured Rubric (LLM-as-Judge) + Field Coverage |
| API / UI | FastAPI + Streamlit |
| Agent integration | MCP (Model Context Protocol) stdio server |
| Compute Backend | PyTorch via CPU, CUDA, or Apple Metal (MPS), depending on host system |

## License

This project is part of an academic thesis. All code is open-source.
