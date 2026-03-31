# CTI-RAG Analyst Support

Retrieval-Augmented Generation (RAG) prototype for Cyber Threat Intelligence (CTI) analyst support.

**Master's Thesis Project** — FH Technikum Wien, Information Systems Management
Author: Joben Preet Bhinder | Advisor: Wolfgang Rogner, BSc. MSc

## Overview

This prototype enables natural-language querying over CTI data using a hybrid retrieval pipeline (BM25 + vector search) combined with a locally running LLM (Llama 3.1 via Ollama). It retrieves relevant vulnerability and threat intelligence context, generates source-grounded answers with citations, and supports systematic evaluation via the RAGAS framework.

### Data Sources
- **NVD/CVE** — National Vulnerability Database (CVSS >= 7.0, 2020–2026; filtered by the configured `snapshot_date`)
- **CISA KEV** — Known Exploited Vulnerabilities catalog (entries added on or before the configured `snapshot_date`)
- **MISP** — CIRCL OSINT feed (threat events, IOCs, ATT&CK mappings; events published on or before the configured `snapshot_date`)

### Key Features
- Hybrid retrieval: BM25 (lexical) + ChromaDB (vector) with Reciprocal Rank Fusion
- Cross-encoder reranking for precision
- Source-grounded generation with transparent citations
- Baseline comparison mode (LLM without retrieval)
- Ablation study support across BM25, vector, and hybrid candidate-generation modes under a shared reranking stage
- RAGAS evaluation (faithfulness, context precision/recall, answer relevancy)
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
make index
make run
```

Useful targets:
- `make warmup-models` — download/cache the embedding and reranker models locally
- `make run QUERY="What is CVE-2024-3094?" MODE=hybrid`
- `make evaluate MODE=hybrid`
- `make ablation`
- `make test-retrieval`

Note:
- the `Makefile` assumes `python3.12`, `ollama`, and `make` are available
- the reranker now loads from the local Hugging Face cache for reproducible offline retrieval runs
- on a fresh machine, run `make warmup-models` once before retrieval or evaluation

### Quickstart via PowerShell (Windows)

For native Windows use, prefer the PowerShell entry points instead of the `Makefile`:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

What the scripts do:
- `setup.ps1` creates `.venv`, installs dependencies, pulls the configured Ollama model when available, warms up the embedding/reranker models, downloads CTI data, and rebuilds the indexes
- `start.ps1` starts the prototype in interactive mode

Useful options:
- `.\setup.ps1 -SkipDownload -SkipIndex` — reuse existing data and indexes
- `.\setup.ps1 -MispMaxEvents 100`
- `.\start.ps1 -Mode bm25`
- `.\start.ps1 -Mode hybrid -Question "What is CVE-2024-3094?"`

Note:
- `setup.ps1` expects Python 3.12 on PATH via `py -3.12`, `python3.12`, or `python`
- `setup.ps1` expects `ollama` on PATH if you want the LLM pulled automatically
- if native Windows dependency issues appear, WSL remains a reasonable fallback, but it is not the primary path documented for this prototype

### 3. Download CTI data

```bash
# Download all sources (NVD takes ~10 min, MISP ~5 min, CISA KEV ~5 sec)
python main.py download --source cisa_kev
python main.py download --source nvd
python main.py download --source misp --misp-max-events 300

# Or download everything at once:
python main.py download --source all
```

### 4. Build search indexes

```bash
python main.py index --clear
```

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
```

### Baseline comparison (no retrieval)

```bash
python main.py baseline "What is CVE-2021-44228?"
```

### Run RAGAS evaluation

```bash
python main.py evaluate
```

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
│   └── eval_queries.yaml            # RAGAS evaluation query set
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
│   │   ├── chain.py                 # LCEL RAG chain with citations
│   │   └── prompts.py               # Prompt templates
│   ├── evaluation/
│   │   └── ragas_eval.py            # RAGAS framework integration
│   └── mcp/                         # (planned) MCP agent extension
├── data/
│   ├── raw/                         # Downloaded CTI data (not in git)
│   ├── processed/                   # Normalized documents
│   └── indexes/                     # ChromaDB + BM25 indexes
└── requirements.txt
```

## Configuration

All parameters are centralized in `configs/settings.yaml` for reproducibility:
- LLM model and inference settings
- Embedding model selection
- Embedding device selection (`cpu`, `cuda`, or `mps` depending on host system)
- Retrieval parameters (top-k, RRF constant, reranking)
- Data source filters (CVSS threshold, year range, global `snapshot_date` cutoff on publication/addition date)
- RAGAS evaluation metrics

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
| LLM | Llama 3.1 8B (Q5_K_M) via Ollama |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector Store | ChromaDB (embedded, local) |
| Lexical Search | rank_bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Orchestration | LangChain (LCEL) |
| Evaluation | RAGAS |
| Compute Backend | PyTorch via CPU, CUDA, or Apple Metal (MPS), depending on host system |

## License

This project is part of an academic thesis. All code is open-source.
